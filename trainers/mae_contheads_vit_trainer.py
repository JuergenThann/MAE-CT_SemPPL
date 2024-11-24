from functools import partial

import kappaprofiler as kp
import torch
import re
from datasets.transforms import transform_from_kwargs, transform_collate_fn

from datasets.sample_wrappers.multi_view_wrapper import MultiViewWrapper
from schedules import schedule_from_kwargs
from utils.factory import create_collection
from .mae_vit_trainer import MaeVitTrainer

from utils.logging_util import log_from_all_ranks
from distributed.config import get_rank

class MaeContheadsVitTrainer(MaeVitTrainer):
    def __init__(self, transforms=None, transforms_schedule=None, **kwargs):
        super().__init__(**kwargs)
        if transforms is not None:
            self.transforms = [
                create_collection(transform, transform_from_kwargs, collate_fn=transform_collate_fn)
                for transform in transforms
            ]
        else:
            self.transforms = None
        self.transforms_schedule = schedule_from_kwargs(transforms_schedule, update_counter=self.update_counter)

    @property
    def dataset_mode(self):
        return "index x class"

    def forward(self, model, batch, train_dataset, mask_generator=None):
        outputs = {}

        (idx, x, y), ctx = batch
        outputs["idx"] = idx.to(model.device, non_blocking=True)
        outputs["y"] = y.to(model.device, non_blocking=True)
        outputs["mask_ratio"] = 0.0

        n_views_overall = 1
        if isinstance(x, list):
            n_views_overall = len(x)
            shape_groups = []
            shape = None
            for item in x:
                if item.shape != shape:
                    shape = item.shape
                    shape_groups.append([])
                shape_groups[-1].append(item)
        else:
            shape_groups = [x]


        view_start_idx = 0
        for i, x in enumerate(shape_groups):
            shape_outputs = {}
            # patch KDMultiViewWrapper
            n_views = 1
            if isinstance(x, list):
                n_views = len(x)
                train_dataset.n_views = n_views
                train_dataset.to_concat_view = MultiViewWrapper.to_concat_view
                train_dataset.to_split_view = partial(MultiViewWrapper.to_split_view, train_dataset)
                # push first to GPU...stacking on CPU takes longer
                with kp.named_profile_async("to_device"):
                    x = [item.to(model.device, non_blocking=True) for item in x]
                x = torch.stack(x, dim=1)
            else:
                with kp.named_profile_async("to_device"):
                    x = x.to(model.device, non_blocking=True)

            # augmentation warmup
            if self.transforms is not None:
                assert x.ndim == 5
                # scale augmentation strength
                if self.transforms_schedule is not None:
                    scale = self.transforms_schedule.get_value(self.update_counter.cur_checkpoint)
                    for transform in self.transforms:
                        transform.scale_strength(scale)
                    shape_outputs["transform_scale"] = scale

                samples = []
                # use float32 to avoid "RuntimeError: "reflection_pad2d_out_template" not implemented for 'BFloat16'"
                with torch.autocast(str(model.device).replace(":0", ""), dtype=torch.float32):
                    for sample in x:
                        samples.append(torch.stack([
                            transform(view)
                            for view, transform
                            in zip(sample, self.transforms[view_start_idx:view_start_idx+n_views])
                        ]))
                    x = torch.stack(samples)
                # patch MultiViewWrapper properties
                train_dataset.n_views = n_views
                train_dataset.to_concat_view = MultiViewWrapper.to_concat_view
                train_dataset.to_split_view = partial(MultiViewWrapper.to_split_view, train_dataset)

            # get batch_size (x.shape is [batch_size, n_views, ...]
            batch_size = len(x)

            # change x from [batch_size, n_views, ...] -> [n_views * batch_size, ...]
            if x.ndim == 5:
                x = MultiViewWrapper.to_concat_view(x)

            # for calculating the loss for logging, a mask generator has to be provided in order to be deterministic
            mask_generator = mask_generator or self.mask_generator

            with kp.named_profile_async("forward"):
                shape_outputs.update(model(x, mask_generator=mask_generator, batch_size=batch_size))
            outputs["mask_ratio"] = outputs["mask_ratio"] + mask_generator.mask_ratio * n_views / n_views_overall

            shape_outputs["x"] = x
            if "view0" in ctx.keys():
                for view_name in [f"view{view_idx}" for view_idx in range(view_start_idx, view_start_idx + n_views)]:
                    shape_outputs.update({f"ctx.{view_name}.{k}": v for k, v in ctx[view_name].items()})
            shape_outputs.update({f"ctx.{k}": v for k, v in ctx.items() if not re.match(r'view\d+$', k)})


            view_start_idx += n_views

            if len(shape_groups) > 1:
                outputs[f"transform{i}"] = shape_outputs
            else:
                outputs.update(shape_outputs)
        return outputs

    def get_loss(self, outputs, model):
        if "transform0" in outputs:
            outputs_ = [v for k, v in outputs.items() if k.startswith("transform")]
        else:
            outputs_ = [outputs]

        idx = outputs["idx"]
        y = outputs["y"]

        losses = {}
        loss_outputs = dict(classes=y, **outputs)

        total_loss = 0.0
        for i, shape_outputs in enumerate(outputs_):
            mae_losses_, mae_outputs_ = super().get_loss(shape_outputs, model)
            mae_outputs_["latent_tokens"] = shape_outputs["latent_tokens"]

            transform_postfix = f"/transform{i}" if len(outputs_) > 1 else ""
            loss_outputs.update({f"{k}{transform_postfix}": v for k, v in mae_outputs_.items()})
            mae_total_loss = mae_losses_.pop("total")
            for loss_name, loss in mae_losses_.items():
                if loss_name in losses:
                    losses[loss_name] = losses[loss_name] + loss
                else:
                    losses[loss_name] = loss

            all_total_losses = []
            for head_name, head in model.contrastive_heads.items():
                shape_outputs[head_name].update({k: v for k, v in shape_outputs.items() if k.startswith('ctx.')})
                shape_outputs[head_name]["shape_idx"] = i
                head_losses, head_outputs = head.get_loss(shape_outputs[head_name], idx=idx, y=y)
                all_total_losses.append(head_losses.pop("total"))
                for loss_name, head_loss in head_losses.items():
                    loss_key = f"{head_name}/{loss_name}"
                    if loss_key in losses:
                        losses[loss_key] = losses[loss_key] + head_loss
                    else:
                        losses[loss_key] = head_loss

                for output_name, head_output in head_outputs.items():
                    loss_outputs[f"{head_name}/{output_name}{transform_postfix}"] = head_output

            total_loss += mae_total_loss + sum(all_total_losses)
        return dict(total=total_loss, **losses), loss_outputs
