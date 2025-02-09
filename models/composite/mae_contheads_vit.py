import kappaprofiler as kp
import torch.nn as nn

from models import model_from_kwargs
from utils.factory import create_collection
from .mae_vit import MaeVit
from models.vit.mask_generators.random_mask_generator import RandomMaskGenerator


class MaeContheadsVit(MaeVit):
    def __init__(self, contrastive_heads=None, decoder=None, predict_head=None, **kwargs):
        super().__init__(decoder=decoder, **kwargs)
        self.predict_head = predict_head
        if contrastive_heads is not None:
            self.contrastive_heads = create_collection(
                contrastive_heads,
                model_from_kwargs,
                stage_path_provider=self.stage_path_provider,
                update_counter=self.update_counter,
                input_shape=self.encoder.output_shape,
            )
            self.contrastive_heads = nn.ModuleDict(self.contrastive_heads)
        else:
            self.contrastive_heads = {}

    @property
    def submodels(self):
        sub = super().submodels
        sub.update({f"head.{key}": value for key, value in self.contrastive_heads.items()})
        return sub

    # noinspection PyMethodOverriding
    def forward(self, x, mask_generator, batch_size, views=None, dataset_key=None):
        outputs = super().forward(x, mask_generator=mask_generator)
        latent_tokens = outputs["latent_tokens"]
        target_latent_tokens = outputs.get("target_latent_tokens", None)
        outputs.update(self.forward_heads(latent_tokens=latent_tokens, target_latent_tokens=target_latent_tokens,
                                          batch_size=batch_size, views=views, dataset_key=dataset_key))
        return outputs

    def forward_heads(self, latent_tokens, target_latent_tokens, batch_size, views=None, dataset_key=None):
        outputs = {}
        view_count = len(views) if views is not None else int(len(latent_tokens) / batch_size)
        for head_name, head in self.contrastive_heads.items():
            outputs[head_name] = {}
            # seperate forward pass because of e.g. BatchNorm
            with kp.named_profile_async(head_name):
                for view_idx in range(view_count):
                    if (
                        head.views_to_consume is not None
                        and (
                            dataset_key not in head.views_to_consume
                            or views[view_idx] not in head.views_to_consume[dataset_key]
                        )
                    ):
                        continue

                    start_idx = view_idx * batch_size
                    end_idx = (view_idx + 1) * batch_size
                    head_outputs = head(
                        x=latent_tokens[start_idx:end_idx],
                        target_x=None if target_latent_tokens is None else target_latent_tokens[start_idx:end_idx],
                        view=view_idx
                    )
                    outputs[head_name][f"view{view_idx}"] = head_outputs
        return outputs

    def predict(self, x, views=None, dataset_key=None):
        outputs = self(x, mask_generator=RandomMaskGenerator(mask_ratio=0.0), batch_size=x.shape[0], views=views, dataset_key=dataset_key)
        relevant_heads = {
            k: v
            for k, v
            in self.contrastive_heads.items()
            if self.predict_head is None or self.predict_head == k
        }
        flat_outputs = dict({
            k1: v3
            for k1, v1 in outputs.items()
            if k1 in relevant_heads.keys() and "view0" in v1
            for _, v2 in v1.items()
            if len(v2) == 1
            for _, v3 in v2.items()
        })
        if len(flat_outputs) == 0:
            raise ValueError("Forward of model is expected to have only one output for prediction.")
        return flat_outputs
