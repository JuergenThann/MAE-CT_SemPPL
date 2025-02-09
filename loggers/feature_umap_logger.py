from functools import partial

import torch
from kappadata.wrappers.mode_wrapper import ModeWrapper
from torcheval.metrics.functional import binary_auprc
from torchmetrics.functional.classification import binary_auroc

import torch.nn.functional as F
from loggers.base.dataset_logger import DatasetLogger
from metrics.functional.knn import knn_metrics
from models.extractors import extractor_from_kwargs
from utils.factory import create_collection
from utils.formatting_util import dict_to_string
from utils.object_from_kwargs import objects_from_kwargs
from datasets.torchvision_dataset_wrapper import TorchvisionDatasetWrapper
from itertools import compress

import numpy as np
import umap


class FeatureUmapLogger(DatasetLogger):
    def __init__(
            self,
            extractors,
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric='euclidean',
            forward_kwargs=None,
            num_samples_to_render=None,
            classes_to_render=None,
            class_names_to_render=None,
            **kwargs,
    ):
        assert classes_to_render is None or class_names_to_render is None
        super().__init__(**kwargs)
        self.extractors = create_collection(extractors, extractor_from_kwargs)
        self.forward_kwargs = objects_from_kwargs(forward_kwargs)
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.num_samples_to_render = num_samples_to_render
        self.class_names_to_render = class_names_to_render
        self.classes_to_render = classes_to_render

    def get_dataset_mode(self, trainer):
        return trainer.dataset_mode

    @property
    def return_ctx(self):
        return True

    def _before_training_impl(self, model, **kwargs):
        for extractor in self.extractors:
            extractor.register_hooks(model)
            extractor.disable_hooks()

    def _forward(self, batch, model, trainer):
        features = {}
        with trainer.autocast_context:
            trainer.forward(model, batch, self.dataset, **self.forward_kwargs)
            for extractor in self.extractors:
                features[str(extractor)] = extractor.extract().cpu()
        batch, _ = batch  # remove ctx
        classes = ModeWrapper.get_item(mode=trainer.dataset_mode, item="class", batch=batch)
        indices = ModeWrapper.get_item(mode=trainer.dataset_mode, item="index", batch=batch)
        return features, classes.clone(), indices.clone()

    # noinspection PyMethodOverriding
    def _log(self, update_counter, model, trainer, logger_info_dict, **_):
        # setup
        for extractor in self.extractors:
            extractor.enable_hooks()

        # source_dataset foward (this is the "queue" from the online nn_accuracy)
        features, y, idx = self.iterate_over_dataset_collated(
            forward_fn=partial(self._forward, model=model, trainer=trainer),
            update_counter=update_counter
        )

        # take only 1st view of the source features (just like NNCLR does it)
        features = {k: v[:len(y)] for k, v in features.items()}
        for feature_key in features.keys():
            x = features[feature_key].to(model.device)
            y = y.to(model.device)
            assert len(x) == len(y), "expecting single view input"

            x = F.normalize(x, dim=-1)

            # calculate
            for batch_normalize in [False]:  # [False, True]:
                if batch_normalize:
                    mean = x.mean(dim=0)
                    std = x.std(dim=0) + 1e-5
                    x = (x - mean) / std

                fit = umap.UMAP(
                    n_neighbors=self.n_neighbors,
                    min_dist=self.min_dist,
                    n_components=self.n_components,
                    metric=self.metric
                )
                u = fit.fit_transform(x.detach().cpu().numpy())
                u = torch.from_numpy(u).to(model.device)

                mean = u.mean(dim=0)
                std = u.std(dim=0) + 1e-5
                u = ((u - mean) / std).tolist()

                forward_kwargs_str = f"/{dict_to_string(self.forward_kwargs)}" if len(self.forward_kwargs) > 0 else ""
                feature_key_bn = f"{feature_key}-batchnorm" if batch_normalize else feature_key
                key = (
                    f"umap_{self.n_components}d/"
                    f"{feature_key_bn}/"
                    f"{self.n_neighbors}neighbors_{self.min_dist}mindist_{self.metric}/"
                    f"{self.dataset_key}"
                    f"{forward_kwargs_str}"
                )
                column_headers = [f"d{i}" for i in range(self.n_components)] + ["label", "idx"]

                if self.classes_to_render is not None:
                    class_mask = [y_ in self.classes_to_render for y_ in y]
                    y = list(compress(y, class_mask))
                    u = list(compress(u, class_mask))
                    idx = list(compress(idx, class_mask))

                dataset = self.data_container.get_dataset(self.dataset_key)
                if isinstance(dataset.root_dataset, TorchvisionDatasetWrapper):
                    y = list([dataset.root_dataset.dataset.classes[y_] for y_ in y])
                    if self.class_names_to_render is not None:
                        class_name_mask = list([y_ in self.class_names_to_render for y_ in y])
                        y = list(compress(y, class_name_mask))
                        u = list(compress(u, class_name_mask))
                        idx = list(compress(idx, class_name_mask))
                else:
                    assert self.class_names_to_render is None, 'class_names_to_render is only supported for TorchvisionDatasetWrapper'

                if self.num_samples_to_render is not None:
                    y_old = y
                    u_old = u
                    y = []
                    u = []
                    counter = {y_: 0 for y_ in set(y_old)}
                    for i in range(len(y_old)):
                        y_ = y_old[i]
                        if counter[y_] >= self.num_samples_to_render:
                            continue
                        counter[y_] += 1
                        u.append(u_old[i])
                        y.append(y_)

                self.writer.add_scatterplot(key, column_headers, [u_ + [y_, idx_] for u_, y_, idx_ in zip(u, y, idx)],
                                            update_counter)
                logger_info_dict[key] = u

        # cleanup
        for extractor in self.extractors:
            extractor.disable_hooks()
