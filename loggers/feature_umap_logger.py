from functools import partial

import torch
from kappadata import ModeWrapper
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
            exclude_negative_labels=True,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.extractors = create_collection(extractors, extractor_from_kwargs)
        self.forward_kwargs = objects_from_kwargs(forward_kwargs)
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric

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
        return features, classes.clone()

    # noinspection PyMethodOverriding
    def _log(self, update_counter, model, trainer, logger_info_dict, **_):
        # setup
        for extractor in self.extractors:
            extractor.enable_hooks()

        # source_dataset foward (this is the "queue" from the online nn_accuracy)
        features, y = self.iterate_over_dataset_collated(
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
                u = (u - mean) / std

                forward_kwargs_str = f"/{dict_to_string(self.forward_kwargs)}" if len(self.forward_kwargs) > 0 else ""
                feature_key_bn = f"{feature_key}-batchnorm" if batch_normalize else feature_key
                key = (
                    f"umap_{self.n_components}d/"
                    f"{feature_key_bn}/"
                    f"{self.n_neighbors}neighbors_{self.min_dist}mindist_{self.metric}/"
                    f"{self.dataset_key}"
                    f"{forward_kwargs_str}"
                )
                column_headers = [f"d{i}" for i in range(self.n_components)] + ["label"]

                dataset = self.data_container.get_dataset(self.dataset_key)
                if isinstance(dataset, TorchvisionDatasetWrapper):
                    dataset_wrapper = dataset
                    y = list([dataset_wrapper.dataset.classes[y_] for y_ in y])

                self.writer.add_scatterplot(key, column_headers, [u_ + [y_] for u_, y_ in zip(u.tolist(), y)],
                                            update_counter)
                logger_info_dict[key] = u

        # cleanup
        for extractor in self.extractors:
            extractor.disable_hooks()
