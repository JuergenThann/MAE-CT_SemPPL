from functools import partial

import torch
from kappadata.wrappers import ModeWrapper

from distributed.config import is_rank0
from models.extractors import extractor_from_kwargs
from utils.factory import create_collection
from .base.multi_dataset_logger import MultiDatasetLogger


class OfflineFeatureLogger(MultiDatasetLogger):
    def __init__(self, dataset_key, extractors, **kwargs):
        super().__init__(**kwargs)
        self.dataset_key = dataset_key
        self.extractors = create_collection(extractors, extractor_from_kwargs)
        # create output folder
        self.out_folder = self.stage_path_provider.stage_output_path / "features"
        self.out_folder.mkdir(exist_ok=True)

    def _before_training(self, model, **kwargs):
        for extractor in self.extractors:
            extractor.register_hooks(model)
            extractor.disable_hooks()

    def _forward(self, batch, model, trainer, dataset):
        features = {}
        with trainer.autocast_context:
            trainer.forward(model=model, batch=batch, dataset=dataset)
            for extractor in self.extractors:
                features[str(extractor)] = extractor.extract().cpu()
        batch, _ = batch  # remove ctx
        classes = ModeWrapper.get_item(mode=trainer.dataset_mode, item="class", batch=batch)
        return features, classes.clone()

    # noinspection PyMethodOverriding
    def _log(self, update_counter, model, trainer, train_dataset, **_):
        # assert trainer.precision == torch.float32
        # setup
        for extractor in self.extractors:
            extractor.enable_hooks()

        # extract
        dataset = self.data_container.datasets[self.dataset_key]
        features, labels = self.iterate_over_dataset(
            forward_fn=partial(self._forward, model=model, trainer=trainer, dataset=dataset),
            dataset_key=self.dataset_key,
            dataset_mode=trainer.dataset_mode,
            batch_size=trainer.effective_batch_size,
            update_counter=update_counter,
            persistent_workers=False,
        )

        # log
        for feature_name, feature in features.items():
            fname = f"{self.dataset_key}-{feature_name}-{update_counter.cur_checkpoint}-features.th"
            if is_rank0():
                torch.save(feature, self.out_folder / fname)
            self.logger.info(f"wrote features to {fname}")
        torch.save(labels, self.out_folder / f"{self.dataset_key}-labels.th")

        # cleanup
        for extractor in self.extractors:
            extractor.disable_hooks()
