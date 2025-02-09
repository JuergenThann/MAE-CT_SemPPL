from loggers.base.dataset_logger import DatasetLogger
import torch
from torchvision.utils import make_grid
from torchvision.transforms import Resize, InterpolationMode
from utils.vit_util import unpatchify_from_1d
from kappadata.wrappers.dataset_wrappers.subset_wrapper import SubsetWrapper


class SampleLogger(DatasetLogger):

    def __init__(self, indices, **kwargs):
        super().__init__(**kwargs)
        self.indices = indices

    def get_dataset_mode(self, trainer):
        return trainer.dataset_mode

    # noinspection PyMethodOverriding
    def _log_after_update(self, update_counter, model, **_):
        if self.every_n_updates is None:
            return
        self._log_samples(update_counter, model)

    # noinspection PyMethodOverriding
    def _log_after_epoch(self, update_counter, model, **_):
        if self.every_n_epochs is None:
            return
        self._log_samples(update_counter, model)

    def _log_samples(self, update_counter, model):
        batches = self.iterate_over_dataset(
            forward_fn=lambda b: b,
            update_counter=update_counter,
        )

        for idcs, samples, classes in batches:
            samples = samples.detach().cpu().numpy()
            samples = samples.transpose(0, 2, 3, 1)

            for i in range(samples.shape[0]):
                idx = idcs[i]
                if idx not in self.indices:
                    continue
                sample = samples[i]
                self.writer.add_image(
                    key=f"sample/idx{idx:05d}",
                    data=dict(
                        data_or_path=sample,
                        caption=f'Sample with index {idx:05d}, class {classes[i]}, class name {self.dataset.class_names[classes[i]]}'
                    ),
                    update_counter=update_counter
                )
