from loggers.base.logger_base import LoggerBase
import torch
from torchvision.utils import make_grid
from torchvision.transforms import Resize, InterpolationMode
from utils.vit_util import unpatchify_from_1d
from datasets.transforms import transform_from_kwargs, transform_collate_fn
from utils.factory import create_collection


class ReconstructionLogger(LoggerBase):

    def __init__(self, x_transform=None, denormalize_pixels=True, **kwargs):
        super().__init__(**kwargs)
        self.samples = None
        self.reconstructions = None
        self.mask = None
        self.x_transform = create_collection(x_transform, transform_from_kwargs, collate_fn=transform_collate_fn)
        self.denormalize_pixels = denormalize_pixels

    # noinspection PyMethodOverriding
    def _track_after_accumulation_step(self, update_outputs, **kwargs):
        self.samples = update_outputs.get('x')
        self.reconstructions = update_outputs.get('x_hat')
        self.mask = update_outputs.get('mask')

    # noinspection PyMethodOverriding
    def _log_after_update(self, update_counter, model, **_):
        if self.every_n_updates is None:
            return
        self._log_reconstruction(update_counter, model)

    # noinspection PyMethodOverriding
    def _log_after_epoch(self, update_counter, model, **_):
        if self.every_n_epochs is None:
            return
        self._log_reconstruction(update_counter, model)

    def _log_reconstruction(self, update_counter, model):
        if self.samples is not None:
            samples = self.x_transform(self.samples[:5, ...])

            mask = self.mask[:5, ...]
            mask_h, mask_w = tuple(
                input_shape // patch_size
                for input_shape, patch_size
                in zip(model.input_shape[-2:], model.encoder.patch_size)
            )
            reshaped_mask = mask.reshape(-1, 1, mask_h, mask_w).repeat_interleave(3, dim=1)
            resized_mask = Resize(samples.shape[-2:], interpolation=InterpolationMode.NEAREST)(reshaped_mask)

            reconstructions = self.reconstructions[:5, ...]
            reconstructions = unpatchify_from_1d(
                patches=reconstructions,
                patch_size=model.encoder.patch_size,
                img_shape=model.input_shape
            )
            reconstructions = self.x_transform(reconstructions)

            if self.denormalize_pixels:
                normalization_dims = tuple(i for i in range(1, reconstructions.ndim))
                reconstructions_min = torch.amin(reconstructions, dim=normalization_dims, keepdim=True)
                reconstructions_max = torch.amax(reconstructions, dim=normalization_dims, keepdim=True)
                reconstructions = (reconstructions - reconstructions_min) / (reconstructions_max - reconstructions_min)

            masked_reconstructions = reconstructions.clone() * resized_mask
            masked_input = samples * (1 - resized_mask)

            merged = torch.cat((samples, resized_mask, masked_input, masked_reconstructions, reconstructions), dim=2)
            merged = make_grid([x for x in merged])
            merged = merged.detach().cpu().numpy()
            merged = merged.transpose(1, 2, 0)

            self.writer.add_image(
                key="reconstruction/test",
                data=dict(
                    data_or_path=merged,
                    caption='from top to bottom: input, mask (white = reconstruct), masked input, masked reconstruction, full reconstruction'
                ),
                update_counter=update_counter
            )
