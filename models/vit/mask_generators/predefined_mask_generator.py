import einops
import torch
from torch.nn.functional import interpolate

from utils.param_checking import to_2tuple
from .base.mask_generator import MaskGenerator


class PredefinedMaskGenerator(MaskGenerator):
    def __init__(self, predefined_masks, original_mask_generator):
        kwargs = {
            'mask_ratio': original_mask_generator.mask_ratio,
            'mask_ratio_schedule': original_mask_generator.mask_ratio_schedule,
            'seed': original_mask_generator.seed,
            'single_mask_seed': original_mask_generator.single_mask_seed,
            'update_counter': original_mask_generator.update_counter
        }
        super().__init__(**kwargs)
        self.mask_size = original_mask_generator.mask_size
        self.predefined_masks = predefined_masks

    def __str__(self):
        mask_size_str = "" if self.mask_size is None else f"mask_size=({self.mask_size[0]},{self.mask_size[1]})"
        return f"{type(self).__name__}({mask_size_str}{self._base_param_str})"

    def generate_noise(self, x, generator=None):
        _, _, H, W = x.shape
        mask = einops.rearrange(self.predefined_masks, "N (H W) -> N H W", H=H, W=W)
        return mask.to(x.device)
