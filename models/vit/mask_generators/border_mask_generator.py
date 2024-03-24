import torch
from torch.nn.functional import interpolate

from utils.param_checking import to_2tuple
from .base.mask_generator import MaskGenerator


class BorderMaskGenerator(MaskGenerator):
    def __init__(self, mask_size=None, **kwargs):
        super().__init__(**kwargs)
        if mask_size is not None:
            self.mask_size = to_2tuple(mask_size)
            assert isinstance(self.mask_size[0], int) and self.mask_size[0] > 0
            assert isinstance(self.mask_size[1], int) and self.mask_size[1] > 0
        else:
            self.mask_size = None

    def __str__(self):
        mask_size_str = "" if self.mask_size is None else f"mask_size=({self.mask_size[0]},{self.mask_size[1]})"
        return f"{type(self).__name__}({mask_size_str}{self._base_param_str})"

    def generate_noise(self, x, generator=None):
        assert generator is None
        N, _, H, W = x.shape
        noise = torch.ones(N, 4, 4, device=x.device)
        noise[:, 1:3, 1:3] = 0
        if self.mask_size is None:
            noise = interpolate(noise.unsqueeze(1), size=(H, W), mode="bilinear").squeeze(1)
        else:
            # make sure "adjacent" patches are masked together by assigning them the same noise
            assert H % self.mask_size[0] == 0, f"{H} % {self.mask_size[0]} != 0"
            assert W % self.mask_size[1] == 0, f"{W} % {self.mask_size[1]} != 0"
            mask_h = int(H / self.mask_size[0])
            mask_w = int(W / self.mask_size[1])
            noise = interpolate(noise.unsqueeze(1), size=(mask_h, mask_w), mode="bilinear").squeeze(1)
            noise = interpolate(noise.unsqueeze(1), size=(H, W), mode="nearest").squeeze(1)

        return noise
