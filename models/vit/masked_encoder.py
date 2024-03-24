import einops
import torch
from torch import nn
import math

from models.poolings.single_pooling import SinglePooling
from .vit_mae import VitMae


class MaskedEncoder(VitMae):
    # noinspection PyMethodOverriding
    def forward(self, x, mask_generator, single_mask=False):
        if mask_generator is None:
            return super().forward(x)

        _, _, w, h = x.shape
        if isinstance(self.patch_embed.patch_size, tuple):
            w = w // self.patch_embed.patch_size[0]
            h = h // self.patch_embed.patch_size[1]
        else:
            w = w // self.patch_embed.patch_size
            h = h // self.patch_embed.patch_size

        # embed patches
        x = self.patch_embed(x)

        # add pos embed
        x = x + self.interpolate_pos_encoding(x, w, h)

        # undo patch_embed flattening
        # (patch_embed is set to flatten in order to not need to unflatten in inference/without mask)
        x = einops.rearrange(x, "b (h w) d -> b d h w", h=h, w=w)

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = mask_generator.get_mask(x, single_mask=single_mask)

        # append cls token
        if self.cls_token is not None:
            cls_token = einops.repeat(self.cls_token, "1 n_tokens dim -> bs n_tokens dim", bs=len(x))
            x = torch.cat((cls_token, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1]
        N = self.pos_embed.shape[1]
        if npatch == N and w == h:
            return self.pos_embed
        # class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed  # [:, 1:]
        dim = x.shape[-1]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w, h = w + 0.1, h + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w / math.sqrt(N), h / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w) == patch_pos_embed.shape[-2] and int(h) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    # noinspection PyMethodOverriding
    def features(self, x, pool_kind=None, mask_generator=None, single_mask=False):
        if mask_generator is not None:
            encoded, _, _ = self(x, mask_generator=mask_generator, single_mask=single_mask)
            return SinglePooling.get_pool_fn(kind=pool_kind, model=self)(encoded)
        return super().features(x, pool_kind=pool_kind)
