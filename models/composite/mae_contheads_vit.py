import kappaprofiler as kp
import torch.nn as nn

from models import model_from_kwargs
from utils.factory import create_collection
from .mae_vit import MaeVit
from models.vit.mask_generators.random_mask_generator import RandomMaskGenerator


class MaeContheadsVit(MaeVit):
    def __init__(self, contrastive_heads=None, decoder=None, **kwargs):
        super().__init__(decoder=decoder, **kwargs)
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

        if self.decoder is None:
            del sub["decoder"]
        return sub

    # noinspection PyMethodOverriding
    def forward(self, x, mask_generator, batch_size):
        outputs = super().forward(x, mask_generator=mask_generator)
        latent_tokens = outputs["latent_tokens"]
        target_latent_tokens = outputs.get("target_latent_tokens", None)
        outputs.update(self.forward_heads(latent_tokens=latent_tokens, target_latent_tokens=target_latent_tokens,
                                          batch_size=batch_size))
        return outputs

    def forward_heads(self, latent_tokens, target_latent_tokens, batch_size):
        outputs = {}
        view_count = int(len(latent_tokens) / batch_size)
        for head_name, head in self.contrastive_heads.items():
            outputs[head_name] = {}
            # seperate forward pass because of e.g. BatchNorm
            with kp.named_profile_async(head_name):
                for view in range(view_count):
                    start_idx = view * batch_size
                    end_idx = (view + 1) * batch_size
                    head_outputs = head(
                        x=latent_tokens[start_idx:end_idx],
                        target_x=None if target_latent_tokens is None else target_latent_tokens[start_idx:end_idx],
                        view=view
                    )
                    outputs[head_name][f"view{view}"] = head_outputs
        return outputs

    def predict(self, x):
        outputs = self(x, mask_generator=RandomMaskGenerator(mask_ratio=0.0), batch_size=x.shape[0])
        flat_outputs = dict({
            k1: v3
            for k1, v1 in outputs.items()
            if k1 in self.contrastive_heads.keys() and "view0" in v1
            for _, v2 in v1.items()
            if len(v2) == 1
            for _, v3 in v2.items()
        })
        return flat_outputs
