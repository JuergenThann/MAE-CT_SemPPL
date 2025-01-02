import numpy as np
import torch
import torch.nn as nn

from initializers.trunc_normal_initializer import TruncNormalInitializer
from models.heads.linear_head import LinearHead
from models.poolings import pooling_from_kwargs
from utils.model_utils import update_ema, copy_params


class EmaLinearHead(LinearHead):
    def __init__(self, target_factor=0.0, **kwargs):
        super().__init__(**kwargs)
        self.target_factor = target_factor
        self.target_head = LinearHead(**kwargs)
        for param in self.target_head.parameters():
            param.requires_grad = False

    def load_state_dict(self, state_dict, strict=True):
        # patch for stage2
        if len([k for k in state_dict.keys() if k.startswith("target_head.")]) == 0:
            for key in list(state_dict.keys()):
                state_dict[f"target_head.{key}"] = state_dict[key]
        super().load_state_dict(state_dict=state_dict, strict=strict)

    def _model_specific_initialization(self):
        copy_params(self, self.target_head)

    def forward(self, x, target_x=None, view=None):
        result = {}

        if x is not None:
            result['logits'] = super().forward(x, view=view)
        if target_x is not None:
            result['target_logits'] = self.target_head(target_x, view=view)

        return result

    def after_update_step(self):
        update_ema(self, self.target_head, self.target_factor)

    def features(self, x):
        result = self.forward(x, target_x=None) if self.training else self.forward(None, target_x=x)
        if len(result) == 1:
            return next(iter(result.values()))
        else:
            raise NotImplementedError()

    def predict(self, x):
        return dict(main=self.features(x))

    def predict_binary(self, x):
        return self.predict(x)
