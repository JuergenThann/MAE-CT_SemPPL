from torch.nn.functional import normalize

from losses.semppl_loss import semppl_loss_fn
from models.heads.ema_linear_head import EmaLinearHead
import torch

from losses.fixmatch_loss import fixmatch_loss_fn


class FixmatchHead(EmaLinearHead):
    def __init__(self, threshold: float, unsupervised_loss_weight: float, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.unsupervised_loss_weight = unsupervised_loss_weight

    def forward(self, x, target_x=None, view=None):
        assert view is not None and 0 <= view < 2

        if self.training:
            logits = super().forward(x)
            if view == 0:
                return dict(logits_weak=logits["logits"])
            elif view == 1:
                return dict(logits_strong=logits["logits"])
        else:
            logits = super().forward(None, target_x)
            if view == 0:
                return dict(logits_weak=logits["target_logits"])
            elif view == 1:
                return dict(logits_strong=logits["target_logits"])

    def get_loss(self, outputs, idx, y):
        logits_weak = outputs["view0"]["logits_weak"]
        logits_strong = outputs.get("view1", {}).get("logits_strong", None)
        loss, output = fixmatch_loss_fn(logits_strong, logits_weak, y, self.threshold, self.unsupervised_loss_weight)
        return dict(total=loss), output
