import torch
from models.heads.ema_linear_head import EmaLinearHead

from losses.fixmatch_loss import fixmatch_loss_fn


class FixmatchHead(EmaLinearHead):
    def __init__(self, threshold: float, unsupervised_loss_weight: float, teacher_pseudo_labeling: bool,
                 strong_augmentation_for_labeled: bool, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.unsupervised_loss_weight = unsupervised_loss_weight
        self.teacher_pseudo_labeling = teacher_pseudo_labeling
        self.strong_augmentation_for_labeled = strong_augmentation_for_labeled

    def forward(self, x, target_x=None, view=None):
        assert view is None or 0 <= view < 2

        if self.training:
            if view is None or view == 0:
                if self.teacher_pseudo_labeling:
                    logits = super().forward(x, target_x=target_x, view=view)
                    return logits
                else:
                    logits = super().forward(x, view=view)
                    return dict(logits=logits["logits"])
            elif view == 1:
                logits = super().forward(x, view=view)
                return dict(logits=logits["logits"])
        else:
            if view is None or view == 0:
                logits = super().forward(None, target_x=target_x)
                return dict(target_logits=logits["target_logits"])
            elif view == 1:
                raise NotImplementedError

    def get_loss(self, outputs, idx, y):
        if self.training:
            if "view1" in outputs:
                if self.strong_augmentation_for_labeled:
                    labeled_logits = outputs["view1"]["logits"]
                else:
                    labeled_logits = outputs["view0"]["logits"]
                unlabeled_logits = outputs["view1"]["logits"]
            else:
                labeled_logits = outputs["view0"]["logits"]
                unlabeled_logits = outputs["view0"]["logits"]
            if 'ctx.confidence' in outputs and not torch.all(torch.isnan(outputs['ctx.confidence'])):
                confidence = outputs['ctx.confidence'].to(self.device)
                pseudo_label_logits = y
                y = outputs['ctx.dominant_gt'].to(self.device)
            else:
                confidence = None
                if self.teacher_pseudo_labeling:
                    pseudo_label_logits = outputs["view0"]["target_logits"]
                else:
                    pseudo_label_logits = outputs["view0"]["logits"]
        else:
            labeled_logits = outputs["view0"]["target_logits"]
            unlabeled_logits = None
            confidence = None
            pseudo_label_logits = None

        loss, output = fixmatch_loss_fn(
            unlabeled_logits,
            labeled_logits,
            pseudo_label_logits,
            confidence,
            y,
            self.threshold,
            self.unsupervised_loss_weight
        )
        return dict(total=loss), output
