from models.heads.ema_linear_head import EmaLinearHead

from losses.fixmatch_loss import fixmatch_loss_fn


class FixmatchHead(EmaLinearHead):
    def __init__(self, threshold: float, unsupervised_loss_weight: float, teacher_pseudo_labeling: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.unsupervised_loss_weight = unsupervised_loss_weight
        self.teacher_pseudo_labeling = teacher_pseudo_labeling

    def forward(self, x, target_x=None, view=None):
        assert view is not None and 0 <= view < 2

        if self.training:
            if view == 0:
                if self.teacher_pseudo_labeling:
                    logits = super().forward(x, target_x=target_x, view=view)
                    return dict(weak_logits=logits["logits"], weak_target_logits=logits["target_logits"])
                else:
                    logits = super().forward(x, view=view)
                    return dict(weak_logits=logits["logits"])
            elif view == 1:
                logits = super().forward(x, view=view)
                return dict(strong_logits=logits["logits"])
        else:
            logits = super().forward(None, target_x=target_x)
            if view == 0:
                return dict(weak_target_logits=logits["target_logits"])
            elif view == 1:
                raise NotImplementedError

    def get_loss(self, outputs, idx, y):
        if self.training:
            weak_logits = outputs["view0"]["weak_logits"]
            strong_logits = outputs["view1"]["strong_logits"]
            if self.teacher_pseudo_labeling:
                pseudo_label_logits = outputs["view0"]["weak_target_logits"]
            else:
                pseudo_label_logits = outputs["view0"]["weak_logits"]
        else:
            weak_logits = outputs["view0"]["weak_target_logits"]
            strong_logits = None
            pseudo_label_logits = None

        loss, output = fixmatch_loss_fn(
            strong_logits,
            weak_logits,
            pseudo_label_logits,
            y,
            self.threshold,
            self.unsupervised_loss_weight
        )
        return dict(total=loss), output
