import numpy as np
import torch

from kappadata.wrappers.mode_wrapper import ModeWrapper
from torch.nn.functional import softmax
from kappadata.utils.one_hot import to_one_hot_matrix
from .model_aware_batch_wrapper import ModelAwareBatchWrapper


# modified copy of kappadata kd_mix_collator
class ProbPseudoMixBatchWrapper(ModelAwareBatchWrapper):
    """
    apply_mode:
    - "batch": apply either all samples in the batch or don't apply
    - "sample": decide for each sample whether or not to apply mixup/cutmix
    lamb_mode:
    - "batch": use the same lambda/bbox for all samples in the batch
    - "sample": sample a lambda/bbox for each sample
    shuffle_mode:
    - "roll": mix sample 0 with sample 1; sample 1 with sample 2; ...
    - "flip": mix sample[0] with sample[-1]; sample[1] with sample[-2]; ... requires even batch_size
    - "random": mix each sample with a randomly drawn other sample
    """

    def __init__(
            self,
            teacher_pseudo_labeling: bool = True,
            prediction_head_name: str = None,
            mixup_alpha: float = None,
            cutmix_alpha: float = None,
            mixup_p: float = 0.5,
            cutmix_p: float = 0.5,
            shuffle_mode: str = "flip",
            seed: int = None,
            label_smoothing: float = 0.0,
            n_classes: int = None,
            weak_augmentation_index: int = None,
            supervised_mixup_mode: str = None,
            unsupervised_mixup_mode: str = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        # check probabilities
        assert isinstance(mixup_p, (int, float)) and 0. <= mixup_p <= 1., f"invalid mixup_p {mixup_p}"
        assert isinstance(cutmix_p, (int, float)) and 0. <= cutmix_p <= 1., f"invalid cutmix_p {cutmix_p}"
        assert 0. < mixup_p + cutmix_p <= 1., f"0 < mixup_p + cutmix_p <= 1 (got {mixup_p + cutmix_p})"
        assert isinstance(label_smoothing, (int, float)) and 0. <= label_smoothing < 1.
        assert label_smoothing == 0 or n_classes is not None
        assert supervised_mixup_mode in [None, 'Mixup']
        assert unsupervised_mixup_mode in [None, 'Mixup', 'ProbPseudoMixup']
        if mixup_p + cutmix_p != 1.:
            raise NotImplementedError

        # check alphas
        if mixup_p == 0.:
            assert mixup_alpha is None
        else:
            assert isinstance(mixup_alpha, (int, float)) and 0. < mixup_alpha
        if cutmix_p == 0.:
            assert cutmix_alpha is None
        else:
            assert isinstance(cutmix_alpha, (int, float)) and 0. < cutmix_alpha

        # check modes
        assert shuffle_mode in ["roll", "flip", "random"], f"invalid shuffle_mode {shuffle_mode}"

        # initialize
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_p = mixup_p
        self.cutmix_p = cutmix_p
        self.shuffle_mode = shuffle_mode
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        self.prediction_head_name = prediction_head_name
        self.label_smoothing = label_smoothing
        self.n_classes = n_classes
        self.weak_augmentation_index = weak_augmentation_index
        self.teacher_pseudo_labeling = teacher_pseudo_labeling
        self.supervised_mixup_mode = supervised_mixup_mode
        self.unsupervised_mixup_mode = unsupervised_mixup_mode

    @property
    def total_p(self) -> float:
        return self.mixup_p + self.cutmix_p

    def __call__(self, batch, dataset_mode, **kwargs):
        assert self.model is not None

        ctx = kwargs.get('ctx')
        return_ctx = ctx is not None

        # extract properties from batch
        idx, x, y = None, None, None
        if ModeWrapper.has_item(mode=dataset_mode, item="index"):
            idx = ModeWrapper.get_item(mode=dataset_mode, item="index", batch=batch)
        if ModeWrapper.has_item(mode=dataset_mode, item="x"):
            x = ModeWrapper.get_item(mode=dataset_mode, item="x", batch=batch)
        if ModeWrapper.has_item(mode=dataset_mode, item="class"):
            y = ModeWrapper.get_item(mode=dataset_mode, item="class", batch=batch)
        batch_size = len(x)

        assert y.ndim == 1, 'Expecting class indices, because ground truth is required'
        gt = y.clone()

        labeled_map = y >= 0
        unlabeled_map = torch.logical_not(labeled_map).clone()
        num_labeled = torch.sum(labeled_map.to(torch.int)).item()
        num_unlabeled = batch_size - num_labeled

        if self.weak_augmentation_index is not None:
            x_pseudo = x[:, self.weak_augmentation_index, ...]
        else:
            x_pseudo = x

        x_l = x[labeled_map, ...]
        x_u = x[unlabeled_map, ...]
        y_l = self.apply_label_smoothing(y[labeled_map], x.dtype)
        gt_l = gt[labeled_map]
        gt_u = gt[unlabeled_map]

        if self.supervised_mixup_mode == 'Mixup':
            x_l, y_l, use_cutmix_l, lamb_l, confidence_l, dominant_gt_l = self.apply_mixup(x_l, y_l, gt_l, None, self.supervised_mixup_mode)
        else:
            use_cutmix_l = torch.full(size=(num_labeled,), fill_value=False)
            lamb_l = torch.full(size=(num_labeled,), fill_value=torch.nan, dtype=x.dtype)
            confidence_l = torch.full(size=(num_labeled,), fill_value=torch.nan, dtype=x.dtype)
            dominant_gt_l = gt_l

        if self.unsupervised_mixup_mode is not None:
            confidence_u, y_pseudo = self.get_pseudo_labels(x_pseudo[unlabeled_map])
            y_pseudo = self.apply_label_smoothing(y_pseudo, x.dtype)
            if self.unsupervised_mixup_mode == 'Mixup':
                # transfered from original code, see
                # https://github.com/amazon-science/semi-vit/blob/4785cbc7c7e642649eb202c4be2342fb1b87ee3d/engine_semi.py#L113
                # when normal mixup is used for pseudo labeled samples, the confidence mask is not updated
                x_u, y_u, use_cutmix_u, lamb_u, _, dominant_gt_u = self.apply_mixup(x_u, y_pseudo, gt_u, confidence_u, self.unsupervised_mixup_mode)
            elif self.unsupervised_mixup_mode == 'ProbPseudoMixup':
                x_u, y_u, use_cutmix_u, lamb_u, confidence_u, dominant_gt_u = self.apply_mixup(x_u, y_pseudo, gt_u, confidence_u, self.unsupervised_mixup_mode)
            else:
                raise NotImplementedError
        else:
            use_cutmix_u = torch.full(size=(num_unlabeled,), fill_value=False)
            lamb_u = torch.full(size=(num_unlabeled,), fill_value=torch.nan, dtype=x.dtype)
            confidence_u = torch.full(size=(num_unlabeled,), fill_value=torch.nan, dtype=x.dtype)
            dominant_gt_u = gt_u
            y_u = -self.apply_label_smoothing(-gt_u-1, x.dtype)

        x[labeled_map, ...] = x_l
        x[unlabeled_map, ...] = x_u

        if y.ndim == 1:
            y = torch.empty(batch_size, self.n_classes, dtype=x.dtype)
        y[labeled_map, ...] = y_l
        y[unlabeled_map, ...] = y_u

        use_cutmix = torch.empty(batch_size, dtype=torch.bool)
        use_cutmix[labeled_map] = use_cutmix_l
        use_cutmix[unlabeled_map] = use_cutmix_u

        lamb = torch.empty(batch_size, dtype=x.dtype)
        lamb[labeled_map] = lamb_l
        lamb[unlabeled_map] = lamb_u

        confidence = torch.empty(batch_size, dtype=x.dtype)
        confidence[labeled_map] = confidence_l
        confidence[unlabeled_map] = confidence_u

        dominant_gt = torch.empty(batch_size, dtype=gt.dtype)
        dominant_gt[labeled_map] = dominant_gt_l
        dominant_gt[unlabeled_map] = dominant_gt_u

        # update properties in batch
        if idx is not None:
            batch = ModeWrapper.set_item(mode=dataset_mode, item="index", batch=batch, value=idx)
        if x is not None:
            batch = ModeWrapper.set_item(mode=dataset_mode, item="x", batch=batch, value=x)
        if y is not None:
            batch = ModeWrapper.set_item(mode=dataset_mode, item="class", batch=batch, value=y)

        if return_ctx:
            ctx["use_cutmix"] = use_cutmix
            ctx["lambda"] = lamb
            ctx["confidence"] = confidence
            ctx["dominant_gt"] = dominant_gt
            return batch, ctx

        return batch

    def apply_label_smoothing(self, y, dtype):
        if y.ndim == 1:
            class_indices = y.clone()
            y = to_one_hot_matrix(y, n_classes=self.n_classes)
        else:
            class_indices = y.argmax(dim=-1)

        # label smoothing
        if self.label_smoothing > 0:
            off_value = self.label_smoothing / self.n_classes
            on_value = 1. - self.label_smoothing + off_value
            y_smooth = torch.full(size=(y.shape[0], self.n_classes), fill_value=off_value, dtype=dtype)
            y_smooth.scatter_(1, class_indices.view(-1, 1), on_value)
            y = y_smooth

        return y

    def get_pseudo_labels(self, x):
        training = self.model.training
        if training:
            self.model.eval()
        with torch.no_grad():
            x_on_model_device = x.to(self.model.device)
            probs = self.model.predict(x_on_model_device)
            if self.prediction_head_name is not None:
                probs = probs[self.prediction_head_name]
            probs = probs.detach().cpu()
        if training:
            self.model.train()
        probs = softmax(probs, dim=-1)
        confidence, y_pseudo = probs.max(dim=-1)
        return confidence, y_pseudo

    def apply_mixup(self, x, y, gt, confidence, mixup_mode):
        batch_size = len(x)

        x2_indices, permutation = self.shuffle(item=torch.arange(batch_size), permutation=None)
        gt2, _ = self.shuffle(item=gt, permutation=permutation)

        # sample parameters (lamb, bbox)
        bbox = None
        if mixup_mode == 'ProbPseudoMixup':
            assert confidence is not None
            use_cutmix = torch.full(size=(batch_size, ), fill_value=False)
            confidence2, _ = self.shuffle(item=confidence, permutation=permutation)
            lamb = confidence / (confidence + confidence2)
            confidence = torch.maximum(confidence, confidence2)
            lamb2, _ = self.shuffle(item=lamb, permutation=permutation)
            dominant_gt = torch.where(torch.gt(lamb, lamb2), gt, gt2)
            y2, _ = self.shuffle(item=y, permutation=permutation)
        else:
            y2, _ = self.shuffle(item=y, permutation=permutation)
            use_cutmix = torch.from_numpy(self.rng.random(batch_size) * self.total_p) < self.cutmix_p
            confidence = torch.full(size=(batch_size, ), fill_value=torch.nan)
            if self.mixup_p > 0.:
                mixup_lamb = torch.from_numpy(self.rng.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size))
                mixup_lamb = mixup_lamb.float()
            else:
                mixup_lamb = torch.empty(batch_size)
            if self.cutmix_p > 0.:
                cutmix_lamb = torch.from_numpy(self.rng.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size))
                h, w = x.shape[-2:]
                bbox, cutmix_lamb = self.get_random_bbox(h=h, w=w, lamb=cutmix_lamb)
                cutmix_lamb = cutmix_lamb.float()
            else:
                cutmix_lamb = torch.empty(batch_size)
            lamb = torch.where(use_cutmix, cutmix_lamb, mixup_lamb)
            dominant_gt_cutmix = torch.where(cutmix_lamb >= 0.5, gt, gt2)
            dominant_gt_mixup = torch.where(mixup_lamb >= 0.5, gt, gt2)
            dominant_gt = torch.where(use_cutmix, dominant_gt_cutmix, dominant_gt_mixup)

        x_clone = x.clone()
        bbox_idx = 0
        for i in range(batch_size):
            j = x2_indices[i]
            if use_cutmix[j]:
                top, left, bot, right = bbox[bbox_idx]
                x[i, ..., top:bot, left:right] = x_clone[j, ..., top:bot, left:right]
                bbox_idx += 1
            else:
                x_lamb = lamb[i].view(*[1] * (x.ndim - 1))
                x[i] = x[i].mul_(x_lamb).add_(x_clone[j].mul(1 - x_lamb))

        y_lamb = lamb.view(-1, 1)
        y.mul_(y_lamb).add_(y2.mul(1. - y_lamb))

        return x, y, use_cutmix, lamb, confidence, dominant_gt

    def get_random_bbox(self, h, w, lamb):
        n_bboxes = len(lamb)
        bbox_hcenter = torch.from_numpy(self.rng.integers(h, size=(n_bboxes,)))
        bbox_wcenter = torch.from_numpy(self.rng.integers(w, size=(n_bboxes,)))

        area_half = 0.5 * (1.0 - lamb).sqrt()
        bbox_h_half = (area_half * h).floor()
        bbox_w_half = (area_half * w).floor()

        top = torch.clamp(bbox_hcenter - bbox_h_half, min=0).type(torch.long)
        bot = torch.clamp(bbox_hcenter + bbox_h_half, max=h).type(torch.long)
        left = torch.clamp(bbox_wcenter - bbox_w_half, min=0).type(torch.long)
        right = torch.clamp(bbox_wcenter + bbox_w_half, max=w).type(torch.long)
        bbox = torch.stack([top, left, bot, right], dim=1)

        lamb_adjusted = 1.0 - (bot - top) * (right - left) / (h * w)

        return bbox, lamb_adjusted

    def shuffle(self, item, permutation):
        if self.shuffle_mode == "roll":
            return item.roll(shifts=1, dims=0), None
        if self.shuffle_mode == "flip":
            assert len(item) % 2 == 0
            return item.flip(0), None
        if self.shuffle_mode == "random":
            if permutation is None:
                permutation = self.rng.permutation(len(item))
            return item[permutation], permutation
        raise NotImplementedError

    @property
    def supports_workers(self):
        return self.unsupervised_mixup_mode is None
