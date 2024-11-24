import torch
import torch.nn.functional as F

from distributed.gather import all_gather_grad, all_gather_nograd


def fixmatch_loss_fn(unlabeled_logits, labeled_logits, pseudo_label_logits, confidence, labels, threshold: float,
                     unsupervised_loss_weight: float):
    assert 0 <= threshold <= 1

    labeled_logits = all_gather_grad(labeled_logits)
    labels = all_gather_nograd(labels)

    if labels.ndim > 1:
        gt = torch.argmax(torch.abs(labels), -1)
        labels = torch.where(labels.sum(-1) > 0, gt, -gt-1)

    labeled_mask = labels >= 0

    # supervised loss
    labeled_logits = labeled_logits[labeled_mask, :]
    known_labels = labels[labeled_mask]
    supervised_loss = F.cross_entropy(labeled_logits, known_labels, reduction='mean')

    # for logging
    labeled_probs = F.softmax(labeled_logits.detach(), dim=1)
    labeled_confidence, _ = torch.max(labeled_probs, dim=1)

    # unsupervised loss
    if unlabeled_logits is not None:
        unlabeled_logits = all_gather_grad(unlabeled_logits)
        unlabeled_logits = unlabeled_logits[~labeled_mask]

        if confidence is not None:
            pseudo_label_confidence = confidence[~labeled_mask]
            pseudo_labels = pseudo_label_logits[~labeled_mask]
        else:
            pseudo_label_logits = all_gather_nograd(pseudo_label_logits)
            pseudo_label_logits = pseudo_label_logits[~labeled_mask, :].detach()
            pseudo_label_probs = F.softmax(pseudo_label_logits, dim=1)
            pseudo_label_confidence, pseudo_labels = torch.max(pseudo_label_probs, dim=1)
        threshold_mask = pseudo_label_confidence >= threshold

        unsupervised_loss_unreduced = (
            F.cross_entropy(
                unlabeled_logits,
                pseudo_labels,
                reduction='none'
            ) * threshold_mask
        )
        unsupervised_loss = unsupervised_loss_unreduced.mean()

        samples_over_threshold = threshold_mask.sum().item()

        if pseudo_labels.ndim > 1:
            pseudo_labels = pseudo_labels.argmax(dim=-1)

        # final loss and log output
        return supervised_loss + unsupervised_loss_weight * unsupervised_loss, dict(
            samples_above_threshold=samples_over_threshold,
            supervised_loss=supervised_loss.item(),
            unsupervised_loss=unsupervised_loss.item(),
            unsupervised_loss_mean_over_threshold=torch.nan if samples_over_threshold == 0 else unsupervised_loss_unreduced.sum().item() / samples_over_threshold,
            classification_confidence_unlabeled=pseudo_label_confidence.mean().item(),
            classification_confidence_unlabeled_over_threshold=pseudo_label_confidence[threshold_mask].mean().item(),
            classification_confidence_labeled=labeled_confidence.mean().item(),
            pseudo_label_accuracy=(pseudo_labels[threshold_mask] == (-labels[~labeled_mask][threshold_mask]-1)).to(torch.float).mean().item(),
            pseudo_label_accuracy_with_under_thresh=(pseudo_labels == (-labels[~labeled_mask]-1)).to(torch.float).mean().item()
        )
    else:
        # final loss and log output
        return supervised_loss, dict(
            supervised_loss=supervised_loss.item(),
            classification_confidence_labeled=labeled_confidence.mean().item()
        )
