import torch
import torch.nn.functional as F

from distributed.gather import all_gather_grad, all_gather_nograd


def fixmatch_loss_fn(strong_logits, weak_logits, pseudo_label_logits, labels, threshold: float,
                     unsupervised_loss_weight: float):
    assert 0 <= threshold <= 1

    labeled_mask = labels >= 0

    weak_logits = all_gather_grad(weak_logits)
    labels = all_gather_nograd(labels)

    # supervised loss
    weak_labeled_logits = weak_logits[labeled_mask, :]
    known_labels = labels[labeled_mask]
    supervised_loss = F.cross_entropy(weak_labeled_logits, known_labels, reduction='mean')

    # for logging
    weak_labeled_probs = F.softmax(weak_labeled_logits.detach(), dim=1)
    weak_labeled_probs_max, _ = torch.max(weak_labeled_probs, dim=1)

    # unsupervised loss
    if strong_logits is not None:
        pseudo_label_logits = all_gather_nograd(pseudo_label_logits)
        strong_logits = all_gather_grad(strong_logits)
        strong_unlabeled_logits = strong_logits[~labeled_mask]

        pseudo_label_logits = pseudo_label_logits[~labeled_mask, :].detach()
        pseudo_label_probs = F.softmax(pseudo_label_logits, dim=1)
        pseudo_label_max_probs, pseudo_labels = torch.max(pseudo_label_probs, dim=1)
        threshold_mask = pseudo_label_max_probs >= threshold

        unsupervised_loss_unreduced = (
            F.cross_entropy(
                strong_unlabeled_logits,
                pseudo_labels,
                reduction='none'
            ) * threshold_mask
        )
        unsupervised_loss = unsupervised_loss_unreduced.mean()

        samples_over_threshold = threshold_mask.sum().item()

        # final loss and log output
        return supervised_loss + unsupervised_loss_weight * unsupervised_loss, dict(
            samples_above_threshold=samples_over_threshold,
            supervised_loss=supervised_loss.item(),
            unsupervised_loss=unsupervised_loss.item(),
            unsupervised_loss_mean_over_threshold=torch.nan if samples_over_threshold == 0 else unsupervised_loss_unreduced.sum().item() / samples_over_threshold,
            classification_confidence_unlabeled=pseudo_label_max_probs.mean().item(),
            classification_confidence_unlabeled_over_threshold=pseudo_label_max_probs[threshold_mask].mean().item(),
            classification_confidence_labeled=weak_labeled_probs_max.mean().item(),
            pseudo_label_accuracy=(pseudo_labels[threshold_mask] == (-labels[~labeled_mask][threshold_mask]-1)).to(torch.float).mean().item()
        )
    else:
        # final loss and log output
        return supervised_loss, dict(
            supervised_loss=supervised_loss.item(),
            classification_confidence_labeled=weak_labeled_probs_max.mean().item()
        )
