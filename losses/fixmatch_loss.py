import torch
import torch.nn.functional as F

from distributed.gather import all_gather_grad, all_gather_nograd


def fixmatch_loss_fn(logits_strong, logits_weak, labels, threshold: float, unsupervised_loss_weight: float):
    assert 0 <= threshold <= 1

    labeled_mask = labels >= 0

    logits_weak = all_gather_grad(logits_weak)
    labels = all_gather_nograd(labels)

    # supervised loss
    logits_weak_labeled = logits_weak[labeled_mask, :]
    known_labels = labels[labeled_mask]
    supervised_loss = F.cross_entropy(logits_weak_labeled, known_labels, reduction='mean')

    # for logging
    probs_weak_labeled = F.softmax(logits_weak_labeled.detach(), dim=1)
    probs_labeled_max, _ = torch.max(probs_weak_labeled, dim=1)

    # unsupervised loss
    if logits_strong is not None:
        logits_strong = all_gather_grad(logits_strong)
        logits_strong_unlabeled = logits_strong[~labeled_mask]

        logits_weak_unlabeled = logits_weak[~labeled_mask, :].detach()
        probs_weak_unlabeled = F.softmax(logits_weak_unlabeled, dim=1)
        probs_unlabeled_max, pseudo_labels = torch.max(probs_weak_unlabeled, dim=1)
        threshold_mask = probs_unlabeled_max >= threshold

        unsupervised_loss_unreduced = (
            F.cross_entropy(
                logits_strong_unlabeled,
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
            classification_confidence_unlabeled=probs_unlabeled_max.mean().item(),
            classification_confidence_unlabeled_over_threshold=probs_unlabeled_max[threshold_mask].mean().item(),
            classification_confidence_labeled=probs_labeled_max.mean().item(),
            pseudo_label_accuracy=(pseudo_labels[threshold_mask] == (-labels[~labeled_mask][threshold_mask]-1)).to(torch.float).mean().item()
        )
    else:
        # final loss and log output
        return supervised_loss, dict(
            supervised_loss=supervised_loss.item(),
            classification_confidence_labeled=probs_labeled_max.mean().item()
        )
