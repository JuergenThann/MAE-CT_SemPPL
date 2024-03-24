from functools import partial

import torch
import torch.nn.functional as F

from distributed.config import get_rank
from distributed.gather import all_gather_grad, all_gather_nograd


def semppl_loss_fn(normed_predicted, normed_nn, temperature, transposed=False, num_negatives=10):
    # this is redundant (nn is already normalized)
    # normed_nn = F.normalize(nn, dim=-1)

    rank = get_rank()
    if transposed:
        normed_nn = all_gather_grad(normed_nn)
        logits = normed_predicted @ normed_nn.T / temperature
        n = normed_predicted.size(0)
    else:
        normed_predicted = all_gather_grad(normed_predicted)
        logits = normed_nn @ normed_predicted.T / temperature
        n = normed_nn.size(0)

    def _sample_negatives(tensor):
        shapes = all_gather_nograd(torch.tensor(tensor.shape, device=tensor.device)[None, :])
        if tensor.shape[0] == 0 or tensor.shape[1] < 2:
            return tensor

        result = torch.empty((tensor.size(0), min(tensor.size(1), 1 + num_negatives)),
                             dtype=tensor.dtype, device=tensor.device)
        result[:, 0] = tensor.diagonal()

        n_0, n_1 = tensor.shape
        diagonal_indices = list(range(shapes[:rank, 0].sum(), n_0*n_1, n_1+1))
        off_diagonal_indices = torch.tensor([i for i in range(0, n_0*n_1) if i not in diagonal_indices])
        off_diagonal = tensor.flatten()[off_diagonal_indices].reshape(n_0, n_1 - 1)
        indices = torch.argsort(torch.rand_like(off_diagonal), dim=1)[:, :num_negatives]
        result[:, 1:] = torch.gather(off_diagonal, dim=1, index=indices)
        return result

    if num_negatives is not None:
        logits = _sample_negatives(logits)
        labels = torch.zeros(n, dtype=torch.int64, device=normed_predicted.device)
    else:
        labels = torch.arange(n * rank, n * (rank + 1), device=normed_predicted.device)
    # reduction="none" has large errors with bfloat16
    # loss = F.cross_entropy(logits, labels, reduction="none")
    loss = F.cross_entropy(logits, labels, reduction='mean')
    return loss
