import math
from typing import TypeVar, Optional, Iterator

import torch
from torch.utils.data import DistributedSampler

from datasets.dataset_wrappers.semisupervised_oversampling_wrapper import SemisupervisedOversamplingWrapper
from distributed.config import is_distributed, get_world_size, get_rank

__all__ = ["SemisupervisedOversamplingSampler", ]

T_co = TypeVar('T_co', covariant=True)


# modified copy of torch.utils.data.DistributedSampler
class SemisupervisedOversamplingSampler(DistributedSampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> # xdoctest: +SKIP
        >>> sampler = SemisupervisedOversamplingSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset, batch_size: int,
                 num_replicas: Optional[int] = None, rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        wrapper = next((w for w in dataset.all_wrappers if isinstance(w, SemisupervisedOversamplingWrapper)), None)
        assert wrapper is not None

        unlabeled_to_labeled_ratio = wrapper.unlabeled_to_labeled_ratio
        assert unlabeled_to_labeled_ratio > 0
        assert unlabeled_to_labeled_ratio % 1 == 0

        assert batch_size % (unlabeled_to_labeled_ratio + 1) == 0

        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.unlabeled_to_labeled_ratio = unlabeled_to_labeled_ratio

        labeled_map = self.dataset.targets >= 0
        self.labeled_indices = torch.nonzero(labeled_map).squeeze()
        self.unlabeled_indices = torch.nonzero(~labeled_map).squeeze()
        self.num_samples = len(self.dataset)

        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and self.num_samples % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (self.num_samples - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(self.num_samples / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            labeled_indices = self.labeled_indices[torch.randperm(self.labeled_indices.numel(), generator=g)]
            unlabeled_indices = self.unlabeled_indices[torch.randperm(self.unlabeled_indices.numel(), generator=g)]
        else:
            labeled_indices = self.labeled_indices
            unlabeled_indices = self.unlabeled_indices

        indices = torch.cat((labeled_indices, unlabeled_indices))
        index_order = torch.arange(indices.numel(), dtype=torch.float)
        labeled_map = index_order % (self.unlabeled_to_labeled_ratio + 1) == 0
        index_order[labeled_map] = index_order[labeled_map] / int(self.unlabeled_to_labeled_ratio + 1)
        index_order[~labeled_map] = index_order[~labeled_map] + self.labeled_indices.numel() - 1 - torch.floor(
            index_order[~labeled_map] / (self.unlabeled_to_labeled_ratio + 1))
        index_order = index_order[index_order < indices.numel()]
        indices = indices[index_order.to(torch.long)].tolist()

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        rank_start_idx = self.num_samples * self.rank
        rank_end_idx = self.num_samples * (self.rank+1)
        indices = indices[rank_start_idx:rank_end_idx]
        assert len(indices) == self.num_samples

        return iter(indices)
