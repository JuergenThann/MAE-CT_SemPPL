import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from distributed.config import get_rank, barrier
from distributed.gather import all_gather_nograd
from models.base.single_model_base import SingleModelBase
from models.poolings.single_pooling import SinglePooling
from schedules import schedule_from_kwargs
from utils.factory import create


class ContrastiveHeadBase(SingleModelBase):
    def __init__(
            self,
            queue_size,
            output_dim,
            pooling,
            detach=False,
            loss_weight=1.,
            exclude_self_from_queue=True,
            loss_schedule=None,
            output_shape=None,
            num_queues=1,
            random_queue_label_init=False,
            n_classes=None,
            **kwargs,
    ):
        assert output_shape is None
        assert num_queues >= 1
        assert not random_queue_label_init or n_classes is not None
        super().__init__(output_shape=(output_dim,), **kwargs)
        self.queue_size = queue_size
        self.num_queues = num_queues
        self.exclude_self_from_queue = exclude_self_from_queue
        self.loss_weight = loss_weight
        self.detach = detach
        self.pooling = create(pooling, SinglePooling) or nn.Identity()
        input_shape = self.pooling(torch.ones(1, *self.input_shape)).shape[1:]
        input_dim = np.prod(input_shape)

        self.loss_schedule = schedule_from_kwargs(loss_schedule, update_counter=self.update_counter)

        self.register_components(input_dim, output_dim)

        # use queue independent of method as an online evaluation metric
        if self.num_queues > 1:
            queue_base_size = (self.num_queues, self.queue_size)
        else:
            queue_base_size = (self.queue_size, )

        if random_queue_label_init:
            y = torch.randint(0, n_classes, queue_base_size, dtype=torch.long)
        else:
            y = -torch.ones(*queue_base_size, dtype=torch.long)

        self.register_buffer("queue", F.normalize(torch.randn(*queue_base_size, output_dim), dim=1))
        self.register_buffer("queue_y", y)
        self.register_buffer("queue_id", -torch.ones(*queue_base_size, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(self.num_queues, dtype=torch.long))

    @staticmethod
    def create_projector(input_dim, hidden_dim, output_dim, last_batchnorm=True):
        # this is the projector according to NNCLR paper
        layers = [
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=not last_batchnorm),
        ]
        if last_batchnorm:
            # some methods use affine=False here
            # layers.append(nn.BatchNorm1d(output_dim, affine=False))
            layers.append(nn.BatchNorm1d(output_dim))
        return nn.Sequential(*layers)

    @staticmethod
    def create_predictor(output_dim, hidden_dim):
        # this is the predictor according to NNCLR paper
        return nn.Sequential(
            nn.Linear(output_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # here should be a bias but it also works well without
            nn.Linear(hidden_dim, output_dim, bias=False),
            # nn.BatchNorm1d(output_dim, affine=False),
        )

    def register_components(self, input_dim, output_dim, **kwargs):
        raise NotImplementedError

    def forward(self, x, target_x=None, view=None):
        if self.detach:
            x = x.detach()
            target_x = None if target_x is None else target_x.detach()
        pooled = self.pooling(x).flatten(start_dim=1)
        target_pooled = None if target_x is None else self.pooling(target_x).flatten(start_dim=1)
        return self._forward(pooled, target_pooled, view)

    def _forward(self, x, target_x=None, view=None):
        raise NotImplementedError

    def get_loss(self, outputs, idx, y):
        loss, loss_outputs = self._get_loss(outputs, idx, y)
        scaled_loss = loss * self.loss_weight
        if self.loss_schedule is not None:
            loss_weight = self.loss_weight * self.loss_schedule.get_value(self.update_counter.cur_checkpoint)
        else:
            loss_weight = self.loss_weight
        loss_outputs["loss_weight"] = loss_weight
        loss_outputs["queue"] = self.queue
        loss_outputs.update(outputs)
        return dict(total=scaled_loss, loss=loss), loss_outputs

    def _get_loss(self, outputs, idx, y):
        raise NotImplementedError

    @torch.no_grad()
    def calculate_nn_accuracy(self, normed_projected0, y, ids, idx0=None, nn0=None, vector_to_write_into_queue=None,
                              enqueue=True, queue_idx=0):
        assert queue_idx < self.num_queues

        queue_y = self.queue_y if self.num_queues == 1 else self.queue_y[queue_idx]

        if idx0 is None and nn0 is None:
            # nnclr already found idx0 and nn0
            idx0, nn0 = self.find_nn(normed_projected0, ids=ids, queue_idx=queue_idx)
        nn_acc = ((y == queue_y[idx0]).sum() / len(y)).item()
        if enqueue:
            if vector_to_write_into_queue is None:
                vector_to_write_into_queue = normed_projected0
            self.dequeue_and_enqueue(vector_to_write_into_queue, y=y, ids=ids, queue_idx=queue_idx)
        return nn_acc

    @torch.no_grad()
    def get_queue_similarity_matrix(self, normed_projected, ids, queue_idx=0):
        assert queue_idx < self.num_queues

        queue_id = self.queue_id if self.num_queues == 1 else self.queue_id[queue_idx]
        queue = self.queue if self.num_queues == 1 else self.queue[queue_idx]

        similarity_matrix = normed_projected @ queue.T
        if self.exclude_self_from_queue:
            # check if queue contains embeddings of the same sample of the previous epoch
            is_own_id = queue_id[None, :] == ids[:, None]
            # set similarity to self to -1
            similarity_matrix[is_own_id] = -1.
        return similarity_matrix

    @torch.no_grad()
    def find_nn(self, normed_projected, ids, topk=0, queue_idx=0, get_label=False):
        assert queue_idx < self.num_queues

        similarity_matrix = self.get_queue_similarity_matrix(normed_projected, ids=ids, queue_idx=queue_idx)
        if topk == 0:
            idx = similarity_matrix.max(dim=1)[1]
        else:
            n = similarity_matrix.shape[0]
            candidate_idx = similarity_matrix.topk(topk, dim=1)[1]
            dice = torch.randint(size=(n,), high=topk)
            idx = candidate_idx[torch.arange(n), dice]

        if get_label:
            queue_y = self.queue_y if self.num_queues == 1 else self.queue_y[queue_idx]
            nearest_neighbor_label = queue_y[idx]
            return idx, nearest_neighbor_label
        else:
            queue = self.queue if self.num_queues == 1 else self.queue[queue_idx]
            nearest_neighbor = queue[idx]
            return idx, nearest_neighbor

    def sample_queue(self, labels, queue_idx=0):
        assert queue_idx < self.num_queues

        queue = self.queue if self.num_queues == 1 else self.queue[queue_idx]
        queue_y = self.queue_y if self.num_queues == 1 else self.queue_y[queue_idx]
        queue_id = self.queue_id if self.num_queues == 1 else self.queue_id[queue_idx]

        matches = (queue_y[None, :] == labels[:, None]).to(torch.float)
        sampled_indices = torch.argsort(torch.rand_like(matches)*matches, descending=True)[:, 0]
        sampled_values = queue[sampled_indices]
        sampled_ids = queue_id[sampled_indices]
        sampled_ys = queue_y[sampled_indices]
        valid_sample_map = sampled_ys == labels

        return sampled_ids, sampled_values, valid_sample_map

    @torch.no_grad()
    def dequeue_and_enqueue(self, normed_projected0, y, ids, queue_idx=0):
        """Adds new samples and removes old samples from the queue in a fifo manner. Also stores
        the labels of the samples.

        Args:
            normed_projected0 (torch.Tensor): batch of projected features.
            y (torch.Tensor): labels of the samples in the batch.
            ids (torch.Tensor): ids of the samples in the batch.
            queue_idx (int): index of the queue to use
        """
        # disable in eval mode (for automatic batch_size finding)
        if not self.training:
            return

        assert queue_idx < self.num_queues

        queue = self.queue if self.num_queues == 1 else self.queue[queue_idx, ...]
        queue_y = self.queue_y if self.num_queues == 1 else self.queue_y[queue_idx, ...]
        queue_id = self.queue_id if self.num_queues == 1 else self.queue_id[queue_idx, ...]

        normed_projected0 = all_gather_nograd(normed_projected0)
        y = all_gather_nograd(y)
        ids = all_gather_nograd(ids)

        batch_size = normed_projected0.shape[0]

        ptr = int(self.queue_ptr[queue_idx])

        if ptr + batch_size > self.queue_size:
            reduced_batch_size = self.queue_size - ptr
            queue[ptr:, ...] = normed_projected0[:reduced_batch_size, ...]
            queue_y[ptr:] = y[:reduced_batch_size]
            queue_id[ptr:] = ids[:reduced_batch_size]

            ptr = 0
            normed_projected0 = normed_projected0[reduced_batch_size:, ...]
            y = y[reduced_batch_size:]
            ids = ids[reduced_batch_size:]
            batch_size = normed_projected0.shape[0]

        queue[ptr:ptr + batch_size] = normed_projected0
        queue_y[ptr:ptr + batch_size] = y
        queue_id[ptr:ptr + batch_size] = ids
        ptr = ptr + batch_size
        self.queue_ptr[queue_idx] = ptr

    def accept_view(self, view):
        return True
