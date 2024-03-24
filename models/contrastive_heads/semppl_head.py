
import torch
from torch.nn.functional import normalize

from losses.semppl_loss import semppl_loss_fn
from .semppl_noqueue_head import SemPPLNoqueueHead
from utils.logging_util import log_from_all_ranks
from distributed.gather import all_gather_nograd, all_reduce_mean_grad, all_reduce_mean_nograd, all_reduce_sum_nograd

class SemPPLHead(SemPPLNoqueueHead):
    def __init__(self, local_scaling_knn=0, num_semantic_positives=0, topk=1, alpha=0.2, c=0.3, lambda_=5,
                 disable_pseudo_labeling=False, vote_threshold=0.0, **kwargs):
        super().__init__(**kwargs)
        self.local_scaling_knn = local_scaling_knn
        self.num_semantic_positives = num_semantic_positives
        self.topk = topk
        self.alpha = alpha
        self.c = c
        self.lambda_ = lambda_
        self.disable_pseudo_labeling = disable_pseudo_labeling
        self.vote_threshold = vote_threshold
        self.large_view_buffer = None
        self.y_buffer = None

    def _get_loss(self, outputs, idx, y, training=None):
        with log_from_all_ranks():
            training = self.training if training is None else training
            dtype = outputs["view0"]["projected"].dtype

            is_large_views = outputs.get("shape_idx", 0) == 0

            with_label_map = y >= 0
            without_label_map = torch.logical_not(with_label_map)

            y = y.clone()
            labels = y.clone()
            labels[labels < 0] = -labels[labels < 0] - 1

            if is_large_views:
                # Enqueue the labeled images in the batch
                if training:
                    for view in range(self.num_large_views):
                        self.dequeue_and_enqueue(
                            normalize(outputs[f"view{view}"]["projected"][with_label_map, ...], dim=-1),
                            y[with_label_map],
                            idx[with_label_map],
                            queue_idx=view
                        )

                # Pseudo-label computation for unlabeled and labeled examples (for analysis)
                votes = torch.empty(len(y), self.num_large_views**2, dtype=y.dtype, device=self.device)
                for j in range(self.num_large_views):
                    ol_j = normalize(outputs[f"view{j}"]["predicted"], dim=-1)
                    for i in range(self.num_large_views):
                        _, nn_y = self.find_nn(ol_j, ids=idx, topk=self.topk, get_label=True,
                                               queue_idx=i)
                        votes[:, self.num_large_views*j+i] = nn_y
                y_pseudo, _ = torch.mode(votes)
                vote_confidence = (votes == y_pseudo[..., None]).float().mean(dim=-1)
                thresholded_map = torch.logical_and(without_label_map, vote_confidence >= self.vote_threshold)
                known_label_acc = (labels[with_label_map] == y_pseudo[with_label_map]).float().mean().item()
                known_label_count = with_label_map.sum().item()
                unknown_label_acc = (labels[without_label_map] == y_pseudo[without_label_map]).float().mean().item()
                unknown_label_count = without_label_map.sum().item()
                thresholded_label_acc = (labels[thresholded_map] == y_pseudo[thresholded_map]).float().mean().item()
                thresholded_label_count = thresholded_map.sum().item()

                if not self.disable_pseudo_labeling:
                    y[thresholded_map] = y_pseudo[thresholded_map]

                self.large_view_buffer = outputs
                self.y_buffer = y
            else:
                vote_confidence = torch.tensor(0, dtype=dtype, device=self.device)
                known_label_acc = 0
                known_label_count = 0
                unknown_label_acc = 0
                unknown_label_count = 0
                thresholded_label_acc = 0
                thresholded_label_count = 0
                y = self.y_buffer

            valid_label_map = y >= 0
            pseudo_label_map = torch.logical_and(without_label_map, valid_label_map)

            l_augm = torch.tensor(0, dtype=dtype, device=self.device)
            i_augm = torch.tensor(0, dtype=dtype, device=self.device)
            l_sempos = torch.tensor(0, dtype=dtype, device=self.device)
            i_sempos = torch.tensor(0, dtype=dtype, device=self.device)
            nn_acc = 0

            augm_count = (self.num_large_views + self.num_small_views) * self.num_large_views
            sempos_count = (self.num_large_views + self.num_small_views) * self.num_large_views * self.num_semantic_positives
            nn_acc_count = self.num_large_views ** 2

            if is_large_views:
                for i in range(self.num_large_views):
                    for j in range(self.num_large_views):
                        # if i == j:
                        #     continue
                        ol_i = normalize(outputs[f"view{i}"]["predicted"], dim=-1)
                        tl_j = normalize(outputs[f"view{j}"]["projected"], dim=-1)
                        l_augm_ij, i_augm_ij, l_sempos_ij, i_sempos_ij = \
                            self._calc_loss(ol_i, tl_j, i, j, y, valid_label_map, pseudo_label_map)
                        l_augm += l_augm_ij
                        l_sempos += l_sempos_ij
                        i_augm += i_augm_ij
                        i_sempos += i_sempos_ij
                        nn_acc += self.calculate_nn_accuracy(tl_j, ids=idx, y=labels, enqueue=False, queue_idx=i)
            else:
                for i in range(self.num_small_views):
                    for j in range(self.num_large_views):
                        os_i = normalize(outputs[f"view{i}"]["predicted"], dim=-1)
                        tl_j = normalize(self.large_view_buffer[f"view{j}"]["projected"], dim=-1)
                        l_augm_ij, i_augm_ij, l_sempos_ij, i_sempos_ij = \
                            self._calc_loss(os_i, tl_j, i, j, y, valid_label_map, pseudo_label_map)
                        l_augm += l_augm_ij
                        l_sempos += l_sempos_ij
                        i_augm += i_augm_ij
                        i_sempos += i_sempos_ij

            i_augm /= augm_count
            l_augm /= augm_count
            i_sempos /= sempos_count
            l_sempos /= sempos_count
            nn_acc /= nn_acc_count

            loss = self.c * (l_augm + self.alpha * l_sempos) + self.lambda_ * (i_augm + i_sempos)

            num_pseudo_labels = pseudo_label_map.sum()
            if num_pseudo_labels > 0:
                pseudo_label_correctness = all_gather_nograd((labels[pseudo_label_map] == y[pseudo_label_map]).float())
                pseudo_label_acc = pseudo_label_correctness.mean().item()
            else:
                pseudo_label_acc = 0

            vote_confidence = all_gather_nograd(vote_confidence)
            # loss = all_reduce_mean_grad(loss)
            if is_large_views:
                known_label_count = all_reduce_sum_nograd(known_label_count)
                known_label_acc = all_reduce_mean_nograd(known_label_acc)
                unknown_label_count = all_reduce_sum_nograd(unknown_label_count)
                unknown_label_acc = all_reduce_mean_nograd(unknown_label_acc)
                thresholded_label_count = all_reduce_sum_nograd(thresholded_label_count)
                thresholded_label_acc = all_reduce_mean_nograd(thresholded_label_acc)
                return loss, dict(nn_accuracy=nn_acc, pseudo_label_acc=pseudo_label_acc, L_augm=l_augm.item(),
                                  L_sempos=l_sempos.item(), L_SemPPL_wo_inv_pen=(self.c * (l_augm+self.alpha*l_sempos)).item(),
                                  I_augm=i_augm.item(), I_sempos=i_sempos.item(), vote_confidence=vote_confidence.mean().item(),
                                  known_label_count=known_label_count, known_label_acc=known_label_acc,
                                  unknown_label_count=unknown_label_count, unknown_label_acc=unknown_label_acc,
                                  thresholded_label_count=thresholded_label_count, thresholded_label_acc=thresholded_label_acc)
            else:
                return loss, dict(L_augm=l_augm.item(),
                                  L_sempos=l_sempos.item(), L_SemPPL_wo_inv_pen=(self.c * (l_augm+self.alpha*l_sempos)).item(),
                                  I_augm=i_augm.item(), I_sempos=i_sempos.item())

    def _calc_loss(self, normed_o_i, normed_t_j, i, j, y, valid_label_map, pseudo_label_map):
        l_augm = torch.tensor(0, dtype=normed_o_i.dtype, device=normed_o_i.device)
        i_augm = torch.tensor(0, dtype=normed_o_i.dtype, device=normed_o_i.device)
        l_sempos = torch.tensor(0, dtype=normed_o_i.dtype, device=normed_o_i.device)
        i_sempos = torch.tensor(0, dtype=normed_o_i.dtype, device=normed_o_i.device)

        loss_ij = semppl_loss_fn(normed_o_i, normed_t_j, temperature=self.temperature, transposed=self.transposed,
                                 num_negatives=self.num_negatives)
        l_augm += loss_ij
        if i < j:
            i_augm -= loss_ij.detach()
        else:
            i_augm += loss_ij

        normed_o_i = normed_o_i[valid_label_map, ...]
        for _ in range(self.num_semantic_positives):
            _, z, valid_z_map = self.sample_queue(y[valid_label_map], queue_idx=j)

            loss_ij = semppl_loss_fn(normed_o_i[valid_z_map, ...], z[valid_z_map, ...],
                                     temperature=self.temperature, transposed=self.transposed,
                                     num_negatives=self.num_negatives)
            l_sempos += loss_ij
            if i < j:
                i_sempos -= loss_ij.detach()
            else:
                i_sempos += loss_ij

        return l_augm, i_augm, l_sempos, i_sempos

    @torch.no_grad()
    def get_queue_similarity_matrix(self, normed_projected, ids, queue_idx=0):
        similarity_matrix = super().get_queue_similarity_matrix(normed_projected, ids=ids, queue_idx=queue_idx)
        if self.local_scaling_knn == 0:
            return similarity_matrix

        # apply local scaling for hubness reduction
        distance_matrix = similarity_matrix
        # retrieve distances of k nearest neighbors
        nearest_neighbor_distances_z = distance_matrix.topk(dim=1, sorted=True, k=self.local_scaling_knn)[0]
        nearest_neighbor_distances_queue = distance_matrix.topk(dim=0, sorted=True, k=self.local_scaling_knn)[0]

        # take distance of furthest neighbor
        sigma_z = nearest_neighbor_distances_z[:, 0]
        sigma_queue = nearest_neighbor_distances_queue[0, :]

        # scale distances
        sigma_matrix = sigma_z[:, None] @ sigma_queue[None, :]
        return 1 - torch.exp(-(distance_matrix ** 2) / sigma_matrix)
