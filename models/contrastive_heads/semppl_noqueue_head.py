from torch.nn.functional import normalize

from losses.semppl_loss import semppl_loss_fn
from .base.contrastive_head_base import ContrastiveHeadBase
import torch


class SemPPLNoqueueHead(ContrastiveHeadBase):
    def __init__(self, temperature, proj_hidden_dim, pred_hidden_dim, num_large_views, num_small_views,
                 transposed=False, num_negatives=10, **kwargs):
        self.proj_hidden_dim = proj_hidden_dim
        self.pred_hidden_dim = pred_hidden_dim
        self.projector, self.predictor = None, None
        self.num_large_views = num_large_views
        self.num_small_views = num_small_views
        super().__init__(**kwargs)
        self.temperature = temperature
        self.transposed = transposed
        self.num_negatives = num_negatives

    def register_components(self, input_dim, output_dim, **kwargs):
        self.projector = self.create_projector(input_dim, self.proj_hidden_dim, output_dim)
        self.predictor = self.create_predictor(output_dim, self.pred_hidden_dim)

    def _forward(self, pooled, target_pooled=None):
        projected = self.projector(pooled)
        predicted = self.predictor(projected)
        return dict(projected=projected, predicted=predicted)

    def _get_loss(self, outputs, idx, y):
        with_label_map = y != -1

        loss = torch.tensor(0, dtype=outputs["view0"]["projected"].dtype, device=y.device)
        loss_count = 0
        for i in range(self.num_large_views):
            for j in range(self.num_large_views):
                if i == j:
                    continue

                ol_i = outputs[f"view{i}"]["predicted"]
                ol_i = normalize(ol_i, dim=-1)
                tl_j = outputs[f"view{j}"]["projected"]
                tl_j = normalize(tl_j, dim=-1)

                loss_count += 1
                loss += semppl_loss_fn(ol_i, tl_j, temperature=self.temperature, transposed=self.transposed,
                                       num_negatives=self.num_negatives)

        loss /= loss_count

        if torch.any(with_label_map):
            nn_acc = self.calculate_nn_accuracy(tl_j[with_label_map, ...], ids=idx[with_label_map, ...],
                                                y=y[with_label_map, ...], enqueue=False)
            self.last_nn_acc = nn_acc
        else:
            nn_acc = self.last_nn_acc
        return loss, dict(nn_accuracy=nn_acc)

