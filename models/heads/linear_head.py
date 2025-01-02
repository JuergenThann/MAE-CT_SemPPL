import numpy as np
import torch
import torch.nn as nn

from initializers.trunc_normal_initializer import TruncNormalInitializer
from models.base.single_model_base import SingleModelBase
from models.poolings import pooling_from_kwargs
from schedules import schedule_from_kwargs


class LinearHead(SingleModelBase):
    def __init__(self, nonaffine_batchnorm=False, pooling=None, initializer=None, views_to_consume=None,
                 loss_weight=1.0, loss_schedule=None, detach_head=False, detach_schedule=None, **kwargs):
        initializer = initializer or TruncNormalInitializer(std=1e-2)
        super().__init__(initializer=initializer, **kwargs)
        self.nonaffine_batchnorm = nonaffine_batchnorm
        self.pooling = pooling_from_kwargs(pooling)
        input_shape = self.pooling(torch.ones(1, *self.input_shape), ctx=self.ctx).shape[1:]
        input_dim = np.prod(input_shape)
        self.norm = nn.BatchNorm1d(input_dim, affine=False) if nonaffine_batchnorm else nn.Identity()
        self.layer = nn.Sequential(
            nn.Flatten(start_dim=1),
            self.norm,
            nn.Linear(input_dim, np.prod(self.output_shape)),
        )
        # normalization values that can be patched by PrepareFeatureStatisticsLogger
        self.mean = None
        self.std = None
        self.views_to_consume = views_to_consume
        self.detach_head = detach_head
        self.loss_weight = loss_weight
        self.loss_schedule = schedule_from_kwargs(loss_schedule, update_counter=self.update_counter)
        self.detach_schedule = schedule_from_kwargs(detach_schedule, update_counter=self.update_counter)

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict=state_dict, strict=strict)

    def register_components(self, input_dim, output_dim, **kwargs):
        raise NotImplementedError

    def _get_detach_head(self):
        detach_head = self.detach_head
        if not self.detach_head and self.detach_schedule is not None:
            detach_head = self.detach_schedule.get_value(self.update_counter.cur_checkpoint) != 0
        return detach_head

    def forward(self, x, target_x=None, view=None):
        x = self.pooling(x, ctx=self.ctx)
        if self.mean is not None and self.std is not None:
            assert isinstance(self.norm, nn.Identity)
            x = (x - self.mean) / self.std

        if self._get_detach_head():
            x = x.detach()

        return self.layer(x)

    def features(self, x):
        return self(x)

    def predict(self, x):
        return dict(main=self(x))

    def predict_binary(self, x):
        return self.predict(x)

    def get_loss(self, outputs, idx, y):
        loss, loss_outputs = self._get_loss(outputs, idx, y)
        if self.loss_schedule is not None:
            loss_weight = self.loss_weight * self.loss_schedule.get_value(self.update_counter.cur_checkpoint)
        else:
            loss_weight = self.loss_weight
        scaled_loss = loss * loss_weight
        loss_outputs["loss_weight"] = loss_weight
        loss_outputs["detach_head"] = 1.0 if self._get_detach_head() else 0.0
        loss_outputs.update(outputs)
        return dict(total=scaled_loss, loss=loss), loss_outputs

    def _get_loss(self, outputs, idx, y):
        raise NotImplementedError
