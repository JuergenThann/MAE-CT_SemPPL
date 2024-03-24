from collections import defaultdict

import numpy as np
import numbers
import math

import torch

from distributed.gather import all_reduce_mean_grad
from loggers.base.logger_base import LoggerBase


class GroupUpdateOutputLogger(LoggerBase):
    def __init__(self, pattern, verbose=False, invert_key=True, category=None, **kwargs):
        super().__init__(**kwargs)
        self.tracked_values = defaultdict(list)
        self.pattern = pattern
        self.verbose = verbose
        self.invert_key = invert_key
        self.category = category

    def _track_after_accumulation_step(self, update_outputs, **kwargs):
        for key, value in update_outputs.items():
            if self.pattern in key:
                if torch.is_tensor(value) and value.numel() == 1:
                    value = value.item()
                if isinstance(value, numbers.Real):
                    if key not in self.tracked_values:
                        self.tracked_values[key] = []
                    if not math.isnan(value):
                        self.tracked_values[key].append(value)
                else:
                    self.tracked_values[key] = value

    @property
    def allows_multiple_interval_types(self):
        return False

    def _log(self, update_counter, **_):
        for key, tracked_values in self.tracked_values.items():
            mean_value = np.mean(tracked_values)
            mean_value = all_reduce_mean_grad(mean_value)
            # change order of key (e.g. key="mocov3/nn_accuracy" -> "nn_accuracy/mocov3")
            if self.verbose:
                self.logger.info(f"{key}: {mean_value:.5f}")
            if self.invert_key:
                key = "/".join(reversed(key.split("/")))
            if self.category is not None:
                key = f"{self.category}/{key}"
            # use to_short_interval_string here because this is basically an online logger
            self.writer.add_scalar(f"{key}/{self.to_short_interval_string()}", mean_value, update_counter)
        self.tracked_values.clear()
