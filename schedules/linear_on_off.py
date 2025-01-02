from .base.base_on_off_schedule import BaseOnOffSchedule
from math import floor


# LEGACY: old yamls use this
class LinearOnOff(BaseOnOffSchedule):
    # noinspection PyUnusedLocal
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_n_on_per_block(self, n_step_blocks, n_values):
        n_step_blocks_per_value = n_step_blocks / n_values
        return [self.min_every_n_steps + floor(block_idx / n_step_blocks_per_value) for block_idx in range(n_step_blocks)]

    def __str__(self):
        return "LinearOnOff"
