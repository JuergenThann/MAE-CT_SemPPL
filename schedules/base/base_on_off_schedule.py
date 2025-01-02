from .basic_schedule import BasicSchedule
from math import ceil


class BaseOnOffSchedule(BasicSchedule):
    def __init__(self, min_every_n_steps: int, max_every_n_steps: int, decreasing: bool, **kwargs):

        super().__init__(abs_start_value=0, abs_delta=1, **kwargs)
        self.min_every_n_steps = min_every_n_steps
        self.max_every_n_steps = max_every_n_steps
        self.on_off_schedules = dict()
        self.decreasing = decreasing

    def _get_value(self, step, total_steps):
        if total_steps not in self.on_off_schedules:
            step_block_size = self.min_every_n_steps + self.max_every_n_steps
            n_step_blocks = int(ceil(total_steps / step_block_size))
            n_values = self.max_every_n_steps - self.min_every_n_steps + 1
            n_on_per_block = self._get_n_on_per_block(n_step_blocks, n_values)

            self.on_off_schedules[total_steps] = list([
                1 if inner_block_idx < n_on else 0
                for n_on in n_on_per_block
                for inner_block_idx in reversed(range(step_block_size))
            ])
            if self.decreasing:
                self.on_off_schedules[total_steps] = list([1 - factor for factor in self.on_off_schedules[total_steps]])

        return self.on_off_schedules[total_steps][step]

    def _get_n_on_per_block(self, n_step_blocks, n_values):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError
