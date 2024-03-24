import fnmatch

from loggers.base.logger_base import LoggerBase
from utils.infer_higher_is_better import higher_is_better_from_metric_key


class BestMetricLogger(LoggerBase):
    def __init__(self, pattern, log_absolute_best=False, **kwargs):
        super().__init__(**kwargs)
        self.pattern = pattern
        self.higher_is_better = higher_is_better_from_metric_key(self.pattern)
        self.best_metric_value = -float("inf") if self.higher_is_better else float("inf")
        self.log_absolute_best = log_absolute_best

    def _extract_metric_value(self, logger_info_dict):
        best_metric_value = -float("inf") if self.higher_is_better else float("inf")
        pattern_matched = False
        for metric_key, metric_value in logger_info_dict.items():
            if "*" in self.pattern or "?" in self.pattern:
                # pattern with * or ?
                if not fnmatch.fnmatch(metric_key, self.pattern):
                    continue
            else:
                # pattern with contains
                if self.pattern not in metric_key:
                    continue
            pattern_matched = True
            assert metric_value is not None, (
                f"couldn't find metric_value {metric_key} (valid metric keys={list(logger_info_dict.keys())}). "
                f"make sure logger that produces the metric key is called beforehand"
            )
            if self.higher_is_better:
                if metric_value > best_metric_value:
                    best_metric_value = metric_value
            else:
                if metric_value < best_metric_value:
                    best_metric_value = metric_value

        if not pattern_matched:
            self.logger.info(f'Could not match pattern {self.pattern}. Available keys: {logger_info_dict.keys()}')

        return best_metric_value

    # noinspection PyMethodOverriding
    def _log(self, update_counter, trainer, model, logger_info_dict, **kwargs):
        metric_value = self._extract_metric_value(logger_info_dict)

        is_new_absolute_best = False
        old_best = self.best_metric_value
        if self.higher_is_better:
            if metric_value > self.best_metric_value:
                is_new_absolute_best = True
                self.best_metric_value = metric_value
        else:
            if metric_value < self.best_metric_value:
                is_new_absolute_best = True
                self.best_metric_value = metric_value

        if self.log_absolute_best:
            if is_new_absolute_best:
                self.logger.info(f"new best metric ({self.pattern}): {old_best} --> {self.best_metric_value}")
            self.writer.add_scalar(
                self.pattern.replace('*', '').replace('?', '').rstrip('/') + "/best_absolute",
                self.best_metric_value,
                update_counter=update_counter
            )
        else:
            self.writer.add_scalar(
                self.pattern.replace('*', '').replace('?', '').rstrip('/') + "/best",
                metric_value,
                update_counter=update_counter
            )

    def _after_training(self, **kwargs):
        if not self.log_absolute_best:
            self.logger.info(f"best {self.pattern}: {self.best_metric_value}")
