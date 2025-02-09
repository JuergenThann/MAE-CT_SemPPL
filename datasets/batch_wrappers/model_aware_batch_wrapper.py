import numpy as np
import torch

from kappadata.collators.base.kd_single_collator import KDSingleCollator
from kappadata.wrappers.mode_wrapper import ModeWrapper
from torch.nn.functional import softmax
from kappadata.utils.one_hot import to_one_hot_matrix
from multiprocessing import Queue


class ModelAwareBatchWrapper:
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.model = None

    @property
    def supports_workers(self):
        return False
