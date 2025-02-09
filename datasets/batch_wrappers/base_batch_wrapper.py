import numpy as np
import torch

from kappadata.collators.base.kd_single_collator import KDSingleCollator
from kappadata.wrappers.mode_wrapper import ModeWrapper
from torch.nn.functional import softmax
from kappadata.utils.one_hot import to_one_hot_matrix
from multiprocessing import Queue


class BaseBatchWrapper:
    def __init__(self):
        pass

    @property
    def supports_workers(self):
        raise NotImplementedError
