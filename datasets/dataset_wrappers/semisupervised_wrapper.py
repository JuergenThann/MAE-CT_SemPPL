from kappadata import KDWrapper
import torch
from torch.utils.data.dataset import Subset
from operator import itemgetter


class SemiSupervisedWrapper(KDWrapper):
    def __init__(self, labeled_percentage, **kwargs):
        super().__init__(**kwargs)
        assert(0 <= labeled_percentage <= 100)
        self.targets = torch.Tensor(self.dataset.targets)
        if isinstance(self.dataset, Subset):
            self.targets = self.targets[self.dataset.indices]
        classes, class_counts = torch.unique(self.targets, return_counts=True)
        for i, class_ in enumerate(classes):
            num_labeled = int(class_counts[i] * (labeled_percentage / 100))
            all_class_indices = torch.nonzero(self.targets == class_, as_tuple=True)
            unlabeled_class_indices = tuple(x[num_labeled:] for x in all_class_indices)
            self.targets[unlabeled_class_indices] = -self.targets[unlabeled_class_indices] - 1

    def getitem_x(self, idx, ctx=None):
        return self.dataset.getitem_x(idx)

    def getitem_class(self, idx, ctx=None):
        return self.targets[idx]
