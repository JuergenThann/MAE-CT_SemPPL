import torch

from kappadata.datasets.kd_wrapper import KDWrapper


# modified copy of kappadata LabelSmoothingWrapper
class SemisupervisedLabelSmoothingWrapper(KDWrapper):
    def __init__(self, dataset, smoothing):
        super().__init__(dataset=dataset)
        assert isinstance(smoothing, (int, float)) and 0. <= smoothing <= 1.
        self.smoothing = smoothing

    def getitem_class(self, idx, ctx=None):
        y = self.dataset.getitem_class(idx, ctx)
        if self.smoothing == 0:
            return y
        assert isinstance(y, int) or (torch.is_tensor(y) and y.ndim == 0)
        n_classes = self.dataset.getdim_class()

        # semi supervised case (can't smooth missing labels)
        is_unlabeled = False
        if y < 0:
            is_unlabeled = True
            y = -y-1

        # binary case (label is scalar)
        if n_classes == 1:
            assert not is_unlabeled, 'semisupervised labels are not yet supported for the binary case'
            off_value = self.smoothing / 2
            if y > 0.5:
                return (y - off_value) * (-1 if is_unlabeled else 1)
            else:
                return (y + off_value) * (-1 if is_unlabeled else 1)

        # multi class (scalar -> vector)
        off_value = self.smoothing / n_classes
        on_value = 1. - self.smoothing + off_value
        y_vector = torch.full(size=(n_classes,), fill_value=off_value)
        y_vector[y] = on_value
        return y_vector * (-1 if is_unlabeled else 1)
