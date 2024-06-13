from kappadata import KDWrapper
import torch
import math


class SemisupervisedOversamplingWrapper(KDWrapper):
    def __init__(self, include_labeled_in_unlabeled: bool, unlabeled_to_labeled_ratio: int = 1, **kwargs):
        super().__init__(**kwargs)
        assert len(self.dataset) > 0 and self.dataset.n_classes > 1

        self.unlabeled_to_labeled_ratio = unlabeled_to_labeled_ratio
        self.indices = torch.arange(len(self.dataset), dtype=torch.int32)
        self.targets = torch.tensor([self.dataset.getitem_class(i) for i in range(len(self.dataset))])

        if include_labeled_in_unlabeled:
            for cl in range(self.dataset.n_classes):
                filtered_indices = self.indices[self.targets == cl]
                self.indices = torch.concatenate((
                    self.indices,
                    filtered_indices
                ))
                self.targets = torch.concatenate((
                    self.targets,
                    torch.ones(len(filtered_indices), dtype=torch.int32)*(-cl-1)
                ))

        unique_classes, unique_counts = torch.unique(self.targets, return_counts=True)
        # classes might have 0 samples
        unique_classes = unique_classes.tolist()
        unique_counts = unique_counts.tolist()
        class_counts = {cl: cnt for cl, cnt in zip(unique_classes, unique_counts)}

        labeled_count = torch.sum(torch.tensor([cnt for cl, cnt in class_counts.items() if cl >= 0]), dtype=torch.int32).item()
        unlabeled_count = torch.sum(torch.tensor([cnt for cl, cnt in class_counts.items() if cl < 0]), dtype=torch.int32).item()

        labeled_factor = 1
        unlabeled_factor = 1
        if labeled_count * self.unlabeled_to_labeled_ratio < unlabeled_count:
            labeled_factor = unlabeled_count / (labeled_count * self.unlabeled_to_labeled_ratio)
        elif labeled_count * self.unlabeled_to_labeled_ratio > unlabeled_count:
            unlabeled_factor = (labeled_count * self.unlabeled_to_labeled_ratio) / unlabeled_count

        # append miniority classes as long as they are not bigger than the majority class
        for cl in unique_classes:
            multiply_factor = (unlabeled_factor if cl < 0 else labeled_factor) - 1
            if multiply_factor == 0:
                continue
            # if class is not contained in dataset -> cant multiply sample
            if class_counts[cl] == 0:
                continue

            # get indices of samples with class to oversample
            sample_idxs = self.indices[self.targets == cl]
            num_samples_to_add = int(round(class_counts[cl] * multiply_factor, 0))
            new_indices = torch.tile(sample_idxs, (math.ceil(multiply_factor),))[:num_samples_to_add]
            self.indices = torch.concatenate((self.indices, new_indices))
            self.targets = torch.concatenate((self.targets, self.targets[new_indices]))

    def __len__(self):
        return len(self.indices)

    def getitem_x(self, idx, ctx=None):
        return self.dataset.getitem_x(self.indices[idx])

    def getitem_class(self, idx, ctx=None):
        return self.targets[idx]
