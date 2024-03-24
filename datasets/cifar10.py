from pathlib import Path

from datasets.base.image_folder import ImageFolder


class Cifar10(ImageFolder):
    CLASS_TO_IDX = {
        'airplane': 0,
        'automobile': 1,
        'bird': 2,
        'cat': 3,
        'deer': 4,
        'dog': 5,
        'frog': 6,
        'horse': 7,
        'ship': 8,
        'truck': 9
    }

    def __init__(self, split=None, **kwargs):
        if split == "test":
            split = "val"
        assert split in ["train", "val"]
        self.train = split == "train"
        super().__init__(**kwargs)

    def get_dataset_identifier(self):
        """ returns an identifier for the dataset (used for retrieving paths from dataset_config_provider) """
        return 'cifar10'

    def get_relative_path(self):
        return Path(self.split)

    def get_class_to_idx(self):
        return self.CLASS_TO_IDX

    def __str__(self):
        return f"{self.get_dataset_identifier()}.{self.split}"
