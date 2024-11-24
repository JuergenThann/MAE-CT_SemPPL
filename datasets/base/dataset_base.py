from kappadata import KDDataset

from providers.dataset_config_provider import DatasetConfigProvider
from utils.collator_from_kwargs import collator_from_kwargs
from utils.factory import create_collection, instantiate


class DatasetBase(KDDataset):
    def __init__(self, collators=None, dataset_config_provider: DatasetConfigProvider = None, dataloader=None, **kwargs):
        collators = create_collection(collators, collator_from_kwargs)
        super().__init__(collators=collators, **kwargs)
        self.dataset_config_provider = dataset_config_provider
        self.dataloader = dataloader
        self.batch_wrappers = None

    def __len__(self):
        raise NotImplementedError

    def getitem_x(self, idx, ctx=None):
        raise NotImplementedError

    # region classification
    def getshape_class(self):
        return self.n_classes,

    @property
    def n_classes(self):
        return None

    @property
    def is_multiclass(self):
        return False

    # endregion
