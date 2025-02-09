from torch.utils.data import DataLoader
from typing import Any, Callable, TypeVar, List, Optional

T = TypeVar('T')
_collate_fn_t = Callable[[List[T]], Any]


class DataLoaderWrapper:
    def __init__(self, dataloader: DataLoader, batch_wrappers: Optional[_collate_fn_t] = None):
        self.dataloader = dataloader
        self.batch_wrappers = batch_wrappers

    def __getattr__(self, name):
        return getattr(self.dataloader, name)

    def __setattr__(self, name, value):
        if name in ['dataloader', 'batch_wrappers']:
            object.__setattr__(self, name, value)
        setattr(self.dataloader, name, value)

    def __delattr__(self, name):
        if name in self.__dict__.keys():
            object.__delattr__(self, name)
        delattr(self.dataloader, name)

    def __iter__(self):
        base_iter = self.dataloader.__iter__()
        wrapped_iter = _DataLoaderWrapperIter(self.dataloader, base_iter, self.batch_wrappers)
        return wrapped_iter


class _DataLoaderWrapperIter:
    def __init__(self, dataloader, base_iter, batch_wrappers):
        self.dataloader = dataloader
        self.base_iter = base_iter
        self.batch_wrappers = batch_wrappers

    def __iter__(self):
        return self

    def __next__(self):
        return_ctx = self.dataloader.dataset.return_ctx
        batch = next(self.base_iter)
        ctx = None
        if return_ctx:
            batch, ctx = batch

        for batch_wrapper in self.batch_wrappers:
            if return_ctx:
                batch, ctx = batch_wrapper(batch=batch, dataset_mode=self.dataloader.dataset.mode,
                                           dataset_key=self.dataloader.dataset.dataset_key, ctx=ctx)
            else:
                batch = batch_wrapper(batch=batch, dataset_mode=self.dataloader.dataset.mode,
                                      dataset_key=self.dataloader.dataset.dataset_key)

        if return_ctx:
            return batch, ctx
        return batch
