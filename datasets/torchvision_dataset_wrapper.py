from .base.xtransform_dataset_base import XTransformDatasetBase
from . import torchvision_dataset_from_kwargs, copy_folder_from_global_to_local
from distributed.config import barrier, is_data_rank0
from utils.num_worker_heuristic import get_fair_cpu_count
import torch


class TorchvisionDatasetWrapper(XTransformDatasetBase):
    def __init__(self,
                 dataset_config_provider,
                 dataset_identifier,
                 torchvision_args,
                 num_classes,
                 is_multiclass=False,
                 **kwargs):
        super().__init__(**kwargs)

        # region modified copy from datasets/base/image_folder.py
        global_root = dataset_config_provider.get_global_dataset_path(dataset_identifier)
        source_mode = dataset_config_provider.get_data_source_mode(dataset_identifier)
        # use local by default
        local_root = None
        if source_mode in [None, "local"]:
            local_root = dataset_config_provider.get_local_dataset_path()

        # get relative path (e.g. train)
        if local_root is None:
            # load data from global_root
            assert global_root is not None and global_root.exists(), f"invalid global_root '{global_root}'"
            source_root = global_root
            self.logger.info(f"data_source (global): '{source_root}'")
        else:
            # load data from local_root
            source_root = local_root / global_root.name
            if is_data_rank0():
                # copy data from global to local
                self.logger.info(f"data_source (global): '{global_root}'")
                self.logger.info(f"data_source (local): '{source_root}'")
                copy_folder_from_global_to_local(
                    global_path=global_root,
                    local_path=source_root,
                    # on karolina 5 was already too much for a single GPU
                    # "A worker process managed by the executor was unexpectedly terminated.
                    # This could be caused by a segmentation fault while calling the function or by an
                    # excessive memory usage causing the Operating System to kill the worker."
                    num_workers=min(10, get_fair_cpu_count()),
                    log_fn=self.logger.info,  # TODO
                )
            barrier()
        # endregion

        torchvision_args["root"] = str(source_root.parent)
        # torchvision_args["transform"] = ToTensor()
        self.dataset = torchvision_dataset_from_kwargs(
            dataset_config_provider=dataset_config_provider,
            **torchvision_args
        )
        self._num_classes = num_classes
        self._is_multiclass = is_multiclass

    def __len__(self):
        return len(self.dataset)

    def getitem_x(self, idx, ctx=None):
        x, _ = self.dataset[idx]
        x = self.x_transform(x, ctx=ctx)
        return x

    def getitem_class(self, idx, ctx=None):
        return self.dataset.targets[idx]

    @property
    def targets(self):
        return self.dataset.targets

    @property
    def class_names(self):
        return self.dataset.classes

    # region classification
    @property
    def n_classes(self):
        return self._num_classes

    @property
    def is_multiclass(self):
        return self._is_multiclass

    # endregion
