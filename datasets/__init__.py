from utils.factory import instantiate
from pathlib import Path
from kappadata.utils.logging import log
import shutil
import os


def dataset_from_kwargs(
        kind,
        dataset_config_provider,
        dataset_wrappers=None,
        sample_wrappers=None,
        batch_wrappers=None,
        **kwargs,
):
    dataset = instantiate(
        module_names=[f"datasets.{kind}"],
        type_names=[kind],
        dataset_config_provider=dataset_config_provider,
        **kwargs,
    )
    if dataset_wrappers is not None:
        assert isinstance(dataset_wrappers, list)
        for dataset_wrapper_kwargs in dataset_wrappers:
            dataset_wrapper_kind = dataset_wrapper_kwargs.pop("kind")
            dataset = instantiate(
                module_names=[
                    f"datasets.dataset_wrappers.{dataset_wrapper_kind}",
                    f"kappadata.wrappers.dataset_wrappers.{dataset_wrapper_kind}"
                ],
                type_names=[dataset_wrapper_kind],
                dataset=dataset,
                **dataset_wrapper_kwargs,
            )
    if sample_wrappers is not None:
        assert isinstance(sample_wrappers, list)
        for sample_wrapper_kwargs in sample_wrappers:
            sample_wrapper_kind = sample_wrapper_kwargs.pop("kind")
            dataset = instantiate(
                module_names=[
                    f"datasets.sample_wrappers.{sample_wrapper_kind}",
                    f"kappadata.wrappers.sample_wrappers.{sample_wrapper_kind}",
                ],
                type_names=[sample_wrapper_kind],
                dataset=dataset,
                **sample_wrapper_kwargs,
            )

    if batch_wrappers is not None:
        assert isinstance(batch_wrappers, list)
        instantiated_batch_wrappers = []
        for batch_wrapper_kwargs in batch_wrappers:
            batch_wrapper_kind = batch_wrapper_kwargs.pop("kind")
            instantiated_batch_wrappers.append(instantiate(
                module_names=[
                    f"datasets.batch_wrappers.{batch_wrapper_kind}"
                ],
                type_names=[batch_wrapper_kind],
                **batch_wrapper_kwargs,
            ))
    else:
        instantiated_batch_wrappers = None
    dataset.batch_wrappers = instantiated_batch_wrappers

    return dataset


def torchvision_dataset_from_kwargs(
    kind,
    dataset_config_provider,
    **kwargs,
):
    dataset = instantiate(
        module_names=[f"torchvision.datasets"],
        type_names=[kind],
        **kwargs,
    )
    return dataset


def _check_src_path(src_path):
    if src_path.exists() and src_path.is_dir():
        return True
    return False


# region modified copy from kappadata/copying/image_folder.py
def copy_folder_from_global_to_local(
    global_path,
    local_path,
    num_workers=0,
    log_fn=None
):
    if not isinstance(global_path, Path):
        global_path = Path(global_path).expanduser()
    if not isinstance(local_path, Path):
        local_path = Path(local_path).expanduser()

    # check src_path exists (src_path can be folder)
    src_path = global_path
    assert _check_src_path(src_path), f"invalid src_path (can be folder) '{src_path}'"

    # if dst_path exists:
    # - autocopy start/end file exists -> already copied -> do nothing
    # - autocopy start file exists && autocopy end file doesn't exist -> incomplete copy -> delete and copy again
    # - autocopy start file doesn't exists -> manually copied dataset -> do nothing
    dst_path = local_path
    start_copy_file = dst_path / "autocopy_start.txt"
    end_copy_file = dst_path / "autocopy_end.txt"
    if dst_path.exists():
        if start_copy_file.exists():
            if end_copy_file.exists():
                # already automatically copied -> do nothing
                log(log_fn, f"dataset was already automatically copied '{dst_path}'")
            else:
                # incomplete copy -> delete and copy again
                log(log_fn, f"found incomplete automatic copy in '{dst_path}' -> deleting folder")
                shutil.rmtree(dst_path)
                dst_path.mkdir()
        else:
            log(log_fn, f"using manually copied dataset '{dst_path}'")
    else:
        dst_path.mkdir(parents=True)

    # create start_copy_file
    with open(start_copy_file, "w") as f:
        f.write("this file indicates that an attempt to copy the dataset automatically was started")

    # copy
    if src_path.exists() and src_path.is_dir():
        # copy folders which contain the raw files (not zipped or anything)
        log(log_fn, f"copying folders of '{src_path}' to '{dst_path}'")
        # copy folder (dirs_exist_ok=True because dst_path is created for start_copy_file)
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    else:
        raise NotImplementedError

    # create end_copy_file
    with open(end_copy_file, "w") as f:
        f.write("this file indicates that the copying the dataset automatically was successful")

    log(log_fn, "finished copying data from global to local")
# endregion
