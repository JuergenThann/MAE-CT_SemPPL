import logging
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

from .wandb_config import WandbConfig


@dataclass
class CliArgs:
    hp: str
    accelerator: str
    devices: str
    wait_for_devices: bool
    num_workers: int
    wandb_mode: str
    wandb_config: str
    wandb_resume_id: str
    cudnn_benchmark: bool
    cuda_profiling: bool
    testrun: bool
    minmodelrun: bool
    mindatarun: bool
    mindurationrun: bool
    name: str
    master_port: int
    sync_batchnorm: bool
    stage_idx: int
    datasets_were_preloaded: bool
    disable_flash_attention: bool
    skip_if_exists_in_wandb: bool

    def log(self):
        logging.info("------------------")
        logging.info(f"CLI ARGS")
        for key, value in vars(self).items():
            if value is not None:
                logging.info(f"{key}: {value}")


def _hp(hp):
    assert isinstance(hp, str)
    path = Path(hp).expanduser().with_suffix(".yaml")
    assert path.exists(), f"hp file '{hp}' doesn't exist"
    return hp


def _devices(devices):
    assert isinstance(devices, str)
    if not devices.isdigit():
        assert all(d.isdigit() for d in devices.split(",")), f"specify multiple devices as 0,1,2,3 (not {devices})"
    return devices


def _wandb_config(wandb_config):
    if wandb_config is not None:
        assert isinstance(wandb_config, str)
        path = (Path("wandb_configs").expanduser() / wandb_config).with_suffix(".yaml")
        assert path.exists(), f"wandb_config file '{path}' doesn't exist"
        return wandb_config


def parse_run_cli_args() -> CliArgs:
    parser = ArgumentParser()
    parser.add_argument("--hp", type=_hp, required=True)
    parser.add_argument("--accelerator", type=str, default="gpu", choices=["cpu", "gpu"])
    parser.add_argument("--devices", type=_devices)
    parser.add_argument("--wait_for_devices", action="store_true")
    parser.add_argument("--name", type=str)
    # dataloading
    parser.add_argument("--num_workers", type=int)
    # wandb
    parser.add_argument("--wandb_mode", type=str, choices=WandbConfig.MODES)
    parser.add_argument("--wandb_config", type=_wandb_config)
    parser.add_argument("--wandb_resume_id", type=str)
    # cudnn benchmark
    cudnn_benchmark_group = parser.add_mutually_exclusive_group()
    cudnn_benchmark_group.add_argument("--cudnn_benchmark", action="store_true")
    cudnn_benchmark_group.add_argument("--no_cudnn_benchmark", action="store_false", dest="cudnn_benchmark")
    cudnn_benchmark_group.set_defaults(cudnn_benchmark=None)
    # cuda profiling
    cuda_profiling_group = parser.add_mutually_exclusive_group()
    cuda_profiling_group.add_argument("--cuda_profiling", action="store_true")
    cuda_profiling_group.add_argument("--no_cuda_profiling", action="store_false", dest="cuda_profiling")
    cuda_profiling_group.set_defaults(cuda_profiling=None)
    # testrun
    testrun_group = parser.add_mutually_exclusive_group()
    testrun_group.add_argument("--testrun", action="store_true")
    testrun_group.add_argument("--minmodelrun", action="store_true")
    testrun_group.add_argument("--mindatarun", action="store_true")
    testrun_group.add_argument("--mindurationrun", action="store_true")
    # distributed
    parser.add_argument("--master_port", type=int)
    # distributed - syncbatchnorm
    sync_batchnorm_group = parser.add_mutually_exclusive_group()
    sync_batchnorm_group.add_argument("--sync_batchnorm", action="store_true")
    sync_batchnorm_group.add_argument("--no_sync_batchnorm", action="store_false", dest="sync_batchnorm")
    sync_batchnorm_group.set_defaults(sync_batchnorm=None)
    # resume run
    parser.add_argument("--stage_idx", type=int)
    # slurm
    parser.add_argument("--datasets_were_preloaded", action="store_true")
    # flash attention
    parser.add_argument("--disable_flash_attention", action="store_true")
    # skip the run if its name is already taken in wandb and the existing run is in state finished or running
    parser.add_argument("--skip_if_exists_in_wandb", action="store_true")

    return CliArgs(**vars(parser.parse_known_args()[0]))
