import logging
import os

import kappaprofiler as kp
import torch
import wandb
import psutil
from torch.distributed import broadcast_object_list
from wandb.util import generate_id

from configs.cli_args import parse_run_cli_args
from configs.static_config import StaticConfig
from configs.util import cliarg_or_staticvalue
from distributed.config import barrier, get_rank, get_local_rank, get_world_size, is_managed, is_rank0, is_distributed
from distributed.run import run_single_or_multiprocess, run_managed
from distributed.gather import all_reduce_sum_grad
from providers.stage_path_provider import StagePathProvider
from train_stage import train_stage
from utils.kappaconfig.util import get_run_hp, save_unresolved_hp
from utils.kappaconfig.util import get_stage_hp_list, get_max_batch_sizes_from_cli
from utils.kappaconfig.util import get_stage_ids_from_cli
from utils.logging_util import add_global_handlers, log_from_all_ranks
from utils.pytorch_cuda_timing import cuda_start_event, cuda_end_event
from utils.version_check import check_versions
from utils.wandb_utils import get_full_run_name, check_exists_in_wandb, get_wandb_config
import time
from pathlib import Path
import atexit
import platform
import tempfile
from signal import signal, SIGQUIT, SIGABRT, SIGTERM
from functools import partial
from random import randint


def main_single(device):
    cli_args = parse_run_cli_args()
    static_config = StaticConfig(uri="static_config.yaml", datasets_were_preloaded=cli_args.datasets_were_preloaded)
    add_global_handlers(log_file_uri=None)
    with log_from_all_ranks():
        logging.info(f"initialized process rank={get_rank()} local_rank={get_local_rank()} pid={os.getpid()}")
    barrier()
    logging.info(f"initialized {get_world_size()} processes")

    # CUDA_LAUNCH_BLOCKING=1 for debugging
    # os.environ["CUDA_LAUNCH_BLOCKING"] = str(1)

    # save hp file to temporary file, so it cannot be changed anymore
    temp_dir = tempfile.gettempdir()
    temp_hp_path = os.path.join(temp_dir, f'hp_unresolved_{os.getpid()}.yaml')
    if is_rank0():
        save_unresolved_hp(cli_args.hp, temp_hp_path)
        for s in [SIGQUIT, SIGABRT, SIGTERM]:
            signal(s, partial(os.remove, path=temp_hp_path))
        atexit.register(os.remove, temp_hp_path)
    barrier()

    # cudnn
    if cli_args.accelerator == "gpu":
        if cliarg_or_staticvalue(cli_args.cudnn_benchmark, static_config.default_cudnn_benchmark):
            torch.backends.cudnn.benchmark = True
            assert not static_config.default_cudnn_deterministic, "cudnn_benchmark can make things non-deterministic"
        else:
            logging.warning(f"disabled cudnn benchmark")
            if static_config.default_cudnn_deterministic:
                torch.backends.cudnn.deterministic = True
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                logging.warning(f"enabled cudnn deterministic")

    # profiling
    if cli_args.accelerator == "gpu":
        if cliarg_or_staticvalue(cli_args.cuda_profiling, static_config.default_cuda_profiling):
            kp.setup_async(cuda_start_event, cuda_end_event)
            logging.info(f"initialized profiler to call sync cuda")
    else:
        kp.setup_async_as_sync()

    # parse stages
    run_hp = get_run_hp(temp_hp_path)
    stage_names, stage_hp_list, ignore_specific_stage_names = get_stage_hp_list(
        run_hp,
        template_path=".",
        testrun=cli_args.testrun,
        minmodelrun=cli_args.minmodelrun,
        mindatarun=cli_args.mindatarun,
        mindurationrun=cli_args.mindurationrun,
    )
    max_batch_sizes = get_max_batch_sizes_from_cli()

    # parse stage_ids from cli (required in case of starting not from the first stage)
    stage_ids = get_stage_ids_from_cli()
    # check that stage_ids actually exist
    for stage_name, stage_id in stage_ids.items():
        stage_path_provider = StagePathProvider(
            output_path=static_config.output_path,
            stage_name=stage_name,
            stage_id=stage_id,
        )
        assert stage_path_provider.stage_output_path_exists, \
            f"invalid stage_name ({stage_name}) or invalid stage_id ({stage_id})"

    device_id = os.environ["CUDA_VISIBLE_DEVICES"]
    hostname = platform.uname().node
    job_name = f'{hostname}_dev{device_id}.lock'
    job_file = Path(static_config.output_path, '.jobs', job_name)
    if not job_file.parent.exists():
        job_file.parent.mkdir(parents=True)
    if job_file.exists():
        waited_for_device = False
        with job_file.open('r') as fs:
            job_ids = [int(i) for i in fs.read().strip().split(',')]
        last_job_id = job_ids[-1]
        current_job_id = job_ids[0]
        job_id = last_job_id + 1
        job_ids.append(job_id)
        with job_file.open('w') as fs:
            fs.write(','.join(str(i) for i in job_ids))
    else:
        waited_for_device = True
        current_job_id = last_job_id = job_id = 1
        job_ids = [job_id]

    with job_file.open('w') as fs:
        fs.write(','.join(str(i) for i in job_ids))

    for s in [SIGQUIT, SIGABRT, SIGTERM]:
        signal(s, partial(finalize_job_after_signal, job_file=job_file, job_id=job_id))
    atexit.register(finalize_job, job_file, job_id)

    if cli_args.wait_for_devices and cli_args.accelerator == 'gpu' and not waited_for_device:
        with log_from_all_ranks():
            logging.info(f"{os.getpid()}: {job_id=}, {current_job_id=}, {last_job_id=}...")
            logging.info(f"{os.getpid()}: Waiting for other processes on device {device_id} to finish...")

        while not waited_for_device:
            time.sleep(60)
            with job_file.open('r') as fs:
                job_ids = [int(i) for i in fs.read().strip().split(',')]
            logging.info(f"{os.getpid()}: {device_id=}, {job_id=}, {job_ids=}...")
            if job_id == job_ids[0]:
                waited_for_device = True

        with log_from_all_ranks():
            logging.info(f"{os.getpid()}: Device ({device_id}) is now ready.")

    if is_rank0() and cli_args.skip_if_exists_in_wandb:
        sleep_time = randint(1, 120)
        logging.info(
            f'Sleeping {sleep_time}s (randomly chosen between 1s and 120s) to avoid race conditions due to "--skip_if_exists_in_wandb".')
        time.sleep(sleep_time)

    # TODO the logging for this is not ideal
    for i, (stage_name, stage_hp) in enumerate(zip(stage_names, stage_hp_list)):
        wandb_exit_code = None
        run_name = stage_name
        try:
            # run only specific stage (defined by cli arg)
            stage_idx_cliarg = cli_args.stage_idx
            if stage_idx_cliarg is not None and stage_idx_cliarg != i:
                continue

            # generate stage_id and sync across devices
            stage_id = generate_id()
            if is_distributed():
                object_list = [stage_id] if is_rank0() else [None]
                broadcast_object_list(object_list)
                stage_id = object_list[0]

            # train stages
            if ignore_specific_stage_names:
                stage_name = stage_hp.get("stage_name", "default_stage")

            # save copy of hp in case it changes while waiting for GPU
            stage_path_provider = StagePathProvider(
                output_path=static_config.output_path,
                model_path=static_config.model_path,
                stage_name=stage_name,
                stage_id=stage_id,
            )

            run_name = get_full_run_name(cli_args, stage_hp, stage_path_provider)
            wandb_config = get_wandb_config(stage_hp, cli_args, static_config)

            skip_run = False
            if cli_args.skip_if_exists_in_wandb and not wandb_config.is_disabled and is_rank0():
                # check if run with same name and state finished or running already exists in WandB. If so, skip.
                run_exists = check_exists_in_wandb(run_name, wandb_config)
                if run_exists:
                    logging.info(f'Run "{run_name}" already exists in WandB - skipping.')
                    skip_run = True
            skip_run = all_reduce_sum_grad(1 if is_rank0() and skip_run else 0).item()
            if skip_run:
                continue

            if is_rank0():
                save_unresolved_hp(temp_hp_path, stage_path_provider.stage_output_path / "hp_unresolved.yaml")
            barrier()

            max_batch_size = max_batch_sizes[stage_name] if stage_name in max_batch_sizes else None
            train_stage(
                stage_hp=stage_hp,
                static_config=static_config,
                cliargs=cli_args,
                device=device,
                stage_name=stage_name,
                stage_id=stage_id,
                max_batch_size=max_batch_size,
                previous_stage_ids=stage_ids,
                wandb_config=wandb_config,
                run_name=run_name
            )
            # remember stage_id for next stages
            stage_ids[stage_name] = stage_id
        except KeyboardInterrupt:
            logging.exception(f"exception on run {run_name}, stage {stage_name}")
            wandb_exit_code = -2
        except:
            logging.exception(f"exception on run {run_name}, stage {stage_name}")
            wandb_exit_code = -1

        proc = psutil.Process()
        logging.info(f'open file handles after training: {proc.open_files()}')

        if wandb_exit_code is not None:
            wandb.finish(exit_code=wandb_exit_code)
            # keyboard interrupt -> cancel all runs
            if wandb_exit_code == -2:
                break
        else:
            wandb.finish()


def finalize_job_after_signal(signum, frame, job_file, job_id):
    finalize_job(job_file, job_id)


def finalize_job(job_file, job_id):
    if job_file.exists():
        with job_file.open('r') as fs:
            job_ids = [int(i) for i in fs.read().strip().split(',')]

        if job_id in job_ids:
            job_ids.remove(job_id)
            if len(job_ids) == 0:
                job_file.unlink()
            else:
                with job_file.open('w') as fs:
                    fs.write(','.join(str(i) for i in job_ids))


def main():
    # parse cli_args immediately for fast cli_args validation
    cli_args = parse_run_cli_args()
    static_config = StaticConfig(uri="static_config.yaml", datasets_were_preloaded=cli_args.datasets_were_preloaded)
    # initialize loggers for setup (seperate id)
    add_global_handlers(log_file_uri=None)

    if is_managed():
        run_managed(
            accelerator=cli_args.accelerator,
            devices=cli_args.devices,
            main_single=main_single,
        )
    else:
        run_single_or_multiprocess(
            accelerator=cli_args.accelerator,
            devices=cli_args.devices,
            main_single=main_single,
            master_port=cli_args.master_port or static_config.master_port,
            mig_devices=static_config.mig_config,
        )


if __name__ == "__main__":
    # check instantly to skip the whole multiprocessing setup in case of wrong versions
    check_versions(verbose=False)
    main()
