import logging
from copy import deepcopy

import torch
import wandb

from configs.wandb_config import WandbConfig
from distributed.config import is_rank0, get_world_size, get_nodes
from providers.config_providers.noop_config_provider import NoopConfigProvider
from providers.config_providers.primitive_config_provider import PrimitiveConfigProvider
from providers.config_providers.wandb_config_provider import WandbConfigProvider
from providers.stage_path_provider import StagePathProvider
from providers.summary_providers.noop_summary_provider import NoopSummaryProvider
from providers.summary_providers.primitive_summary_provider import PrimitiveSummaryProvider
from providers.summary_providers.wandb_summary_provider import WandbSummaryProvider
from utils.kappaconfig.util import remove_large_collections


def init_wandb(
        device: str,
        run_name: str,
        stage_hp: dict,
        wandb_config: WandbConfig,
        stage_path_provider: StagePathProvider,
        resume_id: str,
        account_name: str,
        tags: list,
        notes: str,
        group: str,
        group_tags: dict,
        ignore_stage_name: bool
):
    logging.info("------------------")
    logging.info(f"initializing wandb (mode={wandb_config.mode})")
    # os.environ["WANDB_SILENT"] = "true"

    # create config_provider & summary_provider
    if not is_rank0():
        config_provider = NoopConfigProvider()
        summary_provider = NoopSummaryProvider()
        return config_provider, summary_provider
    elif wandb_config.is_disabled:
        config_provider = PrimitiveConfigProvider(stage_path_provider=stage_path_provider)
        summary_provider = PrimitiveSummaryProvider(stage_path_provider=stage_path_provider)
    else:
        config_provider = WandbConfigProvider(stage_path_provider=stage_path_provider)
        summary_provider = WandbSummaryProvider(stage_path_provider=stage_path_provider)

    config = {
        "run_name": run_name,
        "stage_name": stage_path_provider.stage_name,
        **_lists_to_dict(remove_large_collections(stage_hp)),
    }
    if not wandb_config.is_disabled:
        wandb.login(host=wandb_config.host)
        logging.info(f"logged into wandb (host={wandb_config.host})")
        name = run_name or "None"
        if not ignore_stage_name and stage_path_provider.stage_name != "default_stage":
            name += f"/{stage_path_provider.stage_name}"
        wandb_id = resume_id or stage_path_provider.stage_id
        # can't group by tags -> with group tags you can (by adding it as a field to the config)
        # group_tags:
        #   augmentation: minimal
        #   ablation: warmup
        tags = tags or []
        if group_tags is not None and len(group_tags) > 0:
            logging.info(f"group tags:")
            for group_name, tag in group_tags.items():
                logging.info(f"  {group_name}: {tag}")
                assert tag not in tags, \
                    f"tag '{tag}' from group_tags is also in tags (group_tags={group_tags} tags={tags})"
                tags.append(tag)
                config[group_name] = tag
        if len(tags) > 0:
            logging.info(f"tags:")
            for tag in tags:
                logging.info(f"- {tag}")
        wandb.init(
            entity=wandb_config.entity,
            project=wandb_config.project,
            name=name,
            dir=str(stage_path_provider.stage_output_path),
            save_code=False,
            config=config,
            mode=wandb_config.mode,
            id=wandb_id,
            resume=resume_id is not None,
            tags=[str(tag) for tag in tags],  # ints need to be cast to string
            notes=notes,
            group=group or wandb_id,
        )
    config_provider.update(config)

    # log additional environment properties
    additional_config = {}
    if str(device) == "cpu":
        additional_config["device"] = "cpu"
    else:
        additional_config["device"] = torch.cuda.get_device_name(0)
    additional_config["dist/world_size"] = get_world_size()
    additional_config["dist/nodes"] = get_nodes()
    # hostname from static config which can be more descriptive than the platform.uname().node (e.g. account name)
    additional_config["dist/account_name"] = account_name
    config_provider.update(additional_config)

    return config_provider, summary_provider


def _lists_to_dict(root):
    """ wandb cant handle lists in configs -> transform lists into dicts with str(i) as key """
    #  (it will be displayed as [{"kind": "..."}, ...])
    root = deepcopy(root)
    return _lists_to_dicts_impl(dict(root=root))["root"]


def _lists_to_dicts_impl(root):
    if not isinstance(root, dict):
        return
    for k, v in root.items():
        if isinstance(v, list):
            root[k] = {str(i): vitem for i, vitem in enumerate(v)}
        elif isinstance(v, dict):
            root[k] = _lists_to_dicts_impl(root[k])
    return root


def finish_wandb(wandb_config: WandbConfig):
    if not is_rank0() or wandb_config.is_disabled:
        return
    wandb.finish()
