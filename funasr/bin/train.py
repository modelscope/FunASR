#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import os
import sys
import torch
import torch.nn as nn
import hydra
import logging
import time
import argparse
from io import BytesIO

from contextlib import nullcontext
import torch.distributed as dist

from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.algorithms.join import Join
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tensorboardX import SummaryWriter
from funasr.train_utils.average_nbest_models import average_checkpoints

from funasr.register import tables
from funasr.optimizers import optim_classes
from funasr.train_utils.trainer import Trainer
from funasr.schedulers import scheduler_classes
from funasr.train_utils.initialize import initialize
from funasr.download.download_model_from_hub import download_model
from funasr.models.lora.utils import mark_only_lora_as_trainable
from funasr.train_utils.set_all_random_seed import set_all_random_seed
from funasr.train_utils.load_pretrained_model import load_pretrained_model
from funasr.utils.misc import prepare_model_dir
from funasr.train_utils.model_summary import model_summary
from funasr import AutoModel


@hydra.main(config_name=None, version_base=None)
def main_hydra(kwargs: DictConfig):
    if kwargs.get("debug", False):
        import pdb

        pdb.set_trace()

    assert "model" in kwargs
    if "model_conf" not in kwargs:
        logging.info("download models from model hub: {}".format(kwargs.get("hub", "ms")))
        kwargs = download_model(is_training=kwargs.get("is_training", True), **kwargs)

    main(**kwargs)


def main(**kwargs):

    # set random seed
    set_all_random_seed(kwargs.get("seed", 0))
    torch.backends.cudnn.enabled = kwargs.get("cudnn_enabled", torch.backends.cudnn.enabled)
    torch.backends.cudnn.benchmark = kwargs.get("cudnn_benchmark", torch.backends.cudnn.benchmark)
    torch.backends.cudnn.deterministic = kwargs.get("cudnn_deterministic", True)
    # open tf32
    torch.backends.cuda.matmul.allow_tf32 = kwargs.get("enable_tf32", True)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        tables.print()
    # Check if we are using DDP or FSDP
    use_ddp = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    use_fsdp = kwargs.get("use_fsdp", False)
    # use_ddp = False if use_fsdp else use_fsdp
    if use_ddp or use_fsdp:
        dist.init_process_group(backend=kwargs.get("backend", "nccl"), init_method="env://")
        torch.cuda.set_device(local_rank)

    logging.info("Build model, frontend, tokenizer")
    device = kwargs.get("device", "cuda")
    kwargs["device"] = "cpu"
    model = AutoModel(**kwargs)

    # save config.yaml
    if (
        (use_ddp or use_fsdp)
        and dist.get_rank() == 0
        or not (use_ddp or use_fsdp)
        and local_rank == 0
    ):
        prepare_model_dir(**kwargs)

    # parse kwargs
    kwargs = model.kwargs
    kwargs["device"] = device
    tokenizer = kwargs["tokenizer"]
    frontend = kwargs["frontend"]
    model = model.model
    del kwargs["model"]

    # freeze_param
    freeze_param = kwargs.get("freeze_param", None)
    if freeze_param is not None:
        if "," in freeze_param:
            freeze_param = eval(freeze_param)
        if not isinstance(freeze_param, (list, tuple)):
            freeze_param = (freeze_param,)
        logging.info("freeze_param is not None: %s", freeze_param)
        for t in freeze_param:
            for k, p in model.named_parameters():
                if k.startswith(t + ".") or k == t:
                    logging.info(f"Setting {k}.requires_grad = False")
                    p.requires_grad = False
    if local_rank == 0:
        logging.info(f"{model_summary(model)}")

    if use_ddp:
        model = model.cuda(local_rank)
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=kwargs.get("train_conf", {}).get(
                "find_unused_parameters", False
            ),
        )
    elif use_fsdp:
        # model = FSDP(model).cuda(local_rank)

        def custom_auto_wrap_policy(
            module: nn.Module,
            recurse: bool,
            nonwrapped_numel: int,
            # Additional custom arguments
            min_num_params: int = int(1e8),
        ) -> bool:
            # 根据自定义逻辑决定是否包装模块
            is_large = unwrapped_params >= min_num_params
            requires_grad_uniform = len({p.requires_grad for p in module.parameters()}) == 1
            return is_large and requires_grad_uniform

        # Configure a custom `min_num_params`
        my_auto_wrap_policy = functools.partial(custom_auto_wrap_policy, min_num_params=int(1e5))
        torch.cuda.set_device(local_rank)
        model = FSDP(
            model,
            auto_wrap_policy=custom_auto_wrap_policy,
            mixed_precision=None,
            device_id=torch.cuda.current_device(),
        )
    else:
        model = model.to(device=kwargs.get("device", "cuda"))

    kwargs["device"] = next(model.parameters()).device

    # optim
    logging.info("Build optim")
    optim = kwargs.get("optim", "adam")
    assert optim in optim_classes
    optim_class = optim_classes.get(optim)
    optim = optim_class(model.parameters(), **kwargs.get("optim_conf"))

    # scheduler
    logging.info("Build scheduler")
    scheduler = kwargs.get("scheduler", "warmuplr")
    assert scheduler in scheduler_classes
    scheduler_class = scheduler_classes.get(scheduler)
    scheduler = scheduler_class(optim, **kwargs.get("scheduler_conf"))

    # dataset
    logging.info("Build dataloader")
    dataloader_class = tables.dataloader_classes.get(
        kwargs["dataset_conf"].get("dataloader", "DataloaderMapStyle")
    )
    dataloader = dataloader_class(**kwargs)
    # dataloader_tr, dataloader_val = dataloader_class(**kwargs)
    trainer = Trainer(
        local_rank=local_rank,
        use_ddp=use_ddp,
        use_fsdp=use_fsdp,
        device=kwargs["device"],
        output_dir=kwargs.get("output_dir", "./exp"),
        **kwargs.get("train_conf"),
    )

    scaler = GradScaler(enabled=trainer.use_fp16) if trainer.use_fp16 else None
    scaler = ShardedGradScaler(enabled=trainer.use_fp16) if trainer.use_fsdp else scaler

    trainer.resume_checkpoint(
        model=model,
        optim=optim,
        scheduler=scheduler,
        scaler=scaler,
    )

    tensorboard_dir = os.path.join(kwargs.get("output_dir"), "tensorboard")
    os.makedirs(tensorboard_dir, exist_ok=True)
    try:
        writer = SummaryWriter(tensorboard_dir)  # if trainer.rank == 0 else None
    except:
        writer = None

    dataloader_tr, dataloader_val = None, None
    for epoch in range(trainer.start_epoch, trainer.max_epoch):
        time1 = time.perf_counter()

        for data_split_i in range(trainer.start_data_split_i, dataloader.data_split_num):
            time_slice_i = time.perf_counter()
            dataloader_tr, dataloader_val = dataloader.build_iter(
                epoch, data_split_i=data_split_i, start_step=trainer.start_step
            )

            trainer.train_epoch(
                model=model,
                optim=optim,
                scheduler=scheduler,
                scaler=scaler,
                dataloader_train=dataloader_tr,
                dataloader_val=dataloader_val,
                epoch=epoch,
                writer=writer,
                data_split_i=data_split_i,
                data_split_num=dataloader.data_split_num,
                start_step=trainer.start_step,
            )
            trainer.start_step = 0

            device = next(model.parameters()).device
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()

            time_escaped = (time.perf_counter() - time_slice_i) / 3600.0
            logging.info(
                f"rank: {local_rank}, "
                f"time_escaped_epoch: {time_escaped:.3f} hours, "
                f"estimated to finish {dataloader.data_split_num} data_slices, remaining: {dataloader.data_split_num-data_split_i} slices, {(dataloader.data_split_num-data_split_i)*time_escaped:.3f} hours, "
                f"epoch: {trainer.max_epoch - epoch} epochs, {((trainer.max_epoch - epoch - 1)*dataloader.data_split_num + dataloader.data_split_num-data_split_i)*time_escaped:.3f} hours\n"
            )

        trainer.start_data_split_i = 0
        trainer.validate_epoch(
            model=model, dataloader_val=dataloader_val, epoch=epoch + 1, writer=writer
        )
        scheduler.step()
        trainer.step_in_epoch = 0
        trainer.save_checkpoint(
            epoch + 1, model=model, optim=optim, scheduler=scheduler, scaler=scaler
        )

        time2 = time.perf_counter()
        time_escaped = (time2 - time1) / 3600.0
        logging.info(
            f"rank: {local_rank}, "
            f"time_escaped_epoch: {time_escaped:.3f} hours, "
            f"estimated to finish {trainer.max_epoch} "
            f"epoch: {(trainer.max_epoch - epoch) * time_escaped:.3f} hours\n"
        )
        trainer.train_acc_avg = 0.0
        trainer.train_loss_avg = 0.0

    if trainer.rank == 0:
        average_checkpoints(trainer.output_dir, trainer.avg_nbest_model)

    trainer.close()


if __name__ == "__main__":
    main_hydra()
