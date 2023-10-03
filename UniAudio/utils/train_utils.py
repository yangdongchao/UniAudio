import importlib
import random
import numpy as np
import torch
import warnings
import os
import time
import sys
import yaml
import json
import dataclasses
import logging
import torchaudio
import functools

import torch.utils.tensorboard as tensorboard
import typing as tp

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp import FullStateDictConfig

from pathlib import Path
from typing import Union
from torch import distributed as dist
from torch.optim.lr_scheduler import _LRScheduler
from utils.abs_scheduler import AbsBatchStepScheduler


def seed_everything(seed, cudnn_deterministic=False):
    """
    Function that sets seed for pseudo-random number generators in:
    pytorch, numpy, python.random
    
    Args:
        seed: the integer value seed for global random state
    """
    if seed is not None:
        print(f"Global seed set to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

def to_device(data, device=None, dtype=None, non_blocking=False, copy=False):
    """Change the device of object recursively"""
    if isinstance(data, dict):
        return {
            k: to_device(v, device, dtype, non_blocking, copy) for k, v in data.items()
        }
    elif dataclasses.is_dataclass(data) and not isinstance(data, type):
        return type(data)(
            *[
                to_device(v, device, dtype, non_blocking, copy)
                for v in dataclasses.astuple(data)
            ]
        )
    # maybe namedtuple. I don't know the correct way to judge namedtuple.
    elif isinstance(data, tuple) and type(data) is not tuple:
        return type(data)(
            *[to_device(o, device, dtype, non_blocking, copy) for o in data]
        )
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(v, device, dtype, non_blocking, copy) for v in data)
    elif isinstance(data, np.ndarray):
        return to_device(torch.from_numpy(data), device, dtype, non_blocking, copy)
    elif isinstance(data, torch.Tensor):
        return data.to(device, dtype, non_blocking, copy)
    else:
        return data

def setup_logging(rank, world_size, log_file=None):
    """Make logging setup with a given log level."""
    if log_file is not None:
        assert "RANK" in log_file, "'RANK' must be in log_file, if provided"
        time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        log_file = log_file.replace("RANK", "gpu" + str(rank) + f'_{time_stamp}')

    # Some third-party dependency would set the logging module. override it.
    root = logging.getLogger()
    list(map(root.removeHandler, root.handlers))
    list(map(root.removeFilter, root.filters))

    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format=f"%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] ({rank}/{world_size}) %(message)s",
    )

def yaml_no_alias_safe_dump(data, stream=None, **kwargs):
    """Safe-dump in yaml with no anchor/alias"""
    return yaml.dump(
        data, stream, allow_unicode=True, Dumper=NoAliasSafeDumper, **kwargs
    )

class NoAliasSafeDumper(yaml.SafeDumper):
    # Disable anchor/alias in yaml because looks ugly
    def ignore_aliases(self, data):
        return True

def maybe_resume_checkpoint(args, model, optimizer, scheduler, reporter, train_dl):
    if args.resume is not None and Path(args.resume).is_file():
        checkpoint = args.resume
        logging.info(f"Resume from the provided checkpoitn {args.resume}")
    else:
        ckpts = list(Path(args.exp_dir).glob("ep*.checkpoint"))
        if len(ckpts) == 0:
            logging.info("Training from a randomly initialized model")
            return
        else:
            ckpts.sort(key=lambda x: os.stat(str(x)).st_ctime)
            checkpoint = str(ckpts[-1])
            logging.info(f"Automatically resume from the latest checkpoint {checkpoint}")

    state_dict = torch.load(checkpoint, map_location='cpu')

    FSDP.set_state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=False),
    )

    # model
    model.load_state_dict(state_dict['model'])

    # optimizer, scheduler, reporter 
    optimizer.load_state_dict(
        FSDP.optim_state_dict_to_load(
            state_dict['optimizer'],
            model,
            optimizer,
        )
    )

    # scheduler, reporter
    scheduler.load_state_dict(state_dict['scheduler'])
    reporter.load_state_dict(state_dict['reporter'])

    train_dl.sampler.set_epoch(reporter.get_epoch() + 1)
    train_dl.sampler.refresh()
    del state_dict

def resume_for_inference(resume, exp_dir, model, device):
    if resume is not None:
        checkpoint = resume
        logging.info(f"Resume from the provided checkpoitn {resume}")
    else:
        ckpts = list(Path(exp_dir).glob("ep*.checkpoint"))
        if len(ckpts) == 0:
            raise ValueError("Model for resume is not provided and cannot be detected.")
        else:
            ckpts.sort(key=lambda x: os.stat(str(x)).st_ctime)
            checkpoint = str(ckpts[-1])
            logging.info(f"Automatically resume from the latest checkpoint {checkpoint}")

    state_dict = torch.load(checkpoint, map_location=device)['model']
    state_dict = { k.split("module.")[-1] if k.startswith("module.") else k: v
                   for k, v in state_dict.items()
                 }
    model.load_state_dict(state_dict)

def save_checkpoint(path, model, optimizer, scheduler, reporter):
    FSDP.set_state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=True),
    )

    d = {
          "model": model.state_dict(),
          "optimizer": FSDP.optim_state_dict(model, optimizer),
          "scheduler": scheduler.state_dict(),
          "reporter": reporter.state_dict(),
        }

    if dist.get_rank() == 0:
        logging.info(f"checkpoint {path} saved")
        torch.save(d, path)

class WarmupLR(_LRScheduler, AbsBatchStepScheduler):
    """The WarmupLR scheduler

    This scheduler is almost same as NoamLR Scheduler except for following difference:

    NoamLR:
        lr = optimizer.lr * model_size ** -0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    WarmupLR:
        lr = optimizer.lr * warmup_step ** 0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)

    Note that the maximum lr equals to optimizer.lr in this scheduler.

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: Union[int, float] = 25000,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps

        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps})"

    def get_lr(self):
        step_num = self.last_epoch + 1
        return [
            lr
            * self.warmup_steps**0.5
            * min(step_num**-0.5, step_num * self.warmup_steps**-1.5)
            for lr in self.base_lrs
        ]

def attention_mask(loss_mask, prefix_lm=True):
    """
    Generate the attention mask from the loss mask,
    where the loss mask is in the format [Batch, Length].
    Usually, the loss mask would look like:
      <False> ... <True> ... <False>, which represents the
    prefix, the target sequence and padding respectively.

    This function generates the mask for multi-head attention,
    which is in the shape of [Batch, Length, Length] and features:
    (1) the prefix entries can see all each other, if prefix_lm,
        otherwise causal;
    (2) the target entries are causal to each other and can see all
        prefix entries;
    (3) the padding entries can neither been seen nor see all other
        entries.
    """

    # basic preparation
    device = loss_mask.device
    batch_size, q_len = loss_mask.size()
    axis = torch.arange(q_len).to(device)
    # find the start and end time indices of loss duration
    start = axis.unsqueeze(0).masked_fill(~loss_mask, 1e8).min(dim=1).values
    end = axis.unsqueeze(0).masked_fill(~loss_mask, -1e8).max(dim=1).values
    # we strictly require that there is only one continuous True segment
    # for each example in the loss_mask:
    assert torch.all(end - start == loss_mask.int().sum(dim=-1) - 1)

    # (1) make it causal
    mask = (axis.unsqueeze(1) >= axis.unsqueeze(0)).repeat(batch_size, 1, 1)
    # (2) allow non-causaility in prefix part, if prefix_lm
    if prefix_lm:
        mask = torch.where(start.view(batch_size, 1, 1) > axis.view(1, 1, q_len),
                       True, mask)

    # (3) kill the padding
    mask = torch.where(end.view(batch_size, 1, 1) < axis.view(1, 1, q_len),
                       False, mask)

    return mask

def str2bool(x):
    if x == "true" or x == "True":
        return True
    elif x == "false" or x == "False":
        return False
    else:
        raise NotImplementedError

def find_data_jsons(patterns, rank=None, world_size=None):
    #print('patterns ', patterns)
    ret = []
    if rank is None and world_size is None:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    for pattern in patterns:
        logging.info(f"search pattern {pattern}")
        pattern = pattern.replace("ALL","*")
        dir_path, name = Path(pattern).parents[0], Path(pattern).name
        all_files = list(Path(dir_path).glob(name))
        #print('all_files ', all_files)
        assert len(all_files) % world_size == 0
        all_files.sort()
        all_files = all_files[rank::world_size]
        all_files = [str(f) for f in all_files]
        ret = ret + all_files
    
    assert len(ret) > 0
    return ret
