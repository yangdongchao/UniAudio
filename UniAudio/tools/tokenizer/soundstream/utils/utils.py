import dataclasses
import glob
import importlib
import random
import numpy as np
import torch
import warnings
import os
import time
import torch.utils.tensorboard as tensorboard
from torch import distributed as dist
import sys
import yaml
import json
import re
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def seed_everything(seed, cudnn_deterministic=False):
    """
    Function that sets seed for pseudo-random number generators in:
    pytorch, numpy, python.random
    
    Args:
        seed: the integer value seed for global random state
    """
    if seed is not None:
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

def is_primary():
    return get_rank() == 0


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0

    return dist.get_rank()


def load_yaml_config(path):
    with open(path) as f:
        config = yaml.full_load(f)
    return config


def save_config_to_yaml(config, path):
    assert path.endswith('.yaml')
    with open(path, 'w') as f:
        f.write(yaml.dump(config))
        f.close()


def save_dict_to_json(d, path, indent=None):
    json.dump(d, open(path, 'w'), indent=indent)


def load_dict_from_json(path):
    return json.load(open(path, 'r'))


def write_args(args, path):
    args_dict = dict((name, getattr(args, name)) for name in dir(args)if not name.startswith('_'))
    with open(path, 'a') as args_file:
        args_file.write('==> torch version: {}\n'.format(torch.__version__))
        args_file.write('==> cudnn version: {}\n'.format(torch.backends.cudnn.version()))
        args_file.write('==> Cmd:\n')
        args_file.write(str(sys.argv))
        args_file.write('\n==> args:\n')
        for k, v in sorted(args_dict.items()):
            args_file.write('  %s: %s\n' % (str(k), str(v)))
        args_file.close()


class Logger(object):
    def __init__(self, args):
        self.args = args
        self.save_dir = args.log_dir
        self.is_primary = is_primary()
        
        if self.is_primary:
            os.makedirs(self.save_dir, exist_ok=True)
            
            # save the args and config
            self.config_dir = os.path.join(self.save_dir, 'configs')
            os.makedirs(self.config_dir, exist_ok=True)
            file_name = os.path.join(self.config_dir, 'args.txt')
            write_args(args, file_name)

            log_dir = os.path.join(self.save_dir, 'logs')
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            self.text_writer = open(os.path.join(log_dir, 'log.txt'), 'a') # 'w')
            if args.tensorboard:
                self.log_info('using tensorboard')
                self.tb_writer = torch.utils.tensorboard.SummaryWriter(log_dir=log_dir) # tensorboard.SummaryWriter(log_dir=log_dir)
            else:
                self.tb_writer = None

    def save_config(self, config):
        if self.is_primary:
            save_config_to_yaml(config, os.path.join(self.config_dir, 'config.yaml'))

    def log_info(self, info, check_primary=True):
        if self.is_primary or (not check_primary):
            print(info)
            if self.is_primary:
                info = str(info)
                time_str = time.strftime('%Y-%m-%d-%H-%M')
                info = '{}: {}'.format(time_str, info)
                if not info.endswith('\n'):
                    info += '\n'
                self.text_writer.write(info)
                self.text_writer.flush()

    def add_scalar(self, **kargs):
        """Log a scalar variable."""
        if self.is_primary:
            if self.tb_writer is not None:
                self.tb_writer.add_scalar(**kargs)

    def add_scalars(self, **kargs):
        """Log a scalar variable."""
        if self.is_primary:
            if self.tb_writer is not None:
                self.tb_writer.add_scalars(**kargs)

    def add_image(self, **kargs):
        """Log a scalar variable."""
        if self.is_primary:
            if self.tb_writer is not None:
                self.tb_writer.add_image(**kargs)

    def add_images(self, **kargs):
        """Log a scalar variable."""
        if self.is_primary:
            if self.tb_writer is not None:
                self.tb_writer.add_images(**kargs)

    def close(self):
        if self.is_primary:
            self.text_writer.close()
            self.tb_writer.close()


def cal_model_size(model, name=""):

    all_size = sum(p.numel() for p in model.parameters())/1024.0/1024.0
    return f'Model size of {name}: {all_size:.3f} MB'

    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024

    return f'Model size of {name}: {all_size:.3f} MB'
    # print(f'Model size of {name}: {all_size:.3f}MB')
    # return (param_size, param_sum, buffer_size, buffer_sum, all_size)


def load_obj(obj_path: str, default_obj_path: str = ''):
    """ Extract an object from a given path.
    Args:
        obj_path: Path to an object to be extracted, including the object name.
            e.g.: `src.trainers.meta_trainer.MetaTrainer`
                  `src.models.ada_style_speech.AdaStyleSpeechModel`
        default_obj_path: Default object path.
    
    Returns:
        Extracted object.
    Raises:
        AttributeError: When the object does not have the given named attribute.
    
    """
    obj_path_list = obj_path.rsplit('.', 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f'Object `{obj_name}` cannot be loaded from `{obj_path}`.')
    return getattr(module_obj, obj_name)


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


def save_checkpoint(filepath, obj, ext='pth', num_ckpt_keep=10):
    ckpts = sorted(pathlib.Path(filepath).parent.glob(f'*.{ext}'))
    if len(ckpts) > num_ckpt_keep:
        [os.remove(c) for c in ckpts[:-num_ckpt_keep]]
    torch.save(obj, filepath)


def scan_checkpoint(cp_dir, prefix='ckpt_'):
    pattern = os.path.join(cp_dir, prefix + '????????.pth')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]
