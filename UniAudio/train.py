# Author: UniAudio Teams

# External dependency
import os
import time
import math
import pickle
import numpy as np
import torch
import argparse
import logging
import json
import functools

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

import torch.distributed as dist

# Local dependency
from model import CrossEntropyAndAccuracy
from model import MegaByteModel, ResidualAttentionBlock
from utils.train_utils import (
    seed_everything, 
    setup_logging,
    yaml_no_alias_safe_dump,
    save_checkpoint,
    maybe_resume_checkpoint,
    WarmupLR,
    str2bool,
    find_data_jsons,
)
from utils.dataloader import get_data_iterator_tokenizer_vocabulary
from utils.reporter import Reporter
from utils.arguments import add_tokenizer_arguments


def get_args():
    # TODO: move all argument parsing into utils/arguments.py and make them grouped
    parser = argparse.ArgumentParser()

    # args for randomness
    parser.add_argument('--seed', type=int, default=None, help='seed for initializing training. ')
    parser.add_argument('--cudnn_deterministic', default=False, action='store_true', help='set cudnn.deterministic True')

    # args for data
    parser.add_argument('--train_data_jsons', type=str, nargs="+", help="list of train data jsons, separated by comma,")
    parser.add_argument('--valid_data_jsons', type=str, nargs="+", help="list of valid data jsons, separated by comma,")
    parser.add_argument('--batch_scale', type=int, default=10000, help="summed sequence length of each batch")
    parser.add_argument('--max_length', type=int, default=2000, help="maximum length of each example sequence. -1 means no constraint. The real allowed length may exceed this slightly")
    parser.add_argument('--min_length', type=int, default=100, help="minimum length of each example sequence. -1 means no constraint. The real allowed length may exceed this slightly")
    parser.add_argument('--n_worker', type=int, default=4, help='number of loading workers for each GPU')
    parser.add_argument('--minibatch_debug', type=int, default=-1, help="if > 0, chuncate the data iterator for debug")
    parser.add_argument('--use_task_id', type=str2bool, default=True, help='if True, add the task-id in the sequence')

    # args for training / optimization
    parser.add_argument('--n_epoch', type=int, default=500, help='Total training epoch')
    parser.add_argument('--grad_accum', type=int, default=1, help='help to simulate large batch')
    parser.add_argument('--learning_rate', type=float, default=6e-4, help='The learning rate for training')
    parser.add_argument('--grad_clip', type=float, default=2.0, help='clip gradients at this value, or disable if == 0.0')
    parser.add_argument('--warmup_steps', type=int, default=10000, help="step of warmup")

    # args for model setting
    parser.add_argument('--n_layer', type=int, default=16, help='the layer of transformer')
    parser.add_argument('--n_head', type=int, default=12, help='the number of multiple head in transformer')
    parser.add_argument('--n_embd', type=int, default=768, help='The embedding dim for transformer')
    parser.add_argument('--dropout', type=float, default=0.0, help='for pretraining 0 is good, for finetuning try 0.1+')
    parser.add_argument('--bias', type=bool, default=False, help='do we use bias inside LayerNorm and Linear layers?')
    parser.add_argument('--block_size', type=int, default=8000, help='exact allowed sequence length in the model')
    parser.add_argument('--prefix-lm', type=str2bool, default=False, help="If true, use prefix LM.")
    parser.add_argument('--encoder-decoder-arch', action='store_true', default=False, help="enc-dec arch if true; otherwise decoder-only")
    parser.add_argument('--non-acoustic-repeat', default=1, type=int, help="repeat non-acoustic tokens for MIMO prediction")
    parser.add_argument('--whisper-model', type=str2bool, default=True, help="If true, use whisper model, else use dongchao's model")
    parser.add_argument('--megabytes', type=str2bool, default=True, help="use Meta's megabytes decoder")

    # args for save model and log: 
    parser.add_argument('--exp_dir', type=str, help='directory of this experiment')
    parser.add_argument('--print_freq', type=int, default=500, help='the print frequency')
    parser.add_argument('--save_interval', type=int, default=10000, help='save a checkpoint within an epoch')
    parser.add_argument('--resume', type=str, default=None, help='whether re-train model')

    add_tokenizer_arguments(parser)
    
    args = parser.parse_args()
    
    return args

def main():
    # (1) use DDP anyway (even for 1 GPU)
    dist.init_process_group(backend="nccl", init_method="env://")
    rank, local_rank, world_size = dist.get_rank(), int(os.environ["LOCAL_RANK"]), dist.get_world_size()
    assert torch.cuda.is_available(), "CUDA is not available"
    torch.cuda.set_device(local_rank)

    # (2) arg parsing and logging
    args = get_args()
    if rank == 0:
        os.makedirs(args.exp_dir, exist_ok=True)
        os.makedirs(args.exp_dir + '/logs', exist_ok=True)
    else:
        time.sleep(3)

    log_file = args.exp_dir + '/logs/RANK.log'
    setup_logging(rank, world_size, log_file)
    reporter = Reporter()

    # (3) randomness & cudnn settings 
    # if args.seed is not None or args.cudnn_deterministic:
    #     seed_everything(args.seed, args.cudnn_deterministic)
    torch.manual_seed(1337 + args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    # (4) data related objects: data iterator, tokenizers, vocabulary
    (train_iter, valid_iter, train_tokenizers, valid_tokenizers, token_list, type_bias) = \
        get_data_iterator_tokenizer_vocabulary(
            args=args,
            train_jsons=find_data_jsons(args.train_data_jsons),
            valid_jsons=find_data_jsons(args.valid_data_jsons),
            batch_scale=args.batch_scale,
            minibatch_debug=args.minibatch_debug,
            max_length=args.max_length,
            min_length=args.min_length,
            non_acoustic_repeat=args.non_acoustic_repeat,
            n_worker=args.n_worker,
            seed=args.seed,
            decoder_only=not args.encoder_decoder_arch,
            use_task_id=args.use_task_id,
    )
    # (5) save config
    setattr(args, 'token_list', token_list)
    if rank == 0:
        with open(args.exp_dir + "/config.yaml", "w", encoding="utf-8") as f:
            logging.warning(
                f'Saving the configuration in {args.exp_dir}/config.yaml'
            )
            yaml_no_alias_safe_dump(vars(args), f, indent=4, sort_keys=False)

    # (6) model, wrapped in FSDP
    if not args.megabytes and args.non_acoustic_repeat > 1:
        raise ValueError("Don't use args.non_acoustic_repeat > 1 without MegaBytes model")

    model = MegaByteModel(n_vocab=len(token_list),
                            n_ctx=args.block_size,
                            n_state=args.n_embd,
                            n_head=args.n_head,
                            n_layer=args.n_layer,
                            prefix_lm=args.prefix_lm,
                            na_repeat=args.non_acoustic_repeat,
                            local_sos=token_list.index("<mega_local>"),
                            global_sos=token_list.index("<mega_global>"),
                            )
    logging.warning(
        "num. model params: {:,} (num. trained: {:,} ({:.1f}%))".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            sum(p.numel() for p in model.parameters() if p.requires_grad)
            * 100.0
            / sum(p.numel() for p in model.parameters()),
        )
    )
    print(
        "num. model params: {:,} (num. trained: {:,} ({:.1f}%))".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            sum(p.numel() for p in model.parameters() if p.requires_grad)
            * 100.0
            / sum(p.numel() for p in model.parameters()),
        )
    )
    
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={ResidualAttentionBlock}
    )
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        device_id=torch.cuda.current_device()
    )

    # (7) objects related to optimization: optimizer and scheduler
    optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=1e-3,
            eps=1e-8,
    )
    scheduler = WarmupLR(optimizer, args.warmup_steps)

    # (8) Resume model, optimizer, scaler, etc, if needed. 
    # TODO: Should give warnings if the current arguments are not compatible to the arguments in the saved file
    # TODO: make sure this can still work if FSDP is adopted, esp. for optimizer
    maybe_resume_checkpoint(args, model, optimizer, scheduler, reporter, train_iter)

    # statistics
    logging.info(f'model arch: {model}')
    # TODO: more model statistics, like param budget? 

    # (9) training and evaluation
    start_epoch = reporter.get_epoch() + 1
    if start_epoch > args.n_epoch:
        logging.error(f'already reach the maximum training epochs. Done!')

    logging.info("training start ... ")
    for ep in range(start_epoch, args.n_epoch + 1):
        reporter.set_epoch(ep)
        # (10.1) train
        with reporter.observe("train") as sub_reporter:
            train_one_epoch(
              args=args,
              model=model,
              train_dl=train_iter,
              optimizer=optimizer,
              scheduler=scheduler,
              reporter=sub_reporter,
              parent_reporter=reporter,
            )
        train_iter.sampler.refresh()

        # (10.2) evaluation
        with torch.no_grad():
            with reporter.observe("valid") as sub_reporter:
                validate_model(
                  args=args,
                  model=model,
                  valid_dl=valid_iter,
                  reporter=sub_reporter,
                )
        # (10.3) epoch logging. 
        logging.info(reporter.log_message())
        # (10.4) save checkpoint
        checkpoint_path = args.exp_dir + f"/ep{ep}.checkpoint"
        logging.info(f"Saving checkpoint file {checkpoint_path}")
        save_checkpoint(checkpoint_path, model, optimizer, scheduler, reporter)

def train_one_epoch(args, model, train_dl, optimizer, scheduler, reporter, parent_reporter):
    model = model.train()
    optimizer.zero_grad()

    for b_idx, batch in enumerate(reporter.measure_iter_time(train_dl, "iter_time"), 1):
        seqs, masks, lengths, conti_segs, example_ids, tasks = batch
        data_stats = {
            "batch_size": len(seqs),
            "seq_len": seqs.size(1) // args.non_acoustic_repeat,
        }
        reporter.register(data_stats)

        with reporter.measure_time("forward_time"):
            if args.megabytes:
                logits = model(seqs, masks, conti_segs=conti_segs)
                _, loss, metrics = CrossEntropyAndAccuracy(
                    logits,
                    seqs,
                    masks,
                    prefix_lm=args.prefix_lm,
                )
            else:
                logits = model(seqs[:, :-1], masks[:, :-1], conti_segs=conti_segs)
                _, loss, metrics = CrossEntropyAndAccuracy(
                    logits,
                    seqs[:, 1:],
                    masks[:, 1:],
                    prefix_lm=args.prefix_lm,
                )

            for v in metrics.values(): # Cross-GPU statistics
                dist.all_reduce(v, dist.ReduceOp.AVG)
            reporter.register(metrics, weight=masks.int().sum().item())

        with reporter.measure_time("backward_time"):
            loss.backward()

        with reporter.measure_time("optim_time"):
            if b_idx % args.grad_accum == 0:
                grad_norm = model.clip_grad_norm_(args.grad_clip)
                if math.isnan(grad_norm):
                    logging.warning(f"grad norm is NaN. Discard this gradient")
                    optimizer.zero_grad()

                optimizer.step() # update the model even with ill gradient - sync the training
                optimizer.zero_grad()
                scheduler.step()

            reporter.register(
              {f'lr_param_{i}': pg['lr'] for i, pg in enumerate(optimizer.param_groups)}
            )

        # must call this here so that the saved checkpoint is valid for reporter
        reporter.next()

        if b_idx % args.print_freq == 0:
            logging.info(reporter.log_message(-args.print_freq))
            print(reporter.log_message(-args.print_freq))

        if args.save_interval > 0 and b_idx % args.save_interval == 0:
            checkpoint_path = args.exp_dir + f"/ep{reporter.get_epoch()}-iter{b_idx}.checkpoint"
            logging.info(f"Saving checkpoint file within an epoch: {checkpoint_path}")
            print(f"Saving checkpoint file within an epoch: {checkpoint_path}")
            save_checkpoint(checkpoint_path, model, optimizer, scheduler, parent_reporter)
        #logging.flush() # update
def validate_model(args, model, valid_dl, reporter):
    model = model.eval()

    for b_idx, batch in enumerate(reporter.measure_iter_time(valid_dl, "iter_time"), 1):
        seqs, masks, lengths, conti_segs, example_ids, tasks = batch
        data_stats = {
            "batch_size": len(seqs),
            "seq_len": seqs.size(1) // args.non_acoustic_repeat,
        }
        reporter.register(data_stats)

        with reporter.measure_time("forward_time"):
            if args.megabytes:
                logits = model(seqs, masks, conti_segs=conti_segs)
                _, loss, metrics = CrossEntropyAndAccuracy(
                    logits,
                    seqs,
                    masks,
                    prefix_lm=args.prefix_lm,
                )
            else:
                logits = model(seqs[:, :-1], masks[:, :-1], conti_segs=conti_segs)
                _, loss, metrics = CrossEntropyAndAccuracy(
                    logits,
                    seqs[:, 1:],
                    masks[:, 1:],
                    prefix_lm=args.prefix_lm,
                )

            # Here we currently do not aggregate all statistics across GPUs.
            # As the numbers of batches may be different for different GPUs
            reporter.register(metrics, weight=masks.int().sum().item())

        reporter.next()

if __name__ == '__main__':
    main()    

