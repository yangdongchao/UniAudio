# UniAudio Teams
import torch
import argparse
import logging
import sys
import yaml
import torchaudio

from pathlib import Path

from utils.train_utils import (
    to_device,
    resume_for_inference,
    str2bool,
    setup_logging,
)
from utils.dataloader import get_data_iterator_tokenizer_vocabulary
from utils.task_definition import task_formats
from utils.arguments import add_tokenizer_arguments

from model import (
    DecoderOnlyModel, 
    MegaByteModel, 
    CrossEntropyAndAccuracy,
    MultiHeadAttention,
)

def get_parser():
    parser = argparse.ArgumentParser()

    # model related: use the resume model if provided; otherwise use the latest in exp_dir
    parser.add_argument('--resume', type=str, default=None, help='model to resume. If None, use the latest checkpoint in exp_dir')
    parser.add_argument('--exp_dir', type=str, default=None, help='experiment directory')

    # inference related: 
    parser.add_argument('--inference_mode', type=str, default='sampling', 
                         choices=['sampling', 'greedy', 'teacher-force'])
    parser.add_argument('--temperature', type=float, default=1.0, help='softmax temperature in sampling')
    parser.add_argument('--topk', type=int, default=30, help='can only select top-k candidate in sampling')
    parser.add_argument('--n_samples', type=int, default=1, help="number of samples during inference")
    parser.add_argument('--maxlen_ratio', type=float, default=-1, help='max length ratio w.r.t. prefix')
    parser.add_argument('--minlen_ratio', type=float, default=-1, help='min length ratio w.r.t. prefix')
    parser.add_argument('--seed', type=int, default=888, help='random seed')
    parser.add_argument('--fixed_length', type=str2bool, default=False,
                        help='if True, output has the known and the same length with input'
                              "If this is true, maxlen_ratio and minlen_ratio is not effective")
    parser.add_argument('--reserve_input', type=str2bool, default=True,
                        help="if true, reserve the input sequence for debug")

    # device related
    parser.add_argument('--rank', type=int, default=-1, help='GPU rank. -1 means CPU')

    # data related
    parser.add_argument('--data_json', type=str, default=None, help="data jsons for inference")
    parser.add_argument('--output_dir', type=str, help="tag for decoding")
    parser.add_argument('--generate_target', type=str, default="audio", help="the format of the generated target")

    add_tokenizer_arguments(parser)
    return parser 

def main():
    # (1) arg parsing & train_arg parsing & logging & seed
    parser = get_parser()
    args = parser.parse_args()

    train_config = args.exp_dir + '/config.yaml'
    with open(train_config, "r", encoding="utf-8") as f:
        train_args = yaml.safe_load(f)
        train_args = argparse.Namespace(**train_args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True) 
    setup_logging(args.rank, 0, args.output_dir + f'/inference.RANK.log')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # (2) build device and file
    if args.rank >= 0:
        args.rank = args.rank % torch.cuda.device_count()
        device = torch.device(f'cuda:{args.rank}') # run.pl index from 1
    else:
        device = torch.device('cpu')
    logging.info(f'inference using {device}')

    # later you may want to revise the model args in the inference stage.
    # remember to set a warning then.
    # (3) data loader: build the loader like the valid iter
    train_args.text_emb_tokenizer = "none"
    train_args.phone_tokenizer = 'alignment'
    _, test_iter, _, tokenizers, _, type_bias = \
        get_data_iterator_tokenizer_vocabulary(
            args = train_args,
            train_jsons=[],
            valid_jsons=[args.data_json],
            batch_scale=train_args.batch_scale,
            non_acoustic_repeat=train_args.non_acoustic_repeat,
            n_worker=1,
            use_task_id=train_args.use_task_id,
    )
    print('tokenizers ', tokenizers)
    # (4) Build trained model
    if train_args.whisper_model and train_args.megabytes:
        model = MegaByteModel(n_vocab=len(train_args.token_list),
                              n_ctx=train_args.block_size,
                              n_state=train_args.n_embd,
                              n_head=train_args.n_head,
                              n_layer=train_args.n_layer,
                              prefix_lm=train_args.prefix_lm,
                              na_repeat=train_args.non_acoustic_repeat,
                              local_sos=train_args.token_list.index("<mega_local>"),
                              global_sos=train_args.token_list.index("<mega_global>"),
       )
    else:
        raise NotImplementedError

    resume_for_inference(args.resume, args.exp_dir, model, device)
    model = model.to(device).eval()
    logging.info(f'model built with arch: {model}')

    # (5) check args and build the inference object
    check_arguments(args)
    logging.info(f"decoding args: {args}")
    inference_implementation = InferenceImp(
        model=model,
        token_list=train_args.token_list,
        mode=args.inference_mode,
        target=args.generate_target,
        n_samples=args.n_samples,
        maxlen_ratio=args.maxlen_ratio,
        minlen_ratio=args.minlen_ratio,
        topk=args.topk,
        temperature=args.temperature,
        fixed_length=args.fixed_length,
    )
    # (5) start inference
    writer = open(output_dir / 'results.scp', 'w')
    input_writer = open(output_dir / 'input.scp', 'w')
    for b_idx, batch in enumerate(test_iter):
        seqs , masks, lengths, conti_segs, example_ids, tasks = to_device(
            batch, device=device, non_blocking=False
        )
        # The model cannot handle the very long sentence that exceed this. 
        if len(seqs[0]) // model.na_repeat > train_args.max_length:
            continue

        for i_idx in range(len(seqs)):
            target_type = task_formats[tasks[i_idx]]['type'][-1].replace('_prompt', '')
            if target_type != args.generate_target:
                raise ValueError(f"invalid example. The generate target should be {args.generate_target}")
            logging.info(f'Inference {tasks[i_idx]} example: {example_ids[i_idx]} to generate {target_type}')
            searched_results = inference_implementation(
                seqs[i_idx],
                masks[i_idx], 
                conti_segs[i_idx: i_idx + 1],
            )

            target_tokenizer = tokenizers[target_type]
            for r_idx, r in enumerate(searched_results):
                searched = chunk_after_end(
                    r['tokens'], 
                    train_args.token_list.index(f"<{target_type}_end>")
                )
                searched -= type_bias[target_type]

                if target_type == "audio":
                    detokenized = target_tokenizer.detokenize(searched)
                    file_name = f"{example_ids[i_idx]}_{args.inference_mode}_sample{r_idx}.wav"
                    file_name = str(output_dir / file_name)
                    logging.info(f"save audio into {file_name}")
                    torchaudio.save(
                        file_name, detokenized, target_tokenizer.sr, 
                        bits_per_sample=16, encoding='PCM_S',
                    )
                    writer.write(
                        f'{example_ids[i_idx]}_{r_idx} {tasks[i_idx]} {target_type} {file_name}\n'
                    )
                elif target_type == 'text':
                    detokenized = target_tokenizer.detokenize(searched[::model.na_repeat])
                    logging.info(f'Hypothesis {r_idx}: {detokenized}')
                    writer.write(
                        f'{example_ids[i_idx]}_{r_idx} {tasks[i_idx]} {target_type} {detokenized}\n'
                    )

                elif target_type == 'sv_bool':
                    logging.info(f"sv result: {searched}")
                else:
                    raise NotImplementedError

            # Keep the input for debug
            if args.reserve_input:
                data_types = task_formats[tasks[i_idx]]['type']
                this_seq = seqs[i_idx].reshape(-1, model.na_repeat)
                for d_idx, data_type in enumerate(data_types):
                    start_tok = train_args.token_list.index(f'<{data_type}_start>') 
                    end_tok = train_args.token_list.index(f'<{data_type}_end>')
                    start_index = (this_seq[:, 0] == start_tok).nonzero(as_tuple=True)[0]
                    end_index = (this_seq[:, 0] == end_tok).nonzero(as_tuple=True)[0]
                    if len(start_index) > 1:
                        start_index = start_index[0]
                        end_index = end_index[0]
                    this_seg = this_seq[start_index + 1: end_index].reshape(-1)
                    this_seg -= type_bias[data_type]
                    this_seq = this_seq[end_index + 1:]

                    # audio input
                    if data_type in ['audio', 'audio_prompt']:
                        tokenizer = tokenizers['audio']
                        detokenized = tokenizer.detokenize(this_seg)
                        file_name = f"{example_ids[i_idx]}_input{d_idx}.wav"
                        file_name = str(output_dir / file_name)
                        logging.info(f"save input audio into {file_name}")
                        torchaudio.save(
                            file_name, detokenized, tokenizer.sr,
                            bits_per_sample=16, encoding='PCM_S',
                        )
                        input_writer.write(
                            f'{example_ids[i_idx]}_input{d_idx} {tasks[i_idx]} {data_type} {file_name}\n'
                        )

                    # string input: haven't validate each detokenize function. validate them later
                    # phone tokenizer is validated
                    elif data_type in ['phone', 'text', 'semantic', 'sv_bool']:
                        tokenizer = tokenizers[data_type]
                        print('tokenizer ', tokenizer)
                        detokenized = tokenizer.detokenize(this_seg[::model.na_repeat])
                        input_writer.write(
                            f'{example_ids[i_idx]}_input{d_idx} {tasks[i_idx]} {data_type} {detokenized}\n'
                        )
                        logging.info(f'save input {d_idx} (type={data_type}): {detokenized}')
                    else:
                        logging.info(f'cannot recover input type: {data_type}')


class InferenceImp(object):
    def __init__(self, 
                 model, 
                 token_list, 
                 mode="sampling",
                 target="audio",
                 n_samples=1,
                 maxlen_ratio=10, 
                 minlen_ratio=1,
                 topk=-1,
                 temperature=1.0,
                 fixed_length=False,
        ):
        self.model = model

        # hyper-params
        self.mode = mode
        self.n_samples = n_samples
        self.maxlen_ratio = maxlen_ratio
        self.minlen_ratio = minlen_ratio
        self.topk = topk
        self.temperature = temperature
        self.fixed_length = fixed_length
        logging.info(f"Inference with mode: {mode}. Target: {target}")

        self.na_repeat = model.na_repeat

        # special token-ids
        self.start = token_list.index("<start>")
        self.end = token_list.index("<end>")

        self.mega_global = token_list.index("<mega_global>")
        self.mega_local = token_list.index("<mega_local>")

        self.target_start = token_list.index(f"<{target}_start>")
        self.target_end = token_list.index(f'<{target}_end>')
     
        # valid prediction intervals
        target_token_index = []
        for i, tok in enumerate(token_list):
            if tok.replace(f"<{target}_", "").replace(">", "").isnumeric():
                target_token_index.append(i)

        self.pred_valid_masks = []
        start, end = target_token_index[0], target_token_index[-1] + 1
        if target == 'audio':
            end -= 1 # exclude the speech-edit mask-id
        for idx in range(self.na_repeat):
            mask = torch.ones(len(token_list))
            if target == 'audio':
                mask[start + (end - start) * idx // self.na_repeat
                    :start + (end - start) * (idx + 1) // self.na_repeat] = 0
            else:
                mask[start: end] = 0
            mask[self.target_end: self.target_end + 1] = 0
            self.pred_valid_masks.append(mask.bool())

        self.token_list = token_list
        self.target = target

    @torch.no_grad()
    def __call__(self, seq, mask, conti_segs):
        assert seq.size() == mask.size()
        assert seq.dim() == 1
        device = seq.device
        seq  = seq.unsqueeze(0).expand(self.n_samples, -1)
        mask = mask.unsqueeze(0).expand(self.n_samples, -1)
        conti_segs = conti_segs*self.n_samples
        # (0) full forward like training; for debug only
        if self.mode == "teacher-force":
            full_logits = self.model(seq, mask, None, conti_segs)
            logits, loss, metrics = CrossEntropyAndAccuracy(
                full_logits, seq, mask
            )
        # remove padding
        pad_len = seq[0].eq(0).int().sum().item()
        mask = mask[:, :len(seq[0]) - pad_len]
        seq = seq[:, :len(seq[0]) - pad_len]
        # (1) prefix inference
        prefix_len = torch.logical_and(~mask[0], seq[0].ne(0)).int().sum().item()
        prefix = seq[:, :prefix_len + self.na_repeat]
        mask = mask[:, :prefix_len + self.na_repeat]
        g_cache = {}
        g_cache, g_hooks = install_kv_cache_hook(self.model.g_layers, g_cache)
        prefix_logits = self.model(prefix, mask, g_cache, conti_segs)

        # (2) search loop
        if not self.fixed_length:
            maxlen = int(prefix_len // self.na_repeat * self.maxlen_ratio)
            minlen = int(prefix_len // self.na_repeat * self.minlen_ratio)
        else:
            maxlen = (len(seq[0]) - prefix_len) // self.na_repeat - 2
            minlen = (len(seq[0]) - prefix_len) // self.na_repeat - 2
        logging.info(f'search with max length: {maxlen} and min length: {minlen}')
        if maxlen + prefix_len > self.model.n_ctx - 1:
            maxlen = self.model.n_ctx - prefix_len // self.na_repeat - 1
            logging.info(f'maxlen is too long: change it as: {maxlen}')
        #print('prefix_len ', prefix_len)
        g_prev_tok = torch.ones_like(prefix[:, :self.na_repeat]) * self.target_start
        #print('g_prev_tok ', g_prev_tok.shape)
        searched_results = [{
            'logits': [],
            'tokens': [],
            'scores': [],
            'sum_score': 0.0,
            } for _ in range(self.n_samples)]
        final_results = []
        for g_idx in range(maxlen):
            # (2.1) global inference. AR prediction requires no mask
            #print('begin g_idx ', g_idx)
            global_logits = self.model.global_forward(
                g_prev_tok, None, g_cache, conti_segs=None
            )
            #print('global_logits ', global_logits.shape)
            # (2.2) local inference
            l_cache = {}
            l_cache, l_hooks = install_kv_cache_hook(self.model.l_layers, l_cache)
            l_prev_tok = torch.ones_like(g_prev_tok[:, :1]) * self.mega_local
            for l_idx in range(self.na_repeat):
                local_logits = self.model.local_forward(
                    l_prev_tok, 
                    global_logits[:, l_idx:l_idx + 1], 
                    l_cache, 
                    conti_segs,
                ).squeeze(1)

                # (2.3) hypothesis update
                logp = local_logits.log_softmax(-1)
                topk_values_orig, topk_indices_orig = torch.topk(logp, self.topk, dim=-1)

                logp.masked_fill_(
                    self.pred_valid_masks[l_idx].to(device).unsqueeze(0), float('-inf')
                )
                if g_idx < minlen - 1:
                    logp[:, self.target_end] = float('-inf')

                if torch.any(logp.exp().sum(dim=-1) < 0.9):
                    logging.info(f'warning: invalid logp summation {g_idx} {l_idx} {logp.exp().sum(dim=-1)}')
                    logging.info(f'original topk: {topk_values_orig} {topk_indices_orig}')
                topk_values, topk_indices = torch.topk(logp, self.topk, dim=-1)

                if self.mode == "teacher-force":
                    l_prev_tok = torch.ones_like(l_prev_tok) * \
                        seq[:, prefix_len + (g_idx + 1) * self.na_repeat + l_idx].unsqueeze(1)
                elif self.mode == "greedy":
                    l_prev_tok = topk_indices[:, :1]
                elif self.mode == "sampling":
                    inner_index = torch.multinomial(
                        (topk_values / self.temperature).softmax(-1), 
                        num_samples=1
                    ).squeeze(1)
                    b_idx = torch.arange(len(searched_results)).to(device).long()
                    l_prev_tok = topk_indices[b_idx, inner_index].unsqueeze(1)
                elif self.mode == "beam_search":
                    raise NotImplementedError
                else:
                    raise NotImplementedError

                for i in range(len(searched_results)):
                    searched_results[i]['logits'].append(local_logits[i])
                    searched_results[i]['tokens'].append(l_prev_tok[i, 0])
                    searched_results[i]['scores'].append(logp[i, l_prev_tok[i]])
                    searched_results[i]['sum_score'] += logp[i, l_prev_tok[i]]

            # (2.4) check & select hypotheses
            keep_indices = []
            for i in range(len(searched_results)):
                if searched_results[i]['tokens'][-1].item() == self.target_end:
                    final_results.append(searched_results[i])
                else:
                    keep_indices.append(i)
            if len(keep_indices) < len(searched_results):
                searched_results = [searched_results[j] for j in keep_indices]
                keep_indices = torch.Tensor(keep_indices).to(device).long()
                select_cache(g_cache, keep_indices)

            # (2.5) local finalize
            for hook in l_hooks:
                hook.remove()

            if len(searched_results) == 0:
                break

            g_prev_tok = torch.stack(
                [torch.stack(r['tokens'][-self.na_repeat:], dim=0) 
                 for r in searched_results], dim=0
            )
            # print(f"{g_idx} {g_prev_tok}")

        # (3) Global finalize
        for hook in g_hooks:
            hook.remove()

        if len(searched_results) > 0:
            logging.info(f'still add {len(searched_results)} incomplete examples')
            final_results = final_results + searched_results

        for i in range(len(final_results)):
            if final_results[i]['tokens'][-1] != self.target_end:
                force_end = [torch.Tensor([self.target_end]).to(device).squeeze(0)
                    for _ in range(self.na_repeat)]
                final_results[i]['tokens'] = final_results[i]['tokens'] + force_end

        for i in range(len(final_results)):
            final_results[i]['logits'] = torch.stack(final_results[i]['logits'], dim=0)
            final_results[i]['tokens'] = torch.stack(final_results[i]['tokens'], dim=0)
            final_results[i]['scores'] = torch.stack(final_results[i]['scores'], dim=0)

            # keep the ground-truth. should be top-1
            if self.mode == 'teacher-force':
                final_results[i]['tokens'] = final_results[i]['logits'].argmax(-1)

        return final_results

# Some utilities during inference
def install_kv_cache_hook(model, cache):
    cache = {**cache} if cache is not None else {}
    hooks = []

    def save_to_cache(module, _, output):
        if module not in cache:
            # save as-is, for the first token or cross attention
            cache[module] = output
        else:
            cache[module] = torch.cat([cache[module], output], dim=1).detach()
        return cache[module]

    def install_hooks(layer: torch.nn.Module):
        if isinstance(layer, MultiHeadAttention):
            hooks.append(layer.key.register_forward_hook(save_to_cache))
            hooks.append(layer.value.register_forward_hook(save_to_cache))

    model.apply(install_hooks)
    return cache, hooks

def chunk_after_end(searched, end):
    end_idx = torch.where(
        searched == end, 
        torch.arange(len(searched), device=searched.device),
        float('inf'),
        ).min().int().item()
    return searched[:end_idx]

def select_cache(cache, indices):
    for k, v in cache.items():
        cache[k] = v[indices]

def check_arguments(args):
    # Consider the exclusive situations for each inference argument
    if args.fixed_length:
        assert args.minlen_ratio < 0
        assert args.maxlen_ratio < 0

    if args.inference_mode == "teacher-force":
        assert args.n_samples == 1

    if args.maxlen_ratio > 0 or args.minlen_ratio > 0:
        assert not args.fixed_length


if __name__ == "__main__":
    main()
