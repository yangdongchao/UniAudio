# Auhtor: UniAudio Teams

import os
import sys
import torch
import copy
import random
import logging
import pathlib
import itertools
import torch.distributed as dist

from utils.task_definition import (
    load_data_for_all_tasks,
    task_formats
)

from tools.tokenizer.abs_tokenizer import AbsTokenizer
from tools.tokenizer.soundstream.AudioTokenizer import AudioTokenizer
from tools.tokenizer.soundstream.EncodecTokenizer import EncodecTokenizer
from tools.tokenizer.Text2Phone.Text2PhoneTokenizer import Text2PhoneTokenizer
from tools.tokenizer.Prompt.audio_prompt_tokenizer import AudioPromptTokenizer
from tools.tokenizer.BPE.SPTokenizer import BPETokenizer
from tools.tokenizer.Semantic.Semantic_tokenizer import SemanticTokenizer
from tools.tokenizer.phone.phone_tokenizer import PhoneTokenizer
from tools.tokenizer.T5.T5Tokenizer import FrozenT5Embedder
from tools.tokenizer.classifier.SV_bool_tokenizer import SVBoolTokenizer
from tools.tokenizer.Sing.sing_midi_tokenizer import SingMidiTokenizer
from tools.tokenizer.Sing.sing_phone_tokenizer import SingPhoneTokenizer

def build_data_iterator(
        data_dict,
        tokenizers,
        token_list,
        type_bias,
        max_length=-1,
        min_length=-1,
        non_acoustic_repeat=1,
        batch_scale=1000,
        is_train=True,
        n_worker=1,
        decoder_only=True,
        seed=999,
        minibatch_debug=-1,
        use_task_id=True,
    ):
    find_all_length(data_dict, tokenizers)
    valid_utts = filter_data(data_dict, max_length, min_length) 
    batches = batchfy(data_dict, valid_utts, batch_scale)
    logging.info(f"Finish pre-process all data. {len(data_dict)} examples and {len(batches)} batches")
    if minibatch_debug > 0:
        batches = batches[:min(minibatch_debug, len(batches))]
        logging.info(f"only use {len(batches)} as this is a debug mode")
    dataset = Dataset(batches, data_dict)
    sampler = DDPSyncSampler(size=len(batches), seed=seed)
    # Build iterator. No multi-process when debug
    collate_fn = Collate_Fn_Factory(
            tokenizers, token_list, type_bias, 
            decoder_only=decoder_only, non_acoustic_repeat=non_acoustic_repeat,
            use_task_id=use_task_id,
            is_train=is_train,
    )
    if minibatch_debug != -1:
        iterator = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=0,
            collate_fn=collate_fn,
        )
        logging.info("disable multi-processing data loading: debug mode")
    else:
        iterator = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=n_worker,
            prefetch_factor=min(100, len(batches)),
            collate_fn=collate_fn,
        )
    return iterator

def filter_data(data_dict, max_length, min_length):
    # we find the valid key rather than remove the whole exmaple as the invalid exmaples can 
    # also work as the prompt
    keys = list(data_dict.keys())
    if max_length <= 0 and min_length <= 0:
        return keys

    valid_keys = []
    if max_length > 0:
        for k in keys:
            if  (data_dict[k]['length'] <= max_length or max_length <= 0) \
            and (data_dict[k]['length'] >= min_length or min_length <= 0):
                valid_keys.append(k)
    logging.info(f"you requires length between [{min_length}, {max_length}] so only {len(valid_keys)} examples are reserved.")
    return valid_keys

def find_all_length(data_dict, tokenizers):
    """ length found here is only for batchfy. it is not the real length as there may be more special tokens """
    for example_id, d in data_dict.items():
        data_format = task_formats[d['task']]
        length = 0
        for key, key_type in zip(data_format['keys'], data_format['type']):
            this_length = tokenizers[key_type].find_length(d[key])
            length += this_length
        d['length'] = length

def batchfy(data_dict, batch_utts, batch_scale):
    batch_utts.sort(key=lambda x: data_dict[x]['length'])
    batch_lengths = [data_dict[k]['length'] for k in batch_utts]
    # TODO: maybe length**2 is a better measure of computing complexity
    # Only take care of the uttid rather than the whole example
    batches, batch, summed_tokens = [], [], 0
    for utt, l in zip(batch_utts, batch_lengths):
        if l + summed_tokens > batch_scale:
            assert len(batch) > 0, f"batch_tokens should be larger: {batch_scale}"
            batches.append(copy.deepcopy(batch))
            batch, summed_tokens = [], 0

        summed_tokens += l
        batch.append(utt)

    if len(batch) > 0:
        batches.append(copy.deepcopy(batch))

    # TODO: maybe report statistics
    logging.info(f'After batchfy, there are {len(batches)} batches')
    return batches 

class Dataset(torch.utils.data.Dataset):
    """ Dataset. Each example is exactly a batch """
    def __init__(self, data_split, data_dict):
        self.data_split = data_split
        self.data_dict = data_dict

    def __getitem__(self, index):
        uttids = self.data_split[index]
        return [(uttid, self.data_dict[uttid]) for uttid in uttids]

    def __len__(self):
        return len(self.data_split)

class SequentialSampler(object):
    def __init__(self, sequence):
        self.seq = sequence

    def __iter__(self):
        return iter(self.seq)

    def __len__(self):
        return len(self.seq)

    def refresh(self):
        pass

class DDPSyncSampler(object):
    def __init__(self, size, seed):
        self.size = size
        self.seed = seed
        self.epoch = 0

        # Ensure that data iterator aross all GPUs has the same number of batches
        if dist.is_initialized() and torch.cuda.is_available():
            local_rank = int(os.environ["LOCAL_RANK"])
            device = torch.device(f"cuda:{local_rank}")
            size = torch.Tensor([size]).to(device).int()
            dist.all_reduce(size, dist.ReduceOp.MAX)

            self.pad_number = size.item() - self.size
            self.rank = dist.get_rank()
        else:
            logging.warning("torch.distributed is not available!")
            self.pad_number = 0
            self.rank = 0

        self.refresh()

    def refresh(self):
        seq = list(range(self.size))

        # Assume the batches are sorted from shortest to longest
        # This introduces local randomness by local random shuffling
        # otherwise each global batch will be identical across epochs
        chunk_size, start = 10, 0
        random.seed(self.rank + self.seed + self.epoch)
        while start < self.size:
            seg = seq[start: min(self.size, start + chunk_size)]
            local_random_order = random.sample(list(range(len(seg))), len(seg))
            seg = [seg[i] for i in local_random_order]
            seq[start: min(self.size, start + chunk_size)] = seg
            start += len(seg)

        # even after this shuffle, the batch lengths across GPUs 
        # are very similar
        random.seed(self.seed + self.epoch)
        random.shuffle(seq)

        # so the #batches are identical across GPUs
        if self.pad_number > 0:
            seq = list(range(self.pad_number)) + seq

        self.seq = seq
        self.epoch += 1

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        return iter(self.seq)

    def __len__(self):
        return len(self.seq)

    def get_state_dict(self):
        state_dict = {
            'epoch': self.epoch,
            'seed': self.seed,
        }
        return state_dict

    def load_state_dict(self, d):
        for k, v in d.items():
            setattr(self, k, v)

class Collate_Fn_Factory(object):
    def __init__(self, 
                 tokenizers, 
                 token_list, 
                 type_bias,
                 non_acoustic_repeat=1, 
                 decoder_only=True, 
                 max_length=15000,
                 use_task_id=True,
                 is_train=True,
    ):
        self.tokenizers = tokenizers
        self.token_list = token_list
        self.type_bias = type_bias
        self.non_acoustic_repeat = non_acoustic_repeat
        self.decoder_only = decoder_only
        self.max_length = max_length
        self.is_train = is_train

        # some old experiments contains only one task so they don't use task-id
        # turn this to false to be compatible
        # temp code
        self.use_task_id = use_task_id

        assert non_acoustic_repeat >= 1 and isinstance(non_acoustic_repeat, int)

    def add_special_tokens(self, seq, data_type):

        if data_type == 'global':
            start_tok = self.token_list.index(f'<start>')
            end_tok = self.token_list.index(f'<end>')

        elif data_type == 'mask':
            start_tok, end_tok = 0, 1
            seq = seq.long()

        else:
            start_tok = self.token_list.index(f'<{data_type}_start>')
            end_tok = self.token_list.index(f'<{data_type}_end>')

        start_tok = torch.Tensor([start_tok] * self.non_acoustic_repeat).long()
        end_tok = torch.Tensor([end_tok] * self.non_acoustic_repeat).long()

        ans = torch.cat([start_tok, seq, end_tok], dim=0)
        if data_type == 'mask':
            ans = ans.bool()
        return ans

    def splice_sequence(self, d, keys, types, loss_key):
        sequence, mask, continuous_segment, start = [], [], [], 0

        if self.use_task_id:
            task_id = self.token_list.index(f"<{d['task']}_task>")
            sequence.append(torch.Tensor([task_id for _ in range(self.non_acoustic_repeat)]).int())
            mask.append(torch.Tensor([0 for _ in range(self.non_acoustic_repeat)]).bool())
            start += self.non_acoustic_repeat
        cache, task = {'is_train': self.is_train}, d['task']
        for key, tp in zip(keys, types):
            this_data = self.tokenizers[tp].tokenize(d[key], task, cache)

            if tp not in ['audio', 'audio_prompt'] and self.non_acoustic_repeat > 1:
                this_data = torch.flatten(
                    torch.stack([this_data] * self.non_acoustic_repeat, dim=1),
                    end_dim=1)

            if self.tokenizers[tp].is_discrete:
                this_data = this_data + self.type_bias[tp]
                this_data = self.add_special_tokens(this_data, tp)
                sequence.append(this_data)
            
            else:
                # Some tokenization results are continuous.
                # Use this dummpy token at this moment. Will replace the exact results
                # a.k.a., 'this_data', into the embeddings after the embedding layer.
                # TODO: verify this. We haven't face a task that adopts continuous embeddings
                conti_id = self.token_list.index('<continuous_token>')
                pad_seq = torch.Tensor([conti_id] * len(this_data))
                pad_seq = self.add_special_tokens(pad_seq, tp)
                sequence.append(pad_seq)
               
                # this index excludes the special tokens <xx_start> and <xx_end>.
                conti_start = start + self.non_acoustic_repeat
                conti_end   = start + self.non_acoustic_repeat + len(this_data)
                continuous_segment.append((conti_start, conti_end, tp, this_data))

            _this_data = this_data if self.tokenizers[tp].is_discrete else pad_seq
            this_mask = this_mask = torch.ones(len(_this_data)) * int(key == loss_key)
            mask.append(this_mask.bool())

            start += len(_this_data)

        sequence = torch.cat(sequence, dim=0).to(torch.int64)
        mask = torch.cat(mask, dim=0)
        return sequence, mask, continuous_segment, start

    def decoder_only_collate_fn(self, batch):

        batch_size = len(batch)
        sequences = torch.ones((batch_size, self.max_length)).long() * self.token_list.index('<pad>')
        masks = torch.zeros((batch_size, self.max_length)).bool() # default False
        lengths, example_ids, tasks, continuous_segments = [], [], [], []

        for idx, (example_id, d) in enumerate(batch):
            task_format = task_formats[d['task']]
            sequence, mask, conti_seg, length = self.splice_sequence(
                d, task_format['keys'], task_format['type'], task_format['loss_key']
            )
            sequence = self.add_special_tokens(sequence, data_type='global')
            mask = self.add_special_tokens(mask, data_type='mask')
            length += 2 * self.non_acoustic_repeat
            conti_seg = [(seg[0] + self.non_acoustic_repeat, 
                          seg[1] + self.non_acoustic_repeat, 
                          seg[2], seg[3])
                         for seg in conti_seg] # due to global <sos>

            sequences[idx, :length] = sequence
            masks[idx, :length] = mask
            continuous_segments.append(conti_seg)
            lengths.append(length)
            example_ids.append(example_id)
            tasks.append(d['task'])

        sequences = sequences[:, :max(lengths)].long()
        masks = masks[:, :max(lengths)]
        lengths = torch.Tensor(lengths).long()

        return sequences, masks, lengths, continuous_segments, example_ids, tasks 

    def __call__(self, batch):
        assert len(batch) == 1, "batch size should only be 1"
        batch = batch[0]

        if self.decoder_only:
            return self.decoder_only_collate_fn(batch)
        else:
            return self.encoder_decoder_collate_fn(batch)

def show_data_examples(sequences, masks, lengths, continuous_segments, example_ids, tasks, token_list, n=100):
    for j, (seq, mask, length, example_id, conti_segs) in \
    enumerate(zip(sequences, masks, lengths, example_ids, continuous_segments)):
        if j >= n:
            break
        seq = seq.cpu().tolist()
        seq = [token_list[x] for x in seq]
        mask = mask.cpu().tolist()
        logging.info(f" example: {j} - {example_id}, with length: {length}")
        for seg in conti_segs:
            logging.info(f"conti_seg: start: {seg[0]} , end: {seg[1]}, type: {seg[2]}, size: {seg[3].size()}") 
        for i, (tok, m) in enumerate(zip(seq, mask)):
            logging.info(f"token {i}: {tok} | mask: {m}")

def get_data_iterator_tokenizer_vocabulary(
        args,
        train_jsons,
        valid_jsons,
        batch_scale=3000,
        minibatch_debug=-1,
        max_length=-1,
        min_length=-1,
        non_acoustic_repeat=1,
        n_worker=4,
        decoder_only=True,
        seed=999,
        use_task_id=True,
    ):

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format=f"%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )

    # (1) load all data in the raw format
    logging.info(f"loading train: {train_jsons}")
    train_data_dict = load_data_for_all_tasks(train_jsons)
    logging.info(f"loading valid:  {valid_jsons}")
    valid_data_dict = load_data_for_all_tasks(valid_jsons)

    # (2) build all tokenizers: make it configurable; some tokenizers are not shared between train and valid
    train_tokenizers, valid_tokenizers = {}, {}
    if args.audio_tokenizer is not None and args.audio_tokenizer != "none":
        if args.audio_tokenizer == "soundstream":
            audio_tokenizer = AudioTokenizer()
        elif args.audio_tokenizer == "encodec":
            audio_tokenizer = EncodecTokenizer(select_every=args.audio_tokenizer_select_every)
        else:
            raise NotImplementedError(args.audio_tokenizer)
        train_tokenizers['audio'] = audio_tokenizer
        valid_tokenizers['audio'] = audio_tokenizer
    else:
        audio_tokenizer = None
        logging.info(f"Did not build audio tokenizer: {args.audio_tokenizer}")
    if args.audio_prompt_tokenizer is not None and args.audio_prompt_tokenizer != "none":
        assert audio_tokenizer is not None, "Audio Prompt tokeinzer cannot be built without a audio tokenizer"
        audio_prompt_tokenizer_train = AudioPromptTokenizer(train_data_dict, prompt_length=args.audio_prompt_length * audio_tokenizer.freq, n_codebook=audio_tokenizer.n_codebook)
        audio_prompt_tokenizer_valid = AudioPromptTokenizer(valid_data_dict, prompt_length=args.audio_prompt_length * audio_tokenizer.freq, n_codebook=audio_tokenizer.n_codebook)
        train_tokenizers['audio_prompt'] = audio_prompt_tokenizer_train
        valid_tokenizers['audio_prompt'] = audio_prompt_tokenizer_valid
    else:
        logging.info(f"Did not build audio prompt tokenizer: {args.audio_prompt_tokenizer}")

    if args.phone_tokenizer is not None and args.phone_tokenizer != "none":
        if args.phone_tokenizer == "g2p": 
            phone_tokenizer = Text2PhoneTokenizer(duplicate=args.phone_tokenizer_duplicate)
        elif args.phone_tokenizer == "alignment":
            phone_tokenizer = PhoneTokenizer() 
        train_tokenizers['phone'] = phone_tokenizer
        valid_tokenizers['phone'] = phone_tokenizer
    else:
        logging.info(f"Did not build phone tokenizer: {args.phone_tokenizer}")

    if args.text_tokenizer is not None and args.text_tokenizer != "none":
        text_tokenizer = BPETokenizer()
        train_tokenizers['text'] = text_tokenizer
        valid_tokenizers['text'] = text_tokenizer
    else:
        logging.info(f"Did not build text tokenizer: {args.text_tokenizer}")

    if args.semantic_tokenizer is not None and args.semantic_tokenizer != "none":
        semantic_tokenizer = SemanticTokenizer(duplicate=args.semantic_tokenizer_duplicate)
        train_tokenizers['semantic'] = semantic_tokenizer
        valid_tokenizers['semantic'] = semantic_tokenizer
    else:
        logging.info(f"Did not build semantic tokenizer: {args.semantic_tokenizer}")
    
    if args.FrozenT5Embedder is not None and args.FrozenT5Embedder != "none":
        T5_tokenizer =  FrozenT5Embedder()
        train_tokenizers['text_t5'] = T5_tokenizer
        valid_tokenizers['text_t5'] = T5_tokenizer
    else:
        logging.info(f"Did not build T5_tokenizer tokenizer: {args.FrozenT5Embedder}")
    
    if args.sv_bool_tokenizer is not None and args.sv_bool_tokenizer != "none":
        sv_bool_tokenizer = SVBoolTokenizer()
        train_tokenizers['sv_bool'] = sv_bool_tokenizer
        valid_tokenizers['sv_bool'] = sv_bool_tokenizer
    else:
        logging.info(f"Did not build sv_bool tokenizer: {args.sv_bool_tokenizer}")
    
    if args.singPhoneTokenizer is not None and args.singPhoneTokenizer != "none":
        singPhoneTokenizer =  SingPhoneTokenizer()
        train_tokenizers['sing_phone'] = singPhoneTokenizer
        valid_tokenizers['sing_phone'] = singPhoneTokenizer
    else:
        logging.info(f"Did not build sing_phone tokenizer: {args.singPhoneTokenizer}")
    
    if args.singMidiTokenizer is not None and args.singMidiTokenizer != "none":
        singMidiTokenizer =  SingMidiTokenizer()
        train_tokenizers['sing_midi'] = singMidiTokenizer
        valid_tokenizers['sing_midi'] = singMidiTokenizer
    else:
        logging.info(f"Did not build sing_midi tokenizer: {args.singMidiTokenizer}")
    # (3) build vocabulary and the bias for each data key type
    # The first 128 tokens are reserved for special tokens
    # Note: you never delete or modify the current entries
    # due to backward compatibility: you add new special tokens
    # even though the whole token_list is not beautiful
    token_list = [
      "<pad>",
      "<continuous_token>",
      "<start>", "<end>",
      "<phone_start>", "<phone_end>",
      "<text_start>",  "<text_end>",
      "<audio_start>", "<audio_end>",
      "<image_start>", "<image_end>",
      "<audio_prompt_start>", "<audio_prompt_end>",
      "<text_prompt_start>", "<text_prompt_end>",
      "<semantic_start>", "<semantic_end>",
      "<tts_task>",
      "<plain_tts_task>",
      "<lm_task>",
      "<phone_to_semantic_task>",
      "<semantic_to_acoustic_task>",
      "<mega_global>", "<mega_local>",
      "<text_emb_start>", "<text_emb_end>",
      '<asr_task>',
      '<VC_task>',
      '<AT_task>', '<class_event_start>', '<class_event_end>',
      '<Spex_task>', '<TTA_task>', '<rvq_start>', '<rvq_end>',
      '<TSS_task>', '<text_t5_start>', '<text_t5_end>',
      '<SV_task>', '<sv_bool_start>', '<sv_bool_end>',
      '<SE_task>',
      '<sing_task>', '<sing_phone_start>', '<sing_phone_end>', 
      '<sing_midi_start>', '<sing_midi_end>',
      '<TTM_task>', '<Audit_task>', '<InstructTTS_task>', '<Speech_RIR_task>', '<speech_edit_task>',
    ]
    logging.info(f"Special tokens: {token_list}")
    token_list = token_list + ["<unused_special_token>" for _ in range(128 - len(token_list))]
    start = 128
    type_bias = {}
    for name, tokenizer in train_tokenizers.items():
        assert isinstance(tokenizer, AbsTokenizer)
        # discrete tokenizier, the original ones. e.g., BPE, Phone, Audio Codec
        if tokenizer.is_discrete and tokenizer.codebook_length > 0:
            sub_token_list = [f'<{name}_{i}>' for i in range(tokenizer.codebook_length)]
            token_list = token_list + sub_token_list
            logging.info(f"Token list: [{start}: {start+tokenizer.codebook_length}) is reserved for {name}")
            type_bias[name] = start
            start += tokenizer.codebook_length
        # prompt tokenizers that share the bias with its original tokenizer. e.g., Audio Codec Prompt
        elif tokenizer.codebook_length == 0:
            type_bias[name] = type_bias[name.replace("_prompt", "")]
        # continuous tokenizers that has no bias
        else:
            type_bias[name] = 0
    logging.info(f"type bias: {type_bias}")
    # (4) build data iterator
    valid_iterator = build_data_iterator(
        valid_data_dict,
        valid_tokenizers,
        token_list,
        type_bias,
        max_length=max_length,
        min_length=min_length,
        non_acoustic_repeat=non_acoustic_repeat,
        batch_scale=batch_scale,
        is_train=False,
        n_worker=1,
        minibatch_debug=minibatch_debug,
        decoder_only=decoder_only,
        use_task_id=use_task_id,
    )

    train_iterator = build_data_iterator(
        train_data_dict, 
        train_tokenizers, 
        token_list, 
        type_bias,
        max_length=max_length,
        min_length=min_length,
        non_acoustic_repeat=non_acoustic_repeat,
        batch_scale=batch_scale, 
        is_train=True,
        n_worker=n_worker,
        seed=seed,
        minibatch_debug=minibatch_debug,
        decoder_only=decoder_only,
        use_task_id=use_task_id,
    )
    logging.info('all iterator built')

    return train_iterator, valid_iterator, train_tokenizers, valid_tokenizers, token_list, type_bias

if __name__ == "__main__":
    get_data_iterator_tokenizer_vocabulary(sys.argv[1:2], sys.argv[2:3], n_worker=1) 
