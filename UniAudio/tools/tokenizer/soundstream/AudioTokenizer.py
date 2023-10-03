#

"""Command-line for audio compression."""
import argparse
from pathlib import Path
import sys
import torchaudio
import os
import torch
import typing as tp
import torch.distributed as dist
from collections import OrderedDict
from omegaconf import OmegaConf
import logging

from tools.tokenizer.soundstream.models.soundstream import SoundStream
from tools.tokenizer.abs_tokenizer import AbsTokenizer
from tools.tokenizer.common import clip_by_length
from tools.tokenizer.common import codec_specaug

class AudioTokenizer(AbsTokenizer):
    def __init__(self, 
                 device=torch.device('cpu'), 
                 universal=True,
                 clip_length=450):
        """ soundstream with fixed bandwidth of 4kbps 
            It encodes audio with 50 fps and 8-dim vector for each frame
            The value of each entry is in [0, 1023]
        """
        super(AudioTokenizer, self).__init__()
        # GPU is only for offline tokenization
        # So, when distributed training is launched, this should still be on CPU

        self.device = device
        self.clip_length = clip_length

        # Jinchuan: TODO: make this confiurable
        tag = "universal" if universal else "tts"
        config_path = f'UniAudio/checkpoints/{tag}_model/config.yaml'
        if not os.path.isfile(config_path):
            raise ValueError(f"{config_path} file does not exist.")
        config = OmegaConf.load(config_path)
        self.ckpt_path = f"UniAudio/checkpoints/{tag}_model/model.pth'
        logging.info(f"using config {config_path} and model {self.ckpt_path}")
        self.soundstream = self.build_codec_model(config)
        # properties
        self.sr = 16 * 1000
        self.dim_codebook = 1024
        self.n_codebook = 3
        self.bw = 1.5 # bw=1.5 ---> 3 codebooks
        self.freq = self.n_codebook * 50
        self.mask_id = self.dim_codebook * self.n_codebook

    def build_codec_model(self, config):
        model = eval(config.generator.name)(**config.generator.config)
        parameter_dict = torch.load(self.ckpt_path, map_location='cpu')
        model.load_state_dict(parameter_dict['codec_model']) # load model
        model = model.to(self.device)
        return model

    def _flatten_codebooks(self, arr, offset_size=1024):
        assert len(arr.shape) == 2
        arr = arr.copy()
        if offset_size is not None:
            for n in range(arr.shape[0]):
                arr[n, :] += (offset_size * n)
        flat_arr = arr.ravel("F")
        return flat_arr
    
    def encode(self, wav_root, sr=16000):
        wav, sr = torchaudio.load(wav_root)

        if wav.numel() == 0:
            return None

        if sr != self.sr:
            wav = torchaudio.transforms.Resample(sr, self.sr)(wav)

        wav = wav.unsqueeze(1).to(self.device) # (1,1,len)
        compressed = self.soundstream.encode(wav, target_bw=self.bw) # [n_codebook, 1, n_frames]
        compressed = compressed.squeeze(1).detach().cpu().numpy() # [n_codebook, n_frames]
        flat_codec = self._flatten_codebooks(compressed, self.dim_codebook)
        flat_codec = torch.from_numpy(flat_codec)
        flat_codec = flat_codec.to(torch.int16)
        return flat_codec

    def _detokenize(self, compressed): # assuming only one wave need to be decode
        if compressed.dim() == 2:
            compressed = compressed.unsqueeze(1) # from [n_codebook, n_frames] to [n_codebook, 1, n_frames]
        assert compressed.size(0) == self.n_codebook
        assert compressed.dim() == 3
        out = self.soundstream.decode(compressed.long().to(self.device))
        out = out.detach().cpu().squeeze(0) # [1, len]
        return out

    def detokenize(self, codes):
        assert codes.dim() == 1
        assert len(codes) % self.n_codebook == 0
        
        codes = codes.view(-1, self.n_codebook).transpose(0, 1)
        for i in range(self.n_codebook):
            codes[i] -= i * self.dim_codebook
        out = self.soundstream.decode(codes.long().to(self.device).unsqueeze(1))
        out = out.detach().cpu().squeeze(0)
        return out

    @property
    def is_discrete(self):
        return True

    def tokenize(self, wav, task=None, cache=None):

        if isinstance(wav, str):
            # if x is the wave path
            return self.encode(wav)

        elif isinstance(wav, torch.Tensor):
            if wav.dim() == 1: # already done offline
                # Some on-the-fly process
                if task in ["SV"]:
                    wav = clip_by_length(wav, self.clip_length, self.n_codebook)
                # Some task-specific on-the-fly pre-process
                if task in ['speech_edit']:
                    the target to predict
                    if 'speech_edit_target' in cache:
                        wav = cache['speech_edit_target']
                    # the corrupted input 
                    else:
                        cache['speech_edit_target'] = wav.clone().detach()
                        start, end = cache['speech_edit_time_stamp']
                        start = int(start * 1.5) * self.n_codebook
                        end = int(end * 1.5) * self.n_codebook
                        wav[start: end] = self.mask_id        
                    return wav
                if task in ['asr'] and cache['is_train']:
                    wav = codec_specaug(
                        wav.view(-1, self.n_codebook).contiguous(),
                        mask_id=self.mask_id,
                    )
                return wav
            if wav.dim() == 2: # transfer to 3 dim
                if wav.numel() == 0:
                    return None
                wav = wav.unsqueeze(1).to(self.device) # (1,1,len)
            compressed = self.soundstream.encode(wav, target_bw=self.bw) # [n_codebook, 1, n_frames]
            compressed = compressed.squeeze(1).detach().cpu().numpy() # [n_codebook, n_frames]
            flat_codec = self._flatten_codebooks(compressed, self.dim_codebook)
            flat_codec = torch.from_numpy(flat_codec)
            flat_codec = flat_codec.to(torch.int16)
            return flat_codec
        else:
            raise NotImplementedError

    def tokenize2(self, wav):
        if isinstance(wav, str):
            # if x is the wave path
            return self.encode(wav)
        elif isinstance(wav, torch.Tensor):
            if wav.dim() == 1: # already done offline
                return wav
            if wav.dim() == 2: # transfer to 3 dim
                wav = wav.unsqueeze(1).to(self.device) # (1,1,len)
            compressed = self.soundstream.encode(wav, target_bw=self.bw) # [n_codebook, 1, n_frames]
            return compressed
        else:
            raise NotImplementedError

    @property
    def codebook_length(self):
        return self.dim_codebook * self.n_codebook + 1

    def find_length(self, x):
        return self.tokenize(x).shape[0] // self.n_codebook


if __name__ == '__main__':
    tokenizer = AudioTokenizer(device=torch.device('cuda:0')).cuda()
    # wav = tokenizer.detokenize(codec)
    # import torchaudio
    # torchaudio.save('desing.wav', wav, 16000, bits_per_sample=16, encoding='PCM_S')
    # print(wav.shape)
