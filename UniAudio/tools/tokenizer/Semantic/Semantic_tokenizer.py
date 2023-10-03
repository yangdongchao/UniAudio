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

from tools.tokenizer.Semantic.hubert_kmeans import HubertWithKmeans
from tools.tokenizer.abs_tokenizer import AbsTokenizer

class SemanticTokenizer(AbsTokenizer):
    def __init__(self, device=torch.device('cpu'), duplicate=True):
        """  Hubert model for extract semantic token
        """
        super(SemanticTokenizer, self).__init__()
        # GPU is only for offline tokenization
        # So, when distributed training is launched, this should still be on CPU
        self.device = device

        self.hubert_path = 'UniAudio/checkpoints/hubert_base_ls960.pt'
        self.quantizer_path = 'UniAudio/checkpoints/hubert_base_ls960_L9_km500.bin'
        self.hubert_kmeans = HubertWithKmeans(checkpoint_path=self.hubert_path, kmeans_path=self.quantizer_path)
        self.hubert_kmeans = self.hubert_kmeans.to(self.device)

        logging.info(f"hubert semantic model works on {self.device}")

        self.sr = 16 * 1000
        self.dim_codebook = 500
        self.duplicate = duplicate # 默认为True，应该保留时序信息
    
    def encode(self, wav_root, sr=16000):
        wav, sr = torchaudio.load(wav_root)
        if wav.numel() == 0:
            return None
        if sr != self.sr:
            wav = torchaudio.transforms.Resample(sr, self.sr)(wav)
        wav = wav.to(self.device)
        flat_codec = self.hubert_kmeans(wav)
        if not self.duplicate:
            flat_codec = self.batch_unique_consecutive(flat_codec)
        flat_codec = flat_codec.to(torch.int16)
        return flat_codec

    @property
    def is_discrete(self):
        return True

    def tokenize(self, wav, task=None, cache=None):
        if isinstance(wav, str):
            # if x is the wave path
            return self.encode(wav) #TODO: make it consistent with torch.Tensor branch
        elif isinstance(wav, torch.Tensor):
            if wav.dim() == 1:
                flat_codec = wav
            elif wav.dim() == 2: # is the input is wav [1, T]
                wav = wav.to(self.device)
                if wav.numel() == 0:
                    return None
                flat_codec = self.hubert_kmeans(wav)
            else:
                raise NotImplementedError
            if not self.duplicate:
                flat_codec = self.batch_unique_consecutive(flat_codec)
            flat_codec = flat_codec.to(torch.int16)
            return flat_codec
        else:
            raise NotImplementedError

    def batch_unique_consecutive(self, t): # delete the repeated tokens
        t = t.unsqueeze(0)
        unique_arr = [torch.unique_consecutive(el) for el in t.unbind(dim = 0)]
        return unique_arr[0]
    @property
    def codebook_length(self):
        return self.dim_codebook

    def find_length(self, x): # we first calculate the codec for wave x, then we get the length
        return self.tokenize(x).shape[0]

    def detokenize(self, x):
        x = x.cpu().tolist()
        ans = ''
        for a in x:
            ans = ans + str(a) + ' '
        return ans[:-1]


if __name__ == '__main__':
    tokenizer = SemanticTokenizer(device=torch.device('cuda:0')).cuda()
    
