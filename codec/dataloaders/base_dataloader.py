import random

import torch
from torch.utils.data import Dataset
import torchaudio
from torchaudio.transforms import Resample


class WaveDataset(Dataset):
    def __init__(
        self,
        flist_file,
        segment_size,
        sampling_rate,
        split=True, # whether or not to get a segment of an audio sample to form the batch
        shuffle=False,
        audio_norm_scale: float = 1.0,
    ):
        self.file_list = self.get_filelist(flist_file)
        if shuffle:
            random.shuffle(self.file_list)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.audio_norm_scale = audio_norm_scale

    def get_filelist(self, fpath):
        with open(fpath, 'r') as f:
            flist = [l.strip() for l in f if l.strip()]
        return flist

    def __getitem__(self, index):
        fname = self.file_list[index]
        audio, sr = torchaudio.load(fname)
        if sr != self.sampling_rate:
            audio = Resample(sr, self.sampling_rate)(audio)
        if self.audio_norm_scale < 1.0:
            audio = audio * self.audio_norm_scale
        
        if self.split:
            if audio.size(1) >= self.segment_size:
                max_audio_start = audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                audio = audio[:, audio_start:audio_start+self.segment_size]
            else:
                audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')
        # in case, audio clip is too short in validation set
        if audio.size(1) < self.segment_size:
            audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        return audio

    def __len__(self):
        return len(self.file_list)
