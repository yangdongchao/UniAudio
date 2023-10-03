import torch
import torchaudio

from tools.tokenizer.abs_tokenizer import AbsTokenizer
from tools.tokenizer.LogMel.default import DefaultFrontend
from tools.tokenizer.LogMel.specaug.specaug import SpecAug

class LogMelTokenizer(AbsTokenizer):
    def __init__(self, 
                 downsampling=4, 
                 sample_rate=16000
        ):
        super(LogMelTokenizer, self).__init__()
        self.logmel = DefaultFrontend(
            n_fft=512,
            hop_length=160,
            fs=sample_rate,
        )

        self.specaug = SpecAug(
            apply_time_warp=True,
            time_warp_window=5,
            time_warp_mode='bicubic',
            apply_freq_mask=True,
            freq_mask_width_range=(0,27),
            num_freq_mask=2,
            apply_time_mask=True,
            time_mask_width_ratio_range=(0, 0.05),
            num_time_mask=10,
        )

        self.downsampling = downsampling
        self.sample_rate = sample_rate

    @property
    def is_discrete(self):
        return False

    @property
    def codebook_length(self):
        return 1 # so it is not a prompt tokenizer

    def find_length(self, x):
        return len(self.tokenize(x)) 

    def tokenize(self, x, task=None, cache=None):
        assert isinstance(x, str)
        x, sr = torchaudio.load(x)

        assert x.size(0) == 1, 'only support single-channel audio'
 
        if sr != self.sample_rate:
            x = torchaudio.transforms.Resample(sr, self.sample_rate)(x)

        x_len = torch.Tensor([x.size(1)]).long()
        x, x_len = self.logmel(x, x_len)

        if task == 'logmel_asr' and cache['is_train']:
            x, x_len = self.specaug(x, x_len)

        x = x.squeeze(0)
        assert x.size(0) == x_len[0]

        # concat every `downsampling` frames as one new frame
        end = x.size(0) // self.downsampling * self.downsampling
        x = x[:end].reshape(x.size(0) // self.downsampling, -1)

        return x
