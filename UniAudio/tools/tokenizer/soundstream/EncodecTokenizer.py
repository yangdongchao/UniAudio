import torch
import torchaudio
import logging

from encodec import EncodecModel
from encodec.utils import convert_audio
import torch.distributed as dist

from tools.tokenizer.abs_tokenizer import AbsTokenizer

class EncodecTokenizer(AbsTokenizer):
    def __init__(self, device=torch.device('cpu'), select_every=1):
        AbsTokenizer.__init__(self)

        self.tokenizer = EncodecModel.encodec_model_24khz()
        self.tokenizer.set_target_bandwidth(6.0)

        self.sr = self.tokenizer.sample_rate
        self.n_codebook = 8
        self.size_codebook = 1024

        self.freq = 75 * self.n_codebook # 75 vectors / frame

        self.device = device
        self.select_every = select_every

    @property
    def is_discrete(self):
        return True

    @property
    def codebook_length(self):
        return self.size_codebook * self.n_codebook

    def tokenize(self, x):
        if isinstance(x, str):
            x, sr = torchaudio.load(x)

            if sr != self.sr:
                convert_audio(x, sr, self.tokenizer.sample_rate, self.tokenizer.channels)

            if x.numel() == 0:
                return None

            x = x.to(self.device)
            encoded_frames = self.tokenizer.encode(x.unsqueeze(0))
            encoded_frames = encoded_frames[0][0][0].to(torch.int16)
            ans = self.flatten(encoded_frames)
            ans = ans[::self.select_every]
            return ans         

        elif isinstance(x, torch.Tensor):
            assert x.dim() == 1, "Input dim should be 1 as a complete version"
            return x[::self.select_every]
        else:
            raise NotImplementedError

    def find_length(self, x):
        return len(self.tokenize(x)) // self.n_codebook

    def tokenize_batch(self, x):
        raise NotImplementedError

    def detokenize(self, x):
        x = x.to(self.device)
        x = x.view(-1, self.n_codebook).transpose(0, 1)
        for i in range(len(x)):
            x[i] -= self.size_codebook * i
        x = x.unsqueeze(0).to(torch.int64)
        x = [(x, None)]
        x = self.tokenizer.decode(x).squeeze(0)
        return x 

    def flatten(self, x):
        for i in range(len(x)):
            x[i] += self.size_codebook * i
        return x.transpose(0, 1).contiguous().reshape(-1)

if __name__ == "__main__":
    tokenizer = EncodecTokenizer(device = torch.device('cuda:0')).cuda()


