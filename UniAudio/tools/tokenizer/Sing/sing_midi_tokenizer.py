import torch
import logging
import sys
from tools.tokenizer.abs_tokenizer import AbsTokenizer

class SingMidiTokenizer(AbsTokenizer):
    def __init__(self, phone_table='UniAudio/tools/tokenizer/Sing/f0_dict.txt'):
        AbsTokenizer.__init__(self)
        phone_dict = open(phone_table, encoding="utf-8").readlines()
        phone_dict = [line.strip().split() for line in phone_dict]
        phone_dict = {line[0]: None for line in phone_dict}
        # print('phone_dict ', phone_dict)
        keys = list(phone_dict.keys())
        for i, k in enumerate(keys):
            phone_dict[k] = i
        self.phone_dict = phone_dict

    @property
    def is_discrete(self):
        return True

    @property
    def codebook_length(self):
        return len(self.phone_dict)

    def find_length(self, x):
        return len(self.tokenize(x))

    def tokenize(self, x, task=None, cache=None):
        if isinstance(x, torch.Tensor):
            assert x.dim() == 1
            #x = torch.unique_consecutive(x) if not self.duplicate else x
            return x.to(torch.int16)
        elif isinstance(x, str):
            seq = x.strip().split(' ')
            if seq[0].isnumeric(): # if we have transfer to number
                x = [int(ph) for ph in seq]
            else:
                x = [self.phone_dict.get(ph) for ph in x.strip().split()]
            x = torch.Tensor(x).to(torch.int16)
            #x = torch.unique_consecutive(x) if not self.duplicate else x
            return x
        else:
            raise NotImplementedError

    def tokenize_batch(self, xs, lengths=None):
        raise NotImplementedError

    def detokenize(self, x):
        raise NotImplementedError
    
if __name__ == '__main__':
    tokenizer = SingMidiTokenizer()