import torch
from tools.tokenizer.abs_tokenizer import AbsTokenizer

class SVBoolTokenizer(AbsTokenizer):
    def __init__(self):
        super(SVBoolTokenizer, self).__init__()

    @property
    def is_discrete(self):
        return True

    @property
    def codebook_length(self):
        return 2 # True or False

    def find_length(self, x):
        return 1

    def tokenize(self, x, task=None, cache=None):
        #print('x ', x)
        if cache is not None and "SV_label" in cache:
            # training, random selected by skpid
            ans = cache["SV_label"]
        else:
            ans = int(x)
        #print('ans ', ans)
        ans = torch.Tensor([ans]).long()
        return ans

    def detokenize(self, x):
        return x.item()
