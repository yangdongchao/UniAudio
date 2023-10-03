import torch
import sentencepiece as spm
import sys
import logging
from tools.tokenizer.abs_tokenizer import AbsTokenizer

class BPETokenizer(AbsTokenizer):
    def __init__(self, model_dir='tools/tokenizer/BPE/sentencepiece_model'):
        
        super().__init__()
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_dir + '/bpe.model')
        
        vocab = open(model_dir + '/char.txt').readlines()
        vocab = [line.strip().split() for line in vocab]
        self.tok2id = {line[0]: int(line[1]) for line in vocab}
        self.id2tok = {int(line[1]): line[0] for line in vocab}
        self.dtype = torch.int32 if self.codebook_length > 2**15 -1 else torch.int16

    @property
    def is_discrete(self):
        return True

    @property
    def codebook_length(self):
        return max(list(self.id2tok.keys())) + 1 # consider 0

    def find_length(self, x):
        return len(self.tokenize(x))

    def detokenize(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().tolist()
        return self.sp.DecodeIds(x)

    def tokenize(self, text, task=None, cache=None): # it return the bpe code and mask
        if isinstance(text, torch.Tensor):
            return text
        elif isinstance(text, str):
            ans = torch.Tensor(self.sp.EncodeAsIds(text)).to(self.dtype)

            if ans.numel() == 0:
                logging.warning(f"empty example: {text} -> {ans}")
                return None

            if ans.max() >= self.codebook_length: 
                logging.warning(f"invalid example: {text} -> {ans}")
                return None

            return ans
        else:
            logging.info(f"unrecognized type: {type(text)}: {text}")
            print(f"unrecognized type: {type(text)}: {text}", flush=True)
            raise NotImplementedError


if __name__ == '__main__':
    bpe_tokenizer = BPETokenizer()
    text = sys.argv[1:]
    text = [int(x) - 3280 for x in text]
    # phone = bpe_tokenizer.tokenize(text)
    ans = bpe_tokenizer.detokenize(text)
    print(ans)
