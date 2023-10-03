import torch
import torch.nn as nn
from tools.tokenizer.Text_Image.clip.clip import tokenize
#from tools.tokenizer.Text_Image.base_codec import BaseCodec
import importlib
from tools.tokenizer.abs_tokenizer import AbsTokenizer
def instantiate_from_config(config):
    if config is None:
        return None
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    module, cls = config["target"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module, package=None), cls)
    return cls(**config.get("params", dict()))

class BPETokenizer(AbsTokenizer):
    def __init__(self, context_length:int = 77,
                 add_start_and_end:bool = False,
                 just_token = False,
                 with_mask:bool = True,
                 pad_value:int = 0,
                 clip_embedding = False,
                 tokenizer_config={
                     'target': 'tools.tokenizer.Text_Image.clip.simple_tokenizer.SimpleTokenizer',
                     'params':{
                        'end_idx': 49152 # 16384 fo DALL-E
                        },
                 },
                 ):
        """
        This is a wrapper class for tokenize of texts.
        For CLIP and DALLE-pytorch tokenize, the default
        arguments are different:
        CLIP based:
            context_length: 77
            add_start_and_end: True

        DALLE-pytorch based:
            context_length: 256
            add_start_and_end: False
        
        """
        super().__init__()
        self.context_length = context_length
        self.add_start_and_end = add_start_and_end
        self.with_mask = with_mask
        self.pad_value = pad_value
        self.just_token = just_token
        self.trainable = False
        self.condition_emb = None
        self.clip_embedding = False
        self.tokenizer = instantiate_from_config(tokenizer_config)
    
    def __repr__(self):
        rep = "Tokenize for text\n\tcontent_length: {}\n\tadd_start_and_end: {}\n\twith_mask: {}"\
                .format(self.context_length, self.add_start_and_end, self.with_mask)
        return rep

    @property
    def is_discrete(self):
        return True

    @property
    def codebook_length(self):
        return 50000

    def check_length(self, token):
        return len(token) <= self.context_length

    def find_length(self, x):
        return len(self.tokenize(x))

    # def tokenize(self, x):
    #     tokens = self.get_tokens(x)['token']
    #     return tokens.squeeze()

    def detokenize(self, x):
        return self.tokenizer.decode(x)

    def tokenize(self, text, task=None, cache=None): # it return the bpe code and mask
        ret = tokenize(text, context_length=self.context_length, 
                         add_start_and_end=self.add_start_and_end,
                         with_mask=self.with_mask, pad_value=self.pad_value,
                         tokenizer=self.tokenizer,
                         just_token=self.just_token)
        ret = ret['token'][0][:ret['mask'].int().sum().item()]
        return ret

if __name__ == '__main__':
    bpe_tokenizer = BPETokenizer()
    text = "I am talking with you"
    phone = bpe_tokenizer.get_tokens(text)
    print(phone) # 'token', 'mask'
    # print(phone['token'].shape)
    # text = bpe_tokenizer.detokenize(phone['token'])
    # print('text ', text)



