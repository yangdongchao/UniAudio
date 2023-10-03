from tools.tokenizer.Text2Phone.modules.txt_processors.en import TxtProcessor
import argparse
from pathlib import Path
import sys
import os
import torch
import typing as tp
from collections import OrderedDict
import tqdm
import pandas as pd
import os
from tools.tokenizer.abs_tokenizer import AbsTokenizer
import numpy as np
import logging

class Text2PhoneTokenizer(AbsTokenizer):
    def __init__(self, duplicate=False):
        """ Transfer the text input to the phone sequence
        """
        super(Text2PhoneTokenizer, self).__init__()
        self.txt_processor = TxtProcessor() # init the text processor
        self.phone_dict_path = 'tools/tokenizer/Text2Phone/dict_phone.txt'
        self.phone_dict = self.load_dict(self.phone_dict_path)
        self.duplicate = duplicate

    def load_dict(self, path):
        f = open(path, 'r')
        idx = 0
        phone_dict = {}
        for line in f:
            tmp = line.split(' ')
            phone = tmp[0]
            phone_dict[phone] = idx
            idx += 1
        return phone_dict

    def get_phone_sequence(self, text): 
        # input the speech text, such as "I am talking with you". output the phone sequence
        phs, txt = self.txt_processor.process(text,  {'use_tone': True})
        return phs

    @property    
    def is_discrete(self):
        return True
    
    def find_length(self, x):
        return len(self.tokenize(x))
    
    def tokenize(self, x, task=None, cache=None):
        if isinstance(x, torch.Tensor):
            x = torch.unique_consecutive(x) if not self.duplicate else x 
            return x
        elif isinstance(x, str):
            phs = self.get_phone_sequence(x)
            idxs = [self.phone_dict[id] for id in phs]
            idxs = np.array(idxs)
            idxs = torch.from_numpy(idxs).to(torch.int16)
            return idxs
        else:
            raise NotImplementedError
   
    @property 
    def codebook_length(self):
        # return len(self.phone_dict.keys())
        return 500
          
if __name__ == '__main__':
    T2P_tokenizer = Text2PhoneTokenizer()
    text = "I am talking with you"
    phone = T2P_tokenizer.tokenize(text)
    print(phone) # AY1 | AE1 M | T AO1 K IH0 NG | W IH1 DH | Y UW1
