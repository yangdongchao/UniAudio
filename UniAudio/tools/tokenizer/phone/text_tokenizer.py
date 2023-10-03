import torch
import logging
import sys
from os.path import join
import copy
from tools.tokenizer.abs_tokenizer import AbsTokenizer
# this scripts only be used in the data process 
# code is based on kalid
# text-to-phone
class Text2PhoneTokenizer(AbsTokenizer):
    def __init__(self, langdir='UniAudio/checkpoints/lang_nosp'):
        super(TextTokenizer, self).__init__()
        self.sil = open(join(langdir,
                "phones/optional_silence.txt")).readline().strip()
        self.oov_word = open(join(langdir, "oov.txt")).readline().strip()
        self.lexicon = {}
        with open(join(langdir, "phones/align_lexicon.txt")) as f:
            for line in f:
                line = line.strip()
                parts = line.split()
                self.lexicon[parts[0]] = parts[2:]  # ignore parts[1]

    def tokenize(self, x):
        # x must be str
        assert isinstance(x, str)
        x = x.replace('"','').replace('|',' ').replace('[','').replace(']','').replace('--',' ').replace('-',' ')
        x = x.replace('.','').replace(',',' ,').replace(':', ':').replace('!','!').replace('?','')
        word_trans = x.split()
        #print('word_trans ', word_trans)
        phone_trans = [self.sil] 
        for i in range(len(word_trans)):
            word = word_trans[i].upper()
            if word not in self.lexicon:
                if word in ['.',',',':',';','!','?']:
                    pronunciation = [self.sil]
                else:
                    pronunciation = self.lexicon[self.oov_word]
            else:
                pronunciation = copy.deepcopy(self.lexicon[word])
            phone_trans += pronunciation
        ans =  ' '.join(phone_trans)
        return  ans

ttokenizer = TextTokenizer()
txt = "Hello"
print(ttokenizer.tokenize(txt))