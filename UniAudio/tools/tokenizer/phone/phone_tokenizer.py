import torch
import logging
from tools.tokenizer.abs_tokenizer import AbsTokenizer
from tools.tokenizer.common import speech_edit_find_time_stamp

class PhoneTokenizer(AbsTokenizer):
    def __init__(self, phone_table='tools/tokenizer/phone/phone_dict', unk_ph=None):
        super(PhoneTokenizer, self).__init__()

        phone_dict = open(phone_table, encoding="utf-8").readlines()
        phone_dict = [line.strip().split() for line in phone_dict]
        phone_dict = {line[0]: None for line in phone_dict}
        keys = list(phone_dict.keys())
        for i, k in enumerate(keys):
            phone_dict[k] = i
        self.phone_dict = phone_dict

        if unk_ph is None:
            self.unk_ph = "<UNK>"
            logging.info("No unknown phone provided. Set it as <UNK>.")
        else:
            self.unk_ph = unk_ph

        if unk_ph not in self.phone_dict:
            logging.info(f"Set unknown phone with number: {len(self.phone_dict)}")
            self.phone_dict[self.unk_ph] = len(self.phone_dict)
        self.unk_id = self.phone_dict[self.unk_ph]
        self.id2phone = {v: k for k, v in self.phone_dict.items()}

    @property
    def is_discrete(self):
        return True

    @property
    def codebook_length(self):
        return len(self.phone_dict)

    def find_length(self, x):
        return len(self.tokenize(x))

    def tokenize(self, x, task=None, cache=None):

        if task in ['speech_edit']:
            cache['speech_edit_time_stamp'] = speech_edit_find_time_stamp(x, self.id2phone)

        if task in ['tts', 'plain_tts', 'phone_to_semantic']:
            duplicate = False
        else: # e.g., offline, speech edit
            duplicate = True
 
        if isinstance(x, torch.Tensor):
            assert x.dim() == 1
            x = torch.unique_consecutive(x).to(torch.int16) if not duplicate else x
        elif isinstance(x, str):
            x = [self.phone_dict.get(ph, self.unk_id) for ph in x.strip().split()]
            x = torch.Tensor(x).to(torch.int16)
            x = torch.unique_consecutive(x) if not duplicate else x
        else:
            raise NotImplementedError

        return x

    def tokenize_batch(self, xs, lengths=None):
        raise NotImplementedError

    def detokenize(self, x):
        assert isinstance(x, torch.Tensor)
        x = x.cpu().tolist()
        x = [self.id2phone[y] for y in x]
        x = " ".join(x)
        return x

