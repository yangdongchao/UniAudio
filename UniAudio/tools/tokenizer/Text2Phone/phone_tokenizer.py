import torch
import logging
from valle.tools.tokenizer.abs_tokenizer import AbsTokenizer

default_phone_dict = "tools/tokenizer/Text2Phone/alignment_dict"

class PhoneTokenizer(AbsTokenizer):
    """
    This is the virtual tokenizer class.
    Other tokenizers should inherit this class.
    typicially:
        Text -> BPE
        Text -> Phone
        Audio -> Codec
        Image -> Codec
        ...
    """

    def __init__(self, phone_table=default_phone_dict, duplicate=False, unk_ph=None):
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
            self.phone_dict[unk_ph] = len(self.phone_dict)
        self.unk_id = phone_dict[unk_ph]

        self.duplicate = duplicate

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
            x = torch.unique_consequtive(x) if not self.duplicate else x
            return x.to(torch.int16)
        elif isinstance(x, str):
            x = [self.phone_dict.get(ph, self.unk_id) for ph in x.strip().split()]
            return torch.Tensor(x).to(torch.int16)
        else:
            raise NotImplementedError
