import re
from modules.data_gen_utils import PUNCS
from modules.txt_processors import zh
from utils.text_norm import NSWNormalizer


class TxtProcessor(zh.TxtProcessor):
    @staticmethod
    def preprocess_text(text):
        text = text.translate(TxtProcessor.table)
        text = NSWNormalizer(text).normalize(remove_punc=False)
        text = re.sub("[\'\"()]+", "", text)
        text = re.sub("[-]+", " ", text)
        text = re.sub(f"[^ A-Za-z\u4e00-\u9fff{PUNCS}&]", "", text)
        text = re.sub(f"([{PUNCS}])+", r"\1", text)  # !! -> !
        text = re.sub(f"([{PUNCS}])", r" \1 ", text)
        text = re.sub(rf"\s+", r"", text)
        return text

    @staticmethod
    def sp_phonemes():
        return ['|', '#', '&']

    @classmethod
    def process(cls, txt, pre_align_args):
        txt = txt.replace('SEP', '&')
        ph_list, txt = super().process(txt, pre_align_args)
        txt = txt.replace('&', ' SEP ')
        ph_list = [p if p != '&' else 'SEP' for p in ph_list if p not in ['|', '#', '<BOS>', '<EOS>']]
        return ph_list, txt
