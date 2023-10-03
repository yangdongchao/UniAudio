from syllabipy.sonoripy import SonoriPy
from modules.txt_processors import en


class TxtProcessor(en.TxtProcessor):
    @classmethod
    def process(cls, txt, pre_align_args):
        txt = cls.preprocess_text(txt)
        phs = []
        for p in txt.split(" "):
            if len(p) == 0:
                continue
            syl = SonoriPy(p)
            if len(syl) == 0:
                phs += list(p)
            else:
                for x in syl:
                    phs += list(x)
                phs += ['|']
        return phs, txt
