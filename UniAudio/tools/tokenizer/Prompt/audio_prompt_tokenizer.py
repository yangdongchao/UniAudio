import torch
import random
from tools.tokenizer.abs_tokenizer import AbsTokenizer
from tools.tokenizer.common import clip_by_length

class AudioPromptTokenizer(AbsTokenizer):
    """ This tokenizer samples a audio prompt from the given speaker 
        User should ensure that each example has the data key 'audio'.

        Sometimes the audio prompt is exactly a piece of audio rather than 
        the labeled speaker. In that case we simply use that piece.
    """
    def __init__(self, data_dict, prompt_length, n_codebook):
        AbsTokenizer.__init__(self)

        self.data_dict = data_dict
        self.spk2utt = self.parse_spk2utt(data_dict)
        self.prompt_length = prompt_length
        self.n_codebook = n_codebook
        self.speakers = list(self.spk2utt.keys())

    def parse_spk2utt(self, data_dict):
        spk2utt = {}
        for example_id, d in data_dict.items():
            if d['task'] not in ['tts', 'VC', 'chn_tts', 'SV']:
                continue

            spk = d['prompt_seq']
            if isinstance(spk, torch.Tensor):
                continue

            if spk not in spk2utt:
                spk2utt[spk] = []
            spk2utt[spk].append(example_id)

        return spk2utt

    @property
    def is_discrete(self):
        return True
    
    @property
    def codebook_length(self):
        """ It shares the same codebook with AudioTokenizer """
        return 0 
        
    def find_length(self, _):
        return self.prompt_length / self.n_codebook

    def tokenize(self, x, task=None, cache=None):
        if cache is not None and 'speech_edit_target' in cache:
            return cache['speech_edit_target']

        elif isinstance(x, torch.Tensor):
            return self.tokenize_audio(x)
        else:
            return self.tokenize_spk(x, task, cache)

    def tokenize_audio(self, x):
        """ here x is a piece of audio. Force the prompt audio be this """
        if len(x) > self.prompt_length:
            start = random.randint(0, len(x) - self.prompt_length - 1)
            start  = start // self.n_codebook * self.n_codebook
            return x[start: start + self.prompt_length]
        else:
            return x

    def tokenize_spk(self, x, task=None, cache=None):
        """ Here x is the spk-id """

        # For speaker verification, simply change the prompt speaker with prob=50%
        if task == 'SV':
            if random.random() > 0.5:
                while True:
                    prev_x = x
                    x = random.sample(self.speakers, 1)[0]
                    if x != prev_x: # cannot sample itself
                        break
                cache['SV_label'] = False
            else:
                cache['SV_label'] = True

        for _ in range(5):
            uttid = random.sample(self.spk2utt[x], 1)[0]
            audio = self.data_dict[uttid]['audio_seq']
            
            if len(audio) <= self.prompt_length + 1:
                continue # ignore the current audio if this is too short

            return clip_by_length(audio, self.prompt_length, self.n_codebook)

        return audio
