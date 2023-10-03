import os
import argparse
import glob
import sys

import numpy as np
import librosa
import torch
import laion_clap
from laion_clap import CLAP_Module
# quantization
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

clap = CLAP_Module(enable_fusion=False, device='cuda') 
clap.load_ckpt() 
model = clap.cuda()
def get_clap_score(audio_file, text_file):
    audio_list = {}
    text_list = {}
    f_a = open(audio_file)
    for line in f_a:
        ans = line.strip().split(' ')
        name = ans[0]
        ph = ans[1]
        audio_list[name] = ph

    f_t = open(text_file)
    for line in f_t:
        ans = line.strip().split(' ')
        name = ans[0]
        ph = ' '.join(ans[1:])
        text_list[name] = ph
    tmp_audio_list = []
    tmp_text_list = []
    clap_scores = []
    for i, key in enumerate(audio_list.keys()):
        audio_embed = model.get_audio_embedding_from_filelist(x = [audio_list[key]], use_tensor=True)
        text_embed = model.get_text_embedding(x=[text_list[key], text_list[key]], use_tensor=True)
        cosine_sim = torch.nn.CosineSimilarity(dim=-1)
        text_embed = text_embed[1:2,:]
        audio_embed = torch.nn.functional.normalize(audio_embed, dim=-1).cpu()
        text_embed = torch.nn.functional.normalize(text_embed, dim=-1).cpu()
        similarity = cosine_sim(audio_embed, text_embed)
        print('similarity ', similarity)
        clap_scores.append(abs(similarity.item()))
    return np.mean(clap_scores)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute PESQ measure.")
    parser.add_argument(
        '-a',
        '--audio_file',
        required=True,
        help="save the audio file path."
    )
    parser.add_argument(
        '-t',
        '--text_file',
        required=True,
        help="save the text file path."
    )
    args = parser.parse_args()
    score = get_clap_score(args.audio_file, args.text_file)
    print('clap score ', score)

