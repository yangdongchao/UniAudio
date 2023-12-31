# Copyright 2022 
# Author     : UniAudio Teams 
# Description: Compute STOI measure.

import os
import glob
import argparse

from tqdm import tqdm
from scipy.io import wavfile
from pystoi import stoi
import numpy as np
import librosa

def calculate_stoi(ref_dir, deg_dir):
    input_files = glob.glob(f"{deg_dir}/*.wav")
    if len(input_files) < 1:
        raise RuntimeError(f"Found no wavs in {ref_dir}")

    stoi_scores = []
    for deg_wav in tqdm(input_files):
        ref_wav = os.path.join(ref_dir, os.path.basename(deg_wav))
        ref, ref_rate = librosa.load(ref_wav)
        deg, deg_rate = librosa.load(deg_wav)
        if ref_rate != 16000:
            ref = librosa.resample(ref, ref_rate, 16000)
            ref_rate = 16000
        if deg_rate != 16000:
            deg = librosa.resample(deg, deg_rate, 16000)
            deg_rate = 16000
        rate = 16000
        # rate, ref = wavfile.read(ref_wav)
        # rate, deg = wavfile.read(deg_wav)
        min_len = min(len(ref), len(deg))
        ref = ref[:min_len]
        deg = deg[:min_len]
        cur_stoi = stoi(ref, deg, rate, extended=False)
        stoi_scores.append(cur_stoi)

    return np.mean(stoi_scores)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Compute STOI measure")

    parser.add_argument(
        '-r',
        '--ref_dir',
        required=True,
        help="Reference wave folder."
    )
    parser.add_argument(
        '-d',
        '--deg_dir',
        required=True,
        help="Degraded wave folder."
    )

    args = parser.parse_args()

    stoi_score = calculate_stoi(args.ref_dir, args.deg_dir)
    print(f"STOI: {stoi_score}")
