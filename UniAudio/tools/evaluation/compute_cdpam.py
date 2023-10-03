# Copyright 2023
# Author     : UnAudio Teams

import os
import glob
import argparse
from tqdm import tqdm
from scipy.io import wavfile
from pystoi import stoi
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import torch
import scipy.signal as signal
# from datasets import load_dataset
import soundfile as sf
import librosa
import cdpam
loss_fn = cdpam.CDPAM(dev='cuda:0')

def calculate_CDPAM(ref_dir, deg_dir):
    deg_files = glob.glob(f"{deg_dir}/*.wav")
    if len(deg_files) < 1:
        raise RuntimeError(f"Found no wavs in {deg_dir}")
    cdpam_scores = []
    for deg_wav in tqdm(deg_files):
        ref_wav = os.path.join(ref_dir, os.path.basename(deg_wav)) # 
        # ref, ref_rate = librosa.load(ref_wav, sr=16000)
        # deg, deg_rate = librosa.load(deg_wav, sr=16000)
        ref = cdpam.load_audio(ref_wav)
        deg = cdpam.load_audio(deg_wav)
        cdpam_score = loss_fn.forward(ref, deg)
        cdpam_scores.append(cdpam_score.item())
    return np.mean(cdpam_scores)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute speaker similarity_score")
    parser.add_argument(
        '-r',
        '--ref_dir',
        required=True,
        help="Reference wave folder or file list."
    )
    parser.add_argument(
        '-d',
        '--deg_dir',
        required=True,
        help="Degraded wave folder."
    )
    args = parser.parse_args()
    similarity_score = calculate_CDPAM(args.ref_dir, args.deg_dir)
    print(f"Speaker similarity: {similarity_score}")
