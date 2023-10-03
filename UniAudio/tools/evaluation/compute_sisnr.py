# Copyright 2022
# Author     : UniAudio Teams
# Description: Compute SISNR measure.s
import glob
import argparse
from tqdm import tqdm
from pesq import pesq
from scipy.io import wavfile
import scipy.signal as signal
import torchaudio
from sisnr import sisnr_loss, get_sisnr, estimate_si_sdr
import os
import librosa
def cal_sisnr(ref_dir, deg_dir, sample_rate=16000):
    input_files = glob.glob(f"{deg_dir}/*.wav") # get wavs
    sisnr_scores = 0.0
    for deg_wav in tqdm(input_files):
        ref_wav = os.path.join(ref_dir, os.path.basename(deg_wav))
        clean_signal = librosa.core.load(ref_wav, sr=sample_rate, mono=True)
        enhanced_signal = librosa.core.load(deg_wav, sr=sample_rate, mono=True)
        sisnr_scores += estimate_si_sdr(clean_signal, enhanced_signal)
    return  sisnr_scores/len(input_files) # return the avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute SISNR measure.")
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
    sisnr_score = cal_sisnr(args.ref_dir, args.deg_dir)
    print(f"sisnr_score: {sisnr_score}")
