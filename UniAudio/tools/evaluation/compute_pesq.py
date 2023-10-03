# Copyright 2022 
# Author     : UniAudio Teams 
# Description: Compute PESQ measure.

import os
import glob
import argparse
from tqdm import tqdm
from pesq import pesq
from scipy.io import wavfile
import scipy.signal as signal
import librosa
from pypesq import pesq as nb_pesq
def WB_PESQ(ref, est, sr=16000):
    if sr != 16000:
        wb_ref = librosa.resample(ref, sr, 16000)
        wb_est = librosa.resample(est, sr, 16000)
    else:
        wb_ref = ref
        wb_est = est
    # pesq will not downsample internally
    return pesq(16000, wb_ref, wb_est, "wb")


def NB_PESQ(ref, est, sr=16000):
    if sr != 8000:
        nb_ref = librosa.resample(ref, sr, 8000)
        nb_est = librosa.resample(est, sr, 8000)
    else:
        nb_ref = ref
        nb_est = est
    # nb_pesq downsample to 8000 internally.
    return nb_pesq(nb_ref, nb_est, 8000)

def cal_pesq(ref_dir, deg_dir):
    input_files = glob.glob(f"{deg_dir}/*.wav")

    nb_pesq_scores = 0.0
    wb_pesq_scores = 0.0
    cnt = 0
    for deg_wav in tqdm(input_files):
        #print(deg_wav)
        ref_wav = os.path.join(ref_dir, os.path.basename(deg_wav))
        ref, ref_rate = librosa.load(ref_wav)
        deg, deg_rate = librosa.load(deg_wav)
        if ref_rate != 16000:
            ref = librosa.resample(ref, ref_rate, 16000)
            ref_rate = 16000
        if deg_rate != 16000:
            deg = librosa.resample(deg, deg_rate, 16000)
            deg_rate = 16000

        min_len = min(len(ref), len(deg))
        ref = ref[:min_len]
        deg = deg[:min_len]
        # nb_pesq_scores += NB_PESQ(ref, deg, 16000)
        # wb_pesq_scores += WB_PESQ(ref, deg, 16000)
        try:
            nb_pesq = pesq(16000, ref, deg, 'nb')
            wb_pesq = pesq(16000, ref, deg, 'wb')
            nb_pesq_scores += nb_pesq
            wb_pesq_scores += wb_pesq
            cnt += 1
        except:
            print('eer')
    #nb_pesq_scores = 0
    return  nb_pesq_scores/cnt, wb_pesq_scores/cnt


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Compute PESQ measure.")

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

    nb_score, wb_score = cal_pesq(args.ref_dir, args.deg_dir)
    print(f"NB PESQ: {nb_score}")
    print(f"WB PESQ: {wb_score}")
