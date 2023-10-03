import os
import argparse
import glob
import sys
# 
sys.path.append('visqol/visqol_lib_py')
import visqol_lib_py
import visqol_config_pb2
import similarity_result_pb2
from tqdm import tqdm

import torch
import torchaudio
from torchaudio.transforms import Resample # Resampling
import numpy as np
import tempfile
import soundfile as sf

VISQOLMANAGER = visqol_lib_py.VisqolManager()
VISQOLMANAGER.Init(visqol_lib_py.FilePath( \
    'visqol/model/lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite'), \
    True, False, 60, True)

def visqol_speech_24k(ests, refs, sr=16000):
    if sr != 16000:
        resample = Resample(sr, 16000)
        ests = resample(ests)
        refs = resample(refs)
        sr = 16000
    ests = ests.view(-1, ests.shape[-1])
    refs = refs.view(-1, refs.shape[-1])
    outs = []
    with tempfile.TemporaryDirectory() as tmpdirname:
        for curinx in range(ests.shape[0]):
            sf.write("{}/est_{:07d}.wav".format(tmpdirname,curinx),ests[curinx].detach().cpu().numpy(),sr)
            sf.write("{}/ref_{:07d}.wav".format(tmpdirname,curinx),refs[curinx].detach().cpu().numpy(),sr)
            out = VISQOLMANAGER.Run( \
                visqol_lib_py.FilePath("{}/ref_{:07d}.wav".format(tmpdirname,curinx)), \
                visqol_lib_py.FilePath("{}/est_{:07d}.wav".format(tmpdirname,curinx)))
            outs.append(out.moslqo)
    # return torch.Tensor([np.mean(outs)]).to(ests.device)

    return np.mean(outs)


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

    input_files = glob.glob(f"{args.deg_dir}/*.wav")

    visqol = []
    for deg_wav in tqdm(input_files):
        ref_wav = os.path.join(args.ref_dir, os.path.basename(deg_wav))
        deg_wav, fs = torchaudio.load(deg_wav)
        ref_wav, fs = torchaudio.load(ref_wav)
        cur_visqol = visqol_speech_24k(deg_wav, ref_wav, sr=fs)
        visqol.append(cur_visqol)
    
    print(f"VISQOL: {np.mean(visqol)}")
