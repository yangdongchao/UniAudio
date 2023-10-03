# Copyright 2022 
# Author     : UniAudio Teams 
# Description: Compute SSIM measure.

import os
import glob
import argparse

import torch
import numpy as np
from tqdm import tqdm
import librosa

from ssim import SSIM


def load_wav(wav_path, sr=24000):
    audio = librosa.core.load(wav_path, sr=sr)[0]
    return audio


_mel_basis = None

def _build_mel_basis(hparams):
    assert hparams.fmax <= hparams.sample_rate // 2
    return librosa.filters.mel(hparams.sample_rate,
                               hparams.n_fft,
                               n_mels=hparams.acoustic_dim,
                               fmin=hparams.fmin,
                               fmax=hparams.fmax)

def _linear_to_mel(spectogram, hparams):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(hparams)
    return np.dot(_mel_basis, spectogram)


def _stft(y, hparams):
    return librosa.stft(y=y,
                        n_fft=hparams.n_fft,
                        hop_length=hparams.hop_size,
                        win_length=hparams.win_size)


def _amp_to_db(x, hparams):
    min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _normalize(S, hparams):
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_acoustic:
            return np.clip((2 * hparams.max_abs_value) * ((S - hparams.min_level_db) /
                                                          (-hparams.min_level_db)) -
                           hparams.max_abs_value,
                           -hparams.max_abs_value, hparams.max_abs_value)
        else:
            return np.clip(hparams.max_abs_value * ((S - hparams.min_level_db) /
                                                    (-hparams.min_level_db)),
                           0, hparams.max_abs_value)

    assert S.max() <= 0 and S.min() - hparams.min_level_db >= 0
    if hparams.symmetric_acoustic:
        return ((2 * hparams.max_abs_value) *
                ((S - hparams.min_level_db) / (-hparams.min_level_db)) -
                hparams.max_abs_value)
    else:
        return (hparams.max_abs_value *
                ((S - hparams.min_level_db) / (-hparams.min_level_db)))


def melspectrogram(wav, hparams, compute_energy=False):
    D = _stft(wav, hparams)
    S = _amp_to_db(_linear_to_mel(np.abs(D), hparams),
                   hparams) - hparams.ref_level_db
    if hparams.signal_normalization:
        S = _normalize(S, hparams)
    
    if compute_energy:
        energy = np.linalg.norm(np.abs(D), axis=0)
        return S.astype(np.float32), energy.astype(np.float32)
    else:
        return S


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Unsupported value encountered.')


def compute_mel(wav_path, args):
    wav = load_wav(wav_path, sr=args.sample_rate)
    mel_spec = melspectrogram(wav, args).T

    return mel_spec


def calculate_ssim(args):
    input_files = glob.glob(f"{args.deg_dir}/*.wav")
    if len(input_files) < 1:
        raise RuntimeError(f"Found no wavs in {args.ref_dir}")

    ssim_obj = SSIM(data_range=args.max_abs_value, channel=1, size_average=False)
    ssims = []

    for deg_wav in tqdm(input_files):
        ref_wav = os.path.join(args.ref_dir, os.path.basename(deg_wav))
        ref_mel = compute_mel(ref_wav, args)
        deg_mel = compute_mel(deg_wav, args)

        ref_mel = torch.from_numpy(ref_mel).view(1, 1, ref_mel.shape[0], ref_mel.shape[1])
        deg_mel = torch.from_numpy(deg_mel).view(1, 1, deg_mel.shape[0], deg_mel.shape[1])

        min_len = min(ref_mel.shape[-2], deg_mel.shape[-2])
        ref_mel = ref_mel[:, :, :min_len, :]
        deg_mel = deg_mel[:, :, :min_len, :]

        ssim = ssim_obj(deg_mel, ref_mel)
        ssims.append(ssim[0].item())

    return np.mean(ssims)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute SSIM")

    parser.add_argument('--ref_dir', '-r', required=True)
    parser.add_argument('--deg_dir', '-d', required=True)

    mel_group = parser.add_argument_group(title="Mel options")
    mel_group.add_argument('--sample_rate', type=int, default=16000)
    mel_group.add_argument('--n_fft', type=int, default=1024)
    mel_group.add_argument('--acoustic_dim', type=int, default=80)
    mel_group.add_argument('--hop_size', type=int, default=256)
    mel_group.add_argument('--win_size', type=int, default=1024)
    mel_group.add_argument('--min_level_db', type=int, default=-100)
    mel_group.add_argument('--ref_level_db', type=int, default=20)
    mel_group.add_argument('--symmetric_acoustic', type=str2bool, default=True)
    mel_group.add_argument('--signal_normalization', type=str2bool, default=True)
    mel_group.add_argument('--allow_clipping_in_normalization', type=str2bool, default=True)
    mel_group.add_argument('--max_abs_value', type=float, default=1)
    mel_group.add_argument('--fmin', type=int, default=0)
    mel_group.add_argument('--fmax', type=int, default=8000)

    args = parser.parse_args()
    # SSIM requires data to have symmetric value range around 0
    args.symmetric_acoustic = True

    ssim_result = calculate_ssim(args)
    print(f"SSIM: {ssim_result}")
