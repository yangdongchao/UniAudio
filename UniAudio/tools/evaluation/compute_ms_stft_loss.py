# Copyright 2022
# Author     : UniAudio Teams 
# Description: Compute multi-resolution STFT Loss.

import os
import glob
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import librosa


def stft(x, fft_size, hop_size, win_size, window):
    x_stft = torch.stft(x, fft_size, hop_size, win_size, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]
    outputs = torch.clamp(real ** 2 + imag ** 2, min=1e-7).transpose(2, 1)
    outputs = torch.sqrt(outputs)

    return outputs


class SpectralConvergence(nn.Module):
    def __init__(self):
        super(SpectralConvergence, self).__init__()

    def forward(self, predicts_mag, targets_mag):
        x = torch.norm(targets_mag - predicts_mag, p='fro')
        y = torch.norm(targets_mag, p='fro')

        return x / y


class LogSTFTMagnitude(nn.Module):
    def __init__(self):
        super(LogSTFTMagnitude, self).__init__()

    def forward(self, predicts_mag, targets_mag):
        log_predicts_mag = torch.log(predicts_mag)
        log_targets_mag = torch.log(targets_mag)
        outputs = F.l1_loss(log_predicts_mag, log_targets_mag)

        return outputs


class STFTLoss(nn.Module):
    def __init__(
        self,
        fft_size=1024,
        hop_size=120,
        win_size=600,
    ):
        super(STFTLoss, self).__init__()

        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.register_buffer('window', torch.hann_window(win_size))
        self.sc_loss = SpectralConvergence()
        self.mag = LogSTFTMagnitude()

    def forward(self, predicts, targets):
        predicts_mag = stft(predicts, self.fft_size, self.hop_size, self.win_size, self.window)
        targets_mag = stft(targets, self.fft_size, self.hop_size, self.win_size, self.window)

        sc_loss = self.sc_loss(predicts_mag, targets_mag)
        mag_loss = self.mag(predicts_mag, targets_mag)

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],
        win_sizes=[600, 1200, 240],
        hop_sizes=[120, 240, 50],
        **kwargs
    ):
        super(MultiResolutionSTFTLoss, self).__init__()
        self.loss_layers = torch.nn.ModuleList()
        for (fft_size, win_size, hop_size) in zip(fft_sizes, win_sizes, hop_sizes):
            self.loss_layers.append(STFTLoss(fft_size, hop_size, win_size))

    def forward(self, fake_signals, true_signals):
        sc_losses, mag_losses = [], []
        for layer in self.loss_layers:
            sc_loss, mag_loss = layer(fake_signals, true_signals)
            sc_losses.append(sc_loss)
            mag_losses.append(mag_loss)

        sc_loss = sum(sc_losses) / len(sc_losses)
        mag_loss = sum(mag_losses) / len(mag_losses)

        return sc_loss, mag_loss


def load_wav(wav_path, sr=24000):
    audio = librosa.core.load(wav_path, sr=sr)[0]
    return audio


def calculate_ms_stft_loss(args):
    input_files = glob.glob(f"{args.deg_dir}/*.wav")
    if len(input_files) < 1:
        raise RuntimeError(f"Found no wavs in {args.ref_dir}")

    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    stft_criterion = MultiResolutionSTFTLoss(
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_sizes=[600, 1200, 240],
    ).to(device)

    stft_loss_results = []

    for deg_wav_path in tqdm(input_files):
        ref_wav_path = os.path.join(args.ref_dir, os.path.basename(deg_wav_path))

        ref_wav = load_wav(ref_wav_path, sr=args.sample_rate)
        deg_wav = load_wav(deg_wav_path, sr=args.sample_rate)
        min_len = min(len(ref_wav), len(deg_wav))
        ref_wav = ref_wav[:min_len]
        deg_wav = deg_wav[:min_len]

        ref_wav = torch.from_numpy(ref_wav).view(1, ref_wav.shape[0]).to(device)
        deg_wav = torch.from_numpy(deg_wav).view(1, deg_wav.shape[0]).to(device)

        sc_loss, mag_loss = stft_criterion(deg_wav, ref_wav)
        full_stft_loss = sc_loss + mag_loss
        stft_loss_results.append(full_stft_loss.cpu())

    return np.mean(stft_loss_results)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Compute multi-resolution STFT Loss.")

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
    parser.add_argument(
        '-s',
        '--sample_rate',
        type=int,
        default=16000,
        help="Sampling rate."
    )
    parser.add_argument(
        '--use_gpu',
        action='store_true',
        help="set to use cpu."
    )

    args = parser.parse_args()

    ms_stft_loss_result = calculate_ms_stft_loss(args)
    print(f"MS-STFT-Loss: {ms_stft_loss_result}")
