import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

LINE_COLORS = ['w', 'r', 'y', 'cyan', 'm', 'b', 'lime']


def spec_to_figure(spec, vmin=None, vmax=None, title=''):
    if isinstance(spec, torch.Tensor):
        spec = spec.cpu().numpy()
    fig = plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
    return fig


def spec_f0_to_figure(spec, f0s, figsize=None):
    max_y = spec.shape[1]
    if isinstance(spec, torch.Tensor):
        spec = spec.detach().cpu().numpy()
        f0s = {k: f0.detach().cpu().numpy() for k, f0 in f0s.items()}
    f0s = {k: f0 / 10 for k, f0 in f0s.items()}
    fig = plt.figure(figsize=(12, 6) if figsize is None else figsize)
    plt.pcolor(spec.T)
    for i, (k, f0) in enumerate(f0s.items()):
        plt.plot(f0.clip(0, max_y), label=k, c=LINE_COLORS[i], linewidth=1, alpha=0.8)
    plt.legend()
    return fig


def dur_to_figure(dur_gt, dur_pred, txt, mels=None, vmin=-5.5, vmax=1):
    dur_gt = dur_gt.cpu().numpy()
    dur_pred = dur_pred.cpu().numpy()
    dur_gt = np.cumsum(dur_gt).astype(int)
    dur_pred = np.cumsum(dur_pred).astype(int)
    fig = plt.figure(figsize=(12, 6))
    for i in range(len(dur_gt)):
        shift = (i % 8) + 1
        plt.text(dur_gt[i], shift * 4, txt[i])
        plt.text(dur_pred[i], 40 + shift * 4, txt[i])
        plt.vlines(dur_gt[i], 0, 40, colors='b')  # blue is gt
        plt.vlines(dur_pred[i], 40, 80, colors='r')  # red is pred
    plt.xlim(0, max(dur_gt[-1], dur_pred[-1]))
    if mels is not None:
        mels = mels.cpu().numpy()
        plt.pcolor(mels.T, vmin=vmin, vmax=vmax)
    return fig


def f0_to_figure(f0_gt, f0_cwt=None, f0_pred=None):
    fig = plt.figure(figsize=(12, 8))
    f0_gt = f0_gt.cpu().numpy()
    plt.plot(f0_gt, color='r', label='gt')
    if f0_cwt is not None:
        f0_cwt = f0_cwt.cpu().numpy()
        plt.plot(f0_cwt, color='b', label='cwt')
    if f0_pred is not None:
        f0_pred = f0_pred.cpu().numpy()
        plt.plot(f0_pred, color='green', label='pred')
    plt.legend()
    return fig
