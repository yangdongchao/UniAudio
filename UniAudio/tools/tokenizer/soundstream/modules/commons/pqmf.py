# -*- coding: utf-8 -*-

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Pseudo QMF modules."""

import numpy as np
import torch
import torch.nn.functional as F

from scipy.signal import kaiser


def design_prototype_filter(taps=62, cutoff_ratio=0.142, beta=9.0):
    """Design prototype filter for PQMF.
    This method is based on `A Kaiser window approach for the design of prototype
    filters of cosine modulated filterbanks`_.
    Args:
        taps (int): The number of filter taps.
        cutoff_ratio (float): Cut-off frequency ratio.
        beta (float): Beta coefficient for kaiser window.
    Returns:
        ndarray: Impluse response of prototype filter (taps + 1,).
    .. _`A Kaiser window approach for the design of prototype filters of cosine modulated filterbanks`:
        https://ieeexplore.ieee.org/abstract/document/681427
    """
    # check the arguments are valid
    assert taps % 2 == 0, "The number of taps mush be even number."
    assert 0.0 < cutoff_ratio < 1.0, "Cutoff ratio must be > 0.0 and < 1.0."

    # make initial filter
    omega_c = np.pi * cutoff_ratio
    with np.errstate(invalid='ignore'):
        h_i = np.sin(omega_c * (np.arange(taps + 1) - 0.5 * taps)) \
            / (np.pi * (np.arange(taps + 1) - 0.5 * taps))
    h_i[taps // 2] = np.cos(0) * cutoff_ratio  # fix nan due to indeterminate form

    # apply kaiser window
    w = kaiser(taps + 1, beta)
    h = h_i * w

    return h


class PQMF(torch.nn.Module):
    """PQMF module.
    This module is based on `Near-perfect-reconstruction pseudo-QMF banks`_.
    .. _`Near-perfect-reconstruction pseudo-QMF banks`:
        https://ieeexplore.ieee.org/document/258122
    """

    def __init__(self, subbands=4, taps=62, cutoff_ratio=0.142, beta=9.0):
        """Initilize PQMF module.
        The cutoff_ratio and beta parameters are optimized for #subbands = 4.
        See dicussion in https://github.com/kan-bayashi/ParallelWaveGAN/issues/195.
        Args:
            subbands (int): The number of subbands.
            taps (int): The number of filter taps.
            cutoff_ratio (float): Cut-off frequency ratio.
            beta (float): Beta coefficient for kaiser window.
        """
        super(PQMF, self).__init__()

        if subbands == 8:
           cutoff_ratio = 0.07949452
        elif subbands == 6:
           cutoff_ratio = 0.10032791
        elif subbands == 4:
            cutoff_ratio = 0.13
        elif subbands == 2:
            cutoff_ratio = 0.25

        # build analysis & synthesis filter coefficients
        h_proto = design_prototype_filter(taps, cutoff_ratio, beta)
        h_analysis = np.zeros((subbands, len(h_proto)))
        h_synthesis = np.zeros((subbands, len(h_proto)))
        for k in range(subbands):
            h_analysis[k] = 2 * h_proto * np.cos(
                (2 * k + 1) * (np.pi / (2 * subbands)) *
                (np.arange(taps + 1) - (taps / 2)) +
                (-1) ** k * np.pi / 4)
            h_synthesis[k] = 2 * h_proto * np.cos(
                (2 * k + 1) * (np.pi / (2 * subbands)) *
                (np.arange(taps + 1) - (taps / 2)) -
                (-1) ** k * np.pi / 4)

        # convert to tensor
        analysis_filter = torch.from_numpy(h_analysis).float().unsqueeze(1)
        synthesis_filter = torch.from_numpy(h_synthesis).float().unsqueeze(0)

        # register coefficients as beffer
        self.register_buffer("analysis_filter", analysis_filter)
        self.register_buffer("synthesis_filter", synthesis_filter)

        # filter for downsampling & upsampling
        updown_filter = torch.zeros((subbands, subbands, subbands)).float()
        for k in range(subbands):
            updown_filter[k, k, 0] = 1.0
        self.register_buffer("updown_filter", updown_filter)
        self.subbands = subbands

        # keep padding info
        self.pad_fn = torch.nn.ConstantPad1d(taps // 2, 0.0)

    def analysis(self, x):
        """Analysis with PQMF.
        Args:
            x (Tensor): Input tensor (B, 1, T).
        Returns:
            Tensor: Output tensor (B, subbands, T // subbands).
        """
        x = F.conv1d(self.pad_fn(x), self.analysis_filter)
        return F.conv1d(x, self.updown_filter, stride=self.subbands)

    def synthesis(self, x):
        """Synthesis with PQMF.
        Args:
            x (Tensor): Input tensor (B, subbands, T // subbands).
        Returns:
            Tensor: Output tensor (B, 1, T).
        """
        # NOTE(kan-bayashi): Power will be dreased so here multipy by # subbands.
        #   Not sure this is the correct way, it is better to check again.
        # TODO(kan-bayashi): Understand the reconstruction procedure
        x = F.conv_transpose1d(x, self.updown_filter * self.subbands, stride=self.subbands)
        return F.conv1d(self.pad_fn(x), self.synthesis_filter)


def _objective(cutoff_ratio):
    h_proto = design_prototype_filter(num_taps, cutoff_ratio, beta)
    conv_h_proto = np.convolve(h_proto, h_proto[::-1], mode='full')
    length_conv_h = conv_h_proto.shape[0]
    half_length = length_conv_h // 2

    check_steps = np.arange((half_length) // (2 * num_subbands)) * 2 * num_subbands
    _phi_new = conv_h_proto[half_length:][check_steps]
    phi_new = np.abs(_phi_new[1:]).max()
    # Since phi_new is not convex, This value should also be considered.
    diff_zero_coef = np.abs(_phi_new[0] - 1 / (2 * num_subbands))

    return phi_new + diff_zero_coef

if __name__ == "__main__":
   model = PQMF(4)
   import numpy as np
   import scipy.optimize as optimize

   x = np.load('data/train/audio/010000.npy')
   x = torch.FloatTensor(x).unsqueeze(0).unsqueeze(0)
   out = model.analysis(x)
   print(out.shape)
   x_hat = model.synthesis(out)
   loss = torch.nn.functional.mse_loss(
    x[..., :x_hat.shape[-1]],
    x_hat[..., :x_hat.shape[-1]],
    reduction="sum"
   )
   print(loss)
   from scipy.io.wavfile import write
   audio = x_hat.squeeze().numpy()
   write('a.wav', 24000, audio)

   model = PQMF(6)
   out = model.analysis(x)
   print(out.shape)
   x_hat = model.synthesis(out)
   loss = torch.nn.functional.mse_loss(
    x[..., :x_hat.shape[-1]],
    x_hat[..., :x_hat.shape[-1]],
    reduction="sum"
   )
   print(loss)
   audio = x_hat.squeeze().numpy()
   write('b.wav', 24000, audio)

   num_subbands = 6
   num_taps = 62
   beta = 9.0

   ret = optimize.minimize(_objective, np.array([0.01]),
                        bounds=optimize.Bounds(0.01, 0.99))
   opt_cutoff_ratio = ret.x[0]
   print(f"optimized cutoff ratio = {opt_cutoff_ratio:.08f}")
