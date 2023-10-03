##########
# world
##########
import librosa
import numpy as np
import copy

import torch

gamma = 0
mcepInput = 3  # 0 for dB, 3 for magnitude
alpha = 0.45
en_floor = 10 ** (-80 / 20)
FFT_SIZE = 2048


def code_harmonic(sp, order):
    import pysptk
    # get mcep
    mceps = np.apply_along_axis(pysptk.mcep, 1, sp, order - 1, alpha, itype=mcepInput, threshold=en_floor)

    # do fft and take real
    scale_mceps = copy.copy(mceps)
    scale_mceps[:, 0] *= 2
    scale_mceps[:, -1] *= 2
    mirror = np.hstack([scale_mceps[:, :-1], scale_mceps[:, -1:0:-1]])
    mfsc = np.fft.rfft(mirror).real

    return mfsc


def decode_harmonic(mfsc, fftlen=FFT_SIZE):
    import pysptk
    # get mcep back
    mceps_mirror = np.fft.irfft(mfsc)
    mceps_back = mceps_mirror[:, :60]
    mceps_back[:, 0] /= 2
    mceps_back[:, -1] /= 2

    # get sp
    spSm = np.exp(np.apply_along_axis(pysptk.mgc2sp, 1, mceps_back, alpha, gamma, fftlen=fftlen).real)

    return spSm


def to_lf0(f0):
    f0[f0 < 1.0e-5] = 1.0e-6
    lf0 = f0.log() if isinstance(f0, torch.Tensor) else np.log(f0)
    lf0[f0 < 1.0e-5] = - 1.0E+10
    return lf0


def to_f0(lf0):
    f0 = np.where(lf0 <= 0, 0.0, np.exp(lf0))
    return f0.flatten()


def formant_enhancement(coded_spectrogram, beta, fs):
    alpha_dict = {
        8000: 0.31,
        16000: 0.58,
        22050: 0.65,
        44100: 0.76,
        48000: 0.77
    }
    alpha = alpha_dict[fs]
    datad = np.zeros((coded_spectrogram.shape[1],))
    sp_dim = coded_spectrogram.shape[1]
    for i in range(coded_spectrogram.shape[0]):
        datad = mc2b(coded_spectrogram[i], datad, sp_dim - 1, alpha)
        datad[1] = datad[1] - alpha * beta * datad[2]
        for j in range(2, sp_dim):
            datad[j] *= 1 + beta
        coded_spectrogram[i] = b2mc(datad, coded_spectrogram[i], sp_dim - 1, alpha)
    return coded_spectrogram


def mc2b(mc, b, m, a):
    """
    Transform Mel Cepstrum to MLSA Digital Filter Coefficients

            void mc2b(mc, b, m, a)

            double *mc  : mel cepstral coefficients
            double *b   : MLSA digital filter coefficients
            int     m   : order of mel cepstrum
            double  a   : all-pass constant

        http://www.asel.udel.edu/icslp/cdrom/vol1/725/a725.pdf
        CELP coding system based on mel-generalized cepstral analysis
    :param mc:
    :param b:
    :param m:
    :param a:
    :return:
    """
    b[m] = mc[m]
    for i in range(1, m + 1):
        b[m - i] = mc[m - i] - a * b[m - i + 1]
    return b


def b2mc(b, mc, m, a):
    """
    Transform MLSA Digital Filter Coefficients to Mel Cepstrum

    void b2mc(b, mc, m, a)

    double *b  : MLSA digital filter coefficients
    double *mc : mel cepstral coefficients
    int    m   : order of mel cepstrum
    double a   : all-pass constant

    http://www.asel.udel.edu/icslp/cdrom/vol1/725/a725.pdf
    CELP coding system based on mel-generalized cepstral analysis
    :param b:
    :param mc:
    :param m:
    :param a:
    :return:
    """
    d = mc[m] = b[m]
    for i in range(1, m + 1):
        o = b[m - i] + a * d
        d = b[m - i]
        mc[m - i] = o
    return mc


f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)


def f0_to_coarse(f0):
    is_torch = isinstance(f0, torch.Tensor)
    f0_mel = 1127 * (1 + f0 / 700).log() if is_torch else 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = (f0_mel + 0.5).long() if is_torch else np.rint(f0_mel).astype(np.int)
    assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (f0_coarse.max(), f0_coarse.min(), f0.min(), f0.max())
    return f0_coarse


def norm_f0(f0, uv, hparams):
    is_torch = isinstance(f0, torch.Tensor)
    if hparams['pitch_norm'] == 'standard':
        f0 = (f0 - hparams['f0_mean']) / hparams['f0_std']
    if hparams['pitch_norm'] == 'log':
        f0 = torch.log2(f0 + 1e-8) if is_torch else np.log2(f0 + 1e-8)
    if uv is not None and hparams['use_uv']:
        f0[uv > 0] = 0
    return f0


def norm_interp_f0(f0, hparams):
    is_torch = isinstance(f0, torch.Tensor)
    if is_torch:
        device = f0.device
        f0 = f0.data.cpu().numpy()
    uv = f0 == 0
    f0 = norm_f0(f0, uv, hparams)
    if sum(uv) == len(f0):
        f0[uv] = 0
    elif sum(uv) > 0:
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    if is_torch:
        uv = torch.FloatTensor(uv)
        f0 = torch.FloatTensor(f0)
        f0 = f0.to(device)
        uv = uv.to(device)
    return f0, uv


def denorm_f0(f0, uv, hparams, pitch_padding=None, min=None, max=None):
    is_torch = isinstance(f0, torch.Tensor)
    if hparams['pitch_norm'] == 'standard':
        f0 = f0 * hparams['f0_std'] + hparams['f0_mean']
    if hparams['pitch_norm'] == 'log':
        f0 = 2 ** f0
    if min is None:
        min = 0
    if max is None:
        max = f0_max
    f0 = f0.clamp(min=min) if is_torch else np.clip(f0, min=min)
    f0 = f0.clamp(max=max) if is_torch else np.clip(f0, max=max)
    if uv is not None and hparams['use_uv']:
        f0[uv > 0] = 0
    if pitch_padding is not None:
        f0[pitch_padding] = 0
    return f0


def pitchfeats(wav, sampling_rate, fft_size, hop_size, win_length, fmin, fmax):
    pitches, magnitudes = librosa.piptrack(wav, sampling_rate,
                                           n_fft=fft_size, win_length=win_length, hop_length=hop_size,
                                           fmin=fmin, fmax=fmax)
    pitches = pitches.T
    magnitudes = magnitudes.T
    assert pitches.shape == magnitudes.shape

    pitches = [pitches[i][find_f0(magnitudes[i])] for i, _ in enumerate(pitches)]

    return np.asarray(pitches)


def find_f0(mags):
    tmp = 0
    mags = list(mags)
    for i, mag in enumerate(mags):
        if mag < tmp:
            # return i-1
            if tmp - mag > 2:
                # return i-1
                return mags.index(max(mags[0:i]))
            else:
                return 0
        else:
            tmp = mag
    return 0
