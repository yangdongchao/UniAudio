import subprocess
import librosa
import librosa.filters
import numpy as np
import torch
from scipy import signal
from scipy.io import wavfile
import torch.nn.functional as F


def save_wav(wav, path, sr, norm=False):
    if norm:
        wav = wav / np.abs(wav).max()
    wav *= 32767
    # proposed by @dsmiller
    wavfile.write(path, sr, wav.astype(np.int16))


def to_mp3(out_path):
    subprocess.check_call(
        f'ffmpeg -threads 1 -loglevel error -i "{out_path}.wav" -vn -ar 44100 -ac 1 -b:a 192k -y -hide_banner "{out_path}.mp3"',
        shell=True, stdin=subprocess.PIPE)
    subprocess.check_call(f'rm -f "{out_path}.wav"', shell=True)


def get_hop_size(hparams):
    hop_size = hparams['hop_size']
    if hop_size is None:
        assert hparams['frame_shift_ms'] is not None
        hop_size = int(hparams['frame_shift_ms'] / 1000 * hparams['audio_sample_rate'])
    return hop_size


###########################################################################################
def griffin_lim(S, hparams, angles=None):
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape)) if angles is None else angles
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles, hparams)
    for i in range(hparams['griffin_lim_iters']):
        angles = np.exp(1j * np.angle(_stft(y, hparams)))
        y = _istft(S_complex * angles, hparams)
    return y


def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav


def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav


def _stft(y, hparams):
    return librosa.stft(y=y, n_fft=hparams['fft_size'], hop_length=get_hop_size(hparams),
                        win_length=hparams['win_size'], pad_mode='constant')


def _istft(y, hparams):
    return librosa.istft(y, hop_length=get_hop_size(hparams), win_length=hparams['win_size'])



def librosa_pad_lr(x, fsize, fshift, pad_sides=1):
    '''compute right padding (final frame) or both sides padding (first and final frames)
    '''
    assert pad_sides in (1, 2)
    # return int(fsize // 2)
    pad = (x.shape[0] // fshift + 1) * fshift - x.shape[0]
    if pad_sides == 1:
        return 0, pad
    else:
        return pad // 2, pad // 2 + pad % 2


# Conversions
_mel_basis = None
_inv_mel_basis = None


def _linear_to_mel(spectogram, hparams):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(hparams)
    return np.dot(_mel_basis, spectogram)


def _mel_to_linear(mel_spectrogram, hparams):
    global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(_build_mel_basis(hparams))
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))


def _build_mel_basis(hparams):
    assert hparams['fmax'] <= hparams['audio_sample_rate'] // 2
    return librosa.filters.mel(hparams['audio_sample_rate'], hparams['fft_size'], n_mels=hparams['audio_num_mel_bins'],
                               fmin=hparams['fmin'], fmax=hparams['fmax'])


def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def db_to_amp(x):
    return 10.0 ** (x * 0.05)


def normalize(S, hparams):
    return (S - hparams['min_level_db']) / -hparams['min_level_db']


def denormalize(D, hparams):
    return (D * -hparams['min_level_db']) + hparams['min_level_db']


#### torch audio


def istft(amp, ang, hparams, pad=False, window=None):
    spec = amp * torch.exp(1j * ang)
    spec_r = spec.real
    spec_i = spec.imag
    spec = torch.stack([spec_r, spec_i], -1)
    if window is None:
        window = torch.hann_window(hparams['win_size']).to(amp.device)
    if pad:
        spec = F.pad(spec, [0, 0, 0, 1], mode='reflect')
    wav = torch.istft(spec, hparams['fft_size'], hparams['hop_size'], hparams['win_size'])
    return wav


def griffin_lim_torch(amp, ang, hparams, n_iters=30):
    """

    Examples:
    >>> x_stft = librosa.stft(wav, n_fft=fft_size, hop_length=hop_size, win_length=win_length, pad_mode="constant")
    >>> x_stft = x_stft[None, ...]
    >>> amp = np.abs(x_stft)
    >>> angle_init = np.exp(2j * np.pi * np.random.rand(*x_stft.shape))
    >>> amp = torch.FloatTensor(amp)
    >>> wav = griffin_lim_torch(amp, angle_init, hparams)

    :param amp: [B, n_fft, T]
    :param ang: [B, n_fft, T]
    :return: [B, T_wav]
    """
    window = torch.hann_window(hparams['win_size']).to(amp.device)
    y = istft(amp, ang, hparams, window=window)
    for i in range(n_iters):
        x_stft = torch.stft(y, hparams['fft_size'], hparams['hop_size'], hparams['win_size'], window)
        x_stft = x_stft[..., 0] + 1j * x_stft[..., 1]
        ang = torch.angle(x_stft)
        y = istft(amp, ang, hparams, window=window)
    return y
