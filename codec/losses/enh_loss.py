import torch
import torch.nn as nn
import torchaudio


def freq_MAE(estimation, target, win=2048, stride=512, srs=None, sudo_sr=None):
    est_spec = torch.stft(
        estimation.view(-1, estimation.shape[-1]),
        n_fft=win,
        hop_length=stride, 
        window=torch.hann_window(win).to(estimation.device).float(),
        return_complex=True,
    )
    est_target = torch.stft(
        target.view(-1, target.shape[-1]),
        n_fft=win,
        hop_length=stride, 
        window=torch.hann_window(win).to(estimation.device).float(),
        return_complex=True,
    )

    if srs is None:
        return (est_spec.real - est_target.real).abs().mean() + (est_spec.imag - est_target.imag).abs().mean()
    else:
        loss = 0
        for i, sr in enumerate(srs):
            max_freq = int(est_spec.shape[-2] * sr / sudo_sr) + 1 
            loss += (est_spec[i][:max_freq].real - est_target[i][:max_freq].real).abs().mean() \
                + (est_spec[i][:max_freq].imag - est_target[i][:max_freq].imag).abs().mean()   
        loss = loss / len(srs)
        # import pdb; pdb.set_trace()
        return loss


def wav_MAE(ests, refs):
    return torch.mean(torch.abs(ests - refs))


def sisnr(x,s,eps=1e-8):
    '''
    Calculate si-snr loss
    x: Bsz*T ests
    s: Bsz*T refs
    '''

    x, s = x.view(-1, x.shape[-1]), s.view(-1, s.shape[-1])
    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)
                
    if x.shape != s.shape:
        raise RuntimeError(
            "Dimension mismatch when calculate si-snr, {} vs {}".format(x.shape, s.shape)
        )
    x_zm = x - torch.mean(x, dim=-1,keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(x_zm*s_zm, dim=-1, keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    
    return - 20*torch.log10(eps+l2norm(t)/(l2norm(x_zm-t) + eps)).mean()


def snr(x,s,eps=1e-8):
    '''
    Calculate si-snr loss
    x: Bsz*T ests
    s: Bsz*T refs
    '''
    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)
        
    if x.shape != s.shape:
        raise RuntimeError(
        "Dimension mismatch when calculate si-snr, {} vs {}".format(x.shape, s.shape)
        )
    x_zm = x - torch.mean(x, dim=-1,keepdim=True)
    s_zm = s - torch.mean(s, dim=-1,keepdim=True)
    return - 20*torch.log10(l2norm(s_zm)/(l2norm(x_zm-s_zm) + eps) + eps).mean()


def mel_MAE(ests, refs, sr = 48000, n_fft = 2048, hop_length = 512, n_mels = 80):
    compute_Melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate = sr, n_fft = n_fft, hop_length = hop_length, n_mels = n_mels)

    ests_melspec = compute_Melspec(ests)
    refs_melspec = compute_Melspec(refs)

    return (ests_melspec - refs_melspec).abs().mean()


# adapted from ENHANCE-PASS
class BasicEnhancementLoss(nn.Module):
    """
    Config:
        sr: sample_rate
        loss_type: List[str]

    """
    def __init__(self, config):
        super(BasicEnhancementLoss, self).__init__()
        self.sr = config.sr
        self.loss_type = config.loss_type
        self.win = config.win
        self.stride = config.stride

        loss_weight = config.loss_weight
        if loss_weight == None:
            self.loss_weight = [1.0] * len(self.loss_type)
        else:
            self.loss_weight = loss_weight

        if 'mel_MAE' in self.loss_type:
            self.compute_Melspec = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sr, n_fft=self.win, hop_length=self.stride, n_mels=80)
            # self.compute_Melspec.to(device)

    def mel_MAE(self, ests, refs):
        ests_melspec = self.compute_Melspec(ests)
        refs_melspec = self.compute_Melspec(refs)

        return (ests_melspec - refs_melspec).abs().mean()

    def __call__(self, ests, refs, wav_lens=None, srs=None):
        loss_dic = {}
        loss = 0
        for i, item in enumerate(self.loss_type):
            # import pdb; pdb.set_trace()
            if item == 'freq_MAE':
                loss_dic[item] = eval(item)(
                    ests, refs, win=self.win, stride=self.stride, srs=srs, sudo_sr=self.sr)
            elif item == 'mel_MAE':
                loss_dic[item] = self.mel_MAE(ests, refs)
                # import pdb; pdb.set_trace()
            else:
                # wave MAE
                loss_dic[item] = eval(item)(ests, refs)
            if self.loss_weight[i] > 0:
                loss = self.loss_weight[i] * loss_dic[item] + loss

        return loss, loss_dic
