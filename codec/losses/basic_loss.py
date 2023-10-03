import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureMatchLoss(nn.Module):
    def __init__(self):
        super(FeatureMatchLoss, self).__init__()

    def forward(self, real_features, fake_features):
        loss = 0
        num_items = 0
        for (fake_feature, real_feature) in zip(fake_features, real_features):
            if isinstance(fake_feature, list):
                for (_fake_feature, _real_feature) in zip(fake_feature, real_feature):
                    loss = loss + F.l1_loss(_fake_feature.float(), _real_feature.float().detach())
                    num_items += 1
            else:
                loss = loss + F.l1_loss(fake_feature.float(), real_feature.float().detach())
                num_items += 1
        loss /= num_items
        return loss


class LeastDLoss(nn.Module):
    def __init__(self):
        super(LeastDLoss, self).__init__()

    def forward(self, disc_outputs):
        loss = 0
        for dg in disc_outputs:
            dg = dg.float()
            l = torch.mean((1-dg)**2)
            loss += l
        return loss


class MSEDLoss(nn.Module):
    def __init__(self):
        super(MSEDLoss, self).__init__()
        self.loss_func = nn.MSELoss()

    def forward(self, score_fake, score_real):
        loss_real = self.loss_func(score_real, score_real.new_ones(score_real.shape))
        loss_fake = self.loss_func(score_fake, score_fake.new_zeros(score_fake.shape))
        loss_d = loss_real + loss_fake
        return loss_d, loss_real, loss_fake


class HingeDLoss(nn.Module):
    def __init__(self):
        super(HingeDLoss, self).__init__()

    def forward(self, score_fake, score_real):
        loss_real = torch.mean(F.relu(1. - score_real))
        loss_fake = torch.mean(F.relu(1. + score_fake))
        loss_d = loss_real + loss_fake
        return loss_d, loss_real, loss_fake


class MSEGLoss(nn.Module):
    def __init__(self):
        super(MSEGLoss, self).__init__()

    def forward(self, scores):
        loss_fake = 0
        num_items = 0
        if isinstance(scores, list):
            for score in scores:
                loss_fake = loss_fake + F.mse_loss(score, score.new_ones(score.shape))
                num_items += 1
        else:
            loss_fake = F.mse_loss(scores, scores.new_ones(scores.shape))
            num_items += 1
        return loss_fake / num_items


class HingeGLoss(nn.Module):
    def __init__(self):
        super(HingeGLoss, self).__init__()

    def forward(self, score_real):
        loss_fake = torch.mean(F.relu(1. - score_real))
        return loss_fake


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
