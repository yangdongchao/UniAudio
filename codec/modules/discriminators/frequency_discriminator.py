import torch
import torch.nn as nn
from omegaconf import DictConfig

from modules.commons.torch_stft import TorchSTFT


class MultiFrequencyDiscriminator(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.stfts = nn.ModuleList([
            TorchSTFT(
                fft_size=x * 4,
                hop_size=x,
                win_size=x * 4,
                normalized=True, # returns the normalized STFT results, i.e., multiplied by frame_length^{-0.5}
                domain=config.domain,
                mel_scale=config.mel_scale,
                sample_rate=config.sample_rate,
            ) for x in config.hop_lengths
        ])

        self.domain = config.domain
        if self.domain == 'double':
            self.discriminators = nn.ModuleList([
                FrequenceDiscriminator(2, c)
                for x, c in zip(config.hop_lengths, config.hidden_channels)])
        else:
            self.discriminators = nn.ModuleList([
                FrequenceDiscriminator(1, c)
                for x, c in zip(config.hop_lengths, config.hidden_channels)])

    def forward(self, y, y_hat, **kwargs):
        if y.ndim == 3:
            y = y.view(-1, y.shape[-1])
        
        if y_hat.ndim == 3:
            y_hat = y_hat.view(-1, y_hat.shape[-1])

        real_outputs = []
        fake_outputs = []
        real_feature_maps = []
        fake_feature_maps = []

        for stft, layer in zip(self.stfts, self.discriminators):
            mag, phase = stft.transform(y.squeeze(1))
            fake_mag, fake_phase = stft.transform(y_hat.squeeze(1))
            if self.domain == 'double':
                mag = torch.stack(torch.chunk(mag, 2, dim=1), dim=1)
                fake_mag = torch.stack(torch.chunk(fake_mag, 2, dim=1), dim=1)
            else:
                mag = mag.unsqueeze(1)
                fake_mag = fake_mag.unsqueeze(1)

            real_out, real_feat_map = layer(mag)
            fake_out, fake_feat_map = layer(fake_mag)
            real_outputs.append(real_out)
            fake_outputs.append(fake_out)
            real_feature_maps.append(real_feat_map)
            fake_feature_maps.append(fake_feat_map)

        return real_outputs, fake_outputs, real_feature_maps, fake_feature_maps


class FrequenceDiscriminator(nn.Module):
    def __init__(self, in_channels, hidden_channels=512):
        super(FrequenceDiscriminator, self).__init__()

        self.discriminator = nn.ModuleList()
        self.discriminator += [
            nn.Sequential(
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.utils.weight_norm(nn.Conv2d(
                    in_channels, hidden_channels // 32,
                    kernel_size=(3, 3), stride=(1, 1)))
            ),
            nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.utils.weight_norm(nn.Conv2d(
                    hidden_channels // 32, hidden_channels // 16,
                    kernel_size=(3, 3), stride=(2, 2)))
            ),
            nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.utils.weight_norm(nn.Conv2d(
                    hidden_channels // 16, hidden_channels // 8,
                    kernel_size=(3, 3), stride=(1, 1)))
            ),
            nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.utils.weight_norm(nn.Conv2d(
                    hidden_channels // 8, hidden_channels // 4,
                    kernel_size=(3, 3), stride=(2, 2)))
            ),
            nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.utils.weight_norm(nn.Conv2d(
                    hidden_channels // 4, hidden_channels // 2,
                    kernel_size=(3, 3), stride=(1, 1)))
            ),
            nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.utils.weight_norm(nn.Conv2d(
                    hidden_channels // 2, hidden_channels,
                    kernel_size=(3, 3), stride=(2, 2)))
            ),
            nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.ReflectionPad2d((1, 1, 1, 1)),
                nn.utils.weight_norm(nn.Conv2d(
                    hidden_channels, 1,
                    kernel_size=(3, 3), stride=(1, 1)))
            )
        ]

    def forward(self, x):
        hiddens = []
        for layer in self.discriminator:
            x = layer(x)
            hiddens.append(x)
        return x, hiddens[:-1]
