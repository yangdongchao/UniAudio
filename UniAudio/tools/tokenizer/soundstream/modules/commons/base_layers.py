from typing import Optional, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
from torch.autograd import Function

from .ops import get_padding


class Mish(nn.Module):
    """Mish activation function.

    This is introduced by
    `Mish: A Self Regularized Non-Monotonic Activation Function`.

    """
    def __init__(self):
        super(Mish, self).__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class LinearNorm(nn.Module):
    """ LinearNorm Projection 
    
    A wrapper of torch.nn.Linear layer.

    """

    def __init__(
        self,
        in_features: int, 
        out_features: int, 
        bias: bool = True,
        spectral_norm: bool = False,
        w_init_gain: str = 'linear',
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        
        if spectral_norm:
            self.linear = nn.utils.spectral_norm(self.linear)
        else:
            nn.init.xavier_uniform_(
                self.linear.weight, gain=nn.init.calculate_gain(w_init_gain))
            if bias:
                nn.init.constant_(self.linear.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return x


class ConvNorm(nn.Module):
    """ Conv1d layer
    
    A wrapper of torch.nn.Conv1d layer.

    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        w_init_gain: Optional[str] = None,
        spectral_norm: Optional[bool] = False,
        channel_last: Optional[bool] = False,
    ):
        super().__init__()
        self.channel_last = channel_last

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        if spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)
        else:
            if w_init_gain is not None:
                torch.nn.init.xavier_uniform_(
                    self.conv.weight,
                    gain=torch.nn.init.calculate_gain(w_init_gain),
                )
        self.channel_last = channel_last

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel_last:
            x = x.transpose(1, 2)
            return self.conv(x).transpose(1, 2)
        out = self.conv(x)
        return out


class EmbeddingTable(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, **kwargs):
        super().__init__(
            num_embeddings, embedding_dim, padding_idx=padding_idx, **kwargs)
        
        self.embed_scale = math.sqrt(embedding_dim)

        nn.init.normal_(self.weight, 0.0, embedding_dim ** -0.5)
        if padding_idx is not None:
            nn.init.constant_(self.weight[padding_idx], 0)

        self.output_dim = embedding_dim

    def forward(self, x):
        x = super().forward(x)
        return x * self.embed_scale


class Conv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding_mode: str = 'zeros',
        bias: bool = True,
        padding = None,
        causal: bool = False,
        w_init_gain = None,
    ):
        self.causal = causal
        if padding is None:
            if causal:
                padding = 0
                self.left_padding = dilation * (kernel_size - 1)
            else:
                padding = get_padding(kernel_size, dilation)
        super(Conv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
            bias=bias)
        if w_init_gain is not None:
            torch.nn.init.xavier_uniform_(
                self.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.causal:
            x = F.pad(x.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)

        return super(Conv1d, self).forward(x)


class ConvTranspose1d(nn.ConvTranspose1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding=None,
        padding_mode: str = 'zeros',
        causal: bool = False,
    ):
        if padding is None:
            padding = 0 if causal else (kernel_size - stride) // 2
        if causal:
            assert padding == 0, "padding is not allowed in causal ConvTranspose1d."
            assert kernel_size == 2 * stride, \
                "kernel_size must be equal to 2*stride is not allowed in causal ConvTranspose1d."
        super(ConvTranspose1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
        )
        self.causal = causal
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super(ConvTranspose1d, self).forward(x)
        if self.causal:
            x = x[:, :, :-self.stride]
        return x


class UpsampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        causal: bool = False,
        activation=nn.PReLU(),
        use_weight_norm: bool = True,
        repeat: bool = False,
    ):
        super(UpsampleLayer, self).__init__()
        self.repeat = repeat
        self.stride = stride
        self.activation = activation
        self.use_weight_norm = use_weight_norm
        if repeat:
            self.layer = Conv1d(
                in_channels, out_channels, kernel_size, causal=causal)
        else:
            self.layer = ConvTranspose1d(
                in_channels, out_channels, kernel_size, stride=stride, causal=causal)
        if use_weight_norm:
            self.layer = weight_norm(self.layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer(x)
        x = self.activation(x) if self.activation is not None else x
        if self.repeat:
            x = torch.transpose(x, 1, 2)
            B, T, C = x.size()
            x = x.repeat(1, 1, self.stride).view(B, -1, C)
            x = torch.transpose(x, 1, 2)
        return x

    def remove_weight_norm(self):
        if self.use_weight_norm:
            remove_weight_norm(self.layer)


class DownsampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        causal: bool = False,
        activation=nn.PReLU(),
        use_weight_norm: bool = True,
        pooling: bool = False,
    ):
        super(DownsampleLayer, self).__init__()
        self.pooling = pooling
        self.stride = stride
        self.activation = activation
        self.use_weight_norm = use_weight_norm
        if pooling:
            self.layer = Conv1d(
                in_channels, out_channels, kernel_size, causal=causal)
            self.pooling = nn.AvgPool1d(kernel_size=stride)
        else:
            self.layer = Conv1d(
                in_channels, out_channels, kernel_size, stride=stride, causal=causal)
        if use_weight_norm:
            self.layer = weight_norm(self.layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer(x)
        x = self.activation(x) if self.activation is not None else x
        if self.pooling:
            x = self.pooling(x)
        return x

    def remove_weight_norm(self):
        if self.use_weight_norm:
            remove_weight_norm(self.layer)


class StyleAdaptiveLayerNorm(nn.Module):
    """Style-adaptive layer normalization module.

    This module is introduced in `Meta-StyleSpeech : Multi-Speaker Adaptive
    Text-to-Speech Generation`, which is similar to the conditional layer normalization
    operation introduced in `Adaspeech: adaptive text to speech for custom voice`.
    If layer_norm_input is set to be False, the operation is the same to Feature-wise
    Linear Modulation (FiLM) proposed in `FiLM: Visual Reasoning with a General Conditioning Layer`.

    Args:
        in_channel (int): The dimension of input channels, often equal to d_model in
            transformer and conformer models.
        layer_norm_input (bool): whether to apply layer normalization on input feature.
            Default: `True`.

    """
    def __init__(self, in_channel: int, layer_norm_input: bool = True):
        super().__init__()
        self.layer_norm_input = layer_norm_input

        self.in_channel = in_channel
        if layer_norm_input:
            self.norm = nn.LayerNorm(in_channel, elementwise_affine=False)

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Forward propagation.
        
        Args:
            x (Tensor): Batch of input features (B, T, C).
            gamma (Tensor): Scale features (B, C).
            beta (Tensor): Shift features (B, C).

        Returns:
            Tensor: Style-adaptive layer-normed features.
        
        """
        if self.layer_norm_input:
            x = self.norm(x)
        out = gamma.unsqueeze(1) * x + beta.unsqueeze(1)

        return out


class PreNet(nn.Module):
    """Tacotron2 decoder prenet, where dropout (default rate = 0.5) is open during both
    training and inference.
    """
    def __init__(
        self,
        in_dim: int,
        sizes: List[int],
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.dropout_rate = dropout_rate
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            # Use dropout in both training and testing phases
            x = F.dropout(F.relu(linear(x)), p=self.dropout_rate, training=True)
        return x


class ConvPrenet(nn.Module):
    r""" Convolution-based Prenet. Residual connection is used.

    Computation flow:

    input -> conv1 -> act_fn + dropout -> conv2 -> act_fn + dropout -> fc -> + -> output
          \                                                                 /
           ------------------------------>----------------------------------

    """
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        dropout: float,
        activation: str = 'mish',
        kernel_size: int = 3,
    ):
        super(ConvPrenet, self).__init__()

        if activation == 'mish':
            act_class = Mish
        elif activation == 'relu':
            act_class = nn.ReLU
        else:
            raise ValueError(f'Activation function {activation} is not in ["mish", "relu"].')

        self.convs = nn.Sequential(
            ConvNorm(hidden_dim, hidden_dim, kernel_size=kernel_size),
            act_class(),
            nn.Dropout(dropout),
            ConvNorm(hidden_dim, hidden_dim, kernel_size=kernel_size),
            act_class(),
            nn.Dropout(dropout),
        )
        self.fc = LinearNorm(hidden_dim, out_dim)

    def forward(
        self,
        input: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """ Forward propagation.

        Args:
            input (tensor): input feature with shape [B, T, C].
            mask (optional(tensor)): mask with ones in padding parts, [B, T]

        Returns:
            output (tensor): output features with shape [B, T, C]

        """
        residual = input
        # convs
        output = input.transpose(1,2)
        output = self.convs(output)
        output = output.transpose(1,2)
        # fc & residual
        output = self.fc(output) + residual

        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0)

        return output


class PositionEncoding(nn.Module):

    def __init__(self, n_dim, max_position=10000):
        super(PositionEncoding, self).__init__()
        self.register_buffer('position_embs', self.init_sinusoid_table(max_position, n_dim))

    def init_sinusoid_table(self, max_position, n_dim):
        emb_tabels = torch.zeros(max_position, n_dim)
        pos = torch.arange(0, max_position)
        pos = pos.float().unsqueeze(dim=1)
        _2i = torch.arange(0, n_dim, step=2).float()
        emb_tabels[:, 0::2] = torch.sin(pos / (10000 ** (_2i / n_dim)))
        emb_tabels[:, 1::2] = torch.cos(pos / (10000 ** (_2i / n_dim)))
        return emb_tabels.unsqueeze(0)

    def forward(self, x):
        return x + self.position_embs[:, :x.size(1)].clone().detach()


class Conv1dGLU(nn.Module):
    """ Causal gated CNN module.
    Conv1d + GLU(Gated Linear Unit) with residual connection.
    For GLU refer to https://arxiv.org/abs/1612.08083 paper.

    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float,
        causal: bool,
        scale_weight: float = 1.0 / math.sqrt(2.0),
    ):
        super().__init__()
        self.out_channels = out_channels
        self.scale_weight = scale_weight
        self.conv1 = Conv1d(
            in_channels=in_channels,
            out_channels=2 * out_channels,
            kernel_size=kernel_size,
            causal=causal,
        )
        self.dropout = nn.Dropout(dropout)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (tensor): input feature with shape [B, C, T].
        """
        residual = x
        x = self.conv1(x)
        x1, x2 = torch.split(x, split_size_or_sections=self.out_channels, dim=1)
        x = x1 * torch.sigmoid(x2)
        x = residual + self.dropout(x)
        x = x * self.scale_weight
        return x


class ReverseLayerF(Function):
    """ https://github.com/fungtion/DANN/blob/master/models/functions.py """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
