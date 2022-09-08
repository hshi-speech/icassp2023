# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez
#       : hao shi (modified)

import math
import time

import torch as th
from torch import nn
from torch.nn import functional as F

from .resample import downsample2, upsample2
from .utils import capture_init
import numpy as np


class CausalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(2, 3),
            stride=(1, 2),
            padding=(1, 0)
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.activation = nn.ELU()

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]
        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[:, :, :-1, :]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        return x


class CausalTransConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_last=False, output_padding=(0, 0)):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(2, 3),
            stride=(1, 2),
            output_padding=output_padding
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        if is_last:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.ELU()

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, F, T]
        Returns:
            [B, C, F, T]
        """
        x = self.conv(x)
        x = x[:, :, :-1, :]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        return x


class spec_encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0), dilation=(1, 1))
        self.activation = nn.ELU()
        self.norm = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.activation(self.norm(self.conv2d(x)))
        return x


class spec_decoder(nn.Module):
    def __init__(self, in_channels, out_channels, paddings=False, is_last=False):
        super().__init__()
        if paddings:
            self.convtrans2d = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 4), stride=(1, 2), padding=(1, 0), dilation=(1, 1))
        else:
            self.convtrans2d = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 2), padding=(1, 0), dilation=(1, 1))
        if is_last:
            self.activation = nn.ReLU()
        else:
            self.activation = nn.ELU()
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        
    def forward(self, x):
        x = self.activation(self.norm(self.convtrans2d(x)))
        return x


class spec_extraction_encoder(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, s=1, p=0):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(k, 3), stride=(s, 1), padding=(p, 1), dilation=(1, 1))
        self.activation = nn.ReLU()
        self.norm = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.activation(self.norm(self.conv2d(x)))
        return x


class BLSTM(nn.Module):
    def __init__(self, dim, layers=2, bi=True):
        super().__init__()
        klass = nn.LSTM
        self.lstm = klass(bidirectional=bi, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = None
        if bi:
            self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x, hidden=None):
        # self.lstm.flatten_parameters()
        x, hidden = self.lstm(x, hidden)
        if self.linear:
            x = self.linear(x)
        return x, hidden


def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)


class Demucs(nn.Module):
    """
    Demucs speech enhancement model.
    Args:
        - chin (int): number of input channels.
        - chout (int): number of output channels.
        - hidden (int): number of initial hidden channels.
        - depth (int): number of layers.
        - kernel_size (int): kernel size for each layer.
        - stride (int): stride for each layer.
        - causal (bool): if false, uses BiLSTM instead of LSTM.
        - resample (int): amount of resampling to apply to the input/output.
            Can be one of 1, 2 or 4.
        - growth (float): number of channels is multiplied by this for every layer.
        - max_hidden (int): maximum number of channels. Can be useful to
            control the size/speed of the model.
        - normalize (bool): if true, normalize the input.
        - glu (bool): if true uses GLU instead of ReLU in 1x1 convolutions.
        - rescale (float): controls custom weight initialization.
            See https://arxiv.org/abs/1911.13254.
        - floor (float): stability flooring when normalizing.
        - sample_rate (float): sample_rate used for training the model.
    """
    @capture_init
    def __init__(self,
                 chin=1,
                 chout=1,
                 hidden=48,
                 depth=5,
                 kernel_size=8,
                 stride=4,
                 causal=True,
                 resample=4,
                 growth=2,
                 max_hidden=10_000,
                 normalize=True,
                 glu=True,
                 rescale=0.1,
                 floor=1e-3,
                 sample_rate=16_000):

        super().__init__()
        if resample not in [1, 2, 4]:
            raise ValueError("Resample should be 1, 2 or 4.")

        spec_channels = [1, 8, 16, 32, 64, 128]
        fused_in_dim_128  = [544, 143, 39, 11, 3]
        fused_in_dim_256  = [576, 159, 47, 15, 5]
        fused_in_dim_512  = [640, 191, 63, 23, 9]
        fused_out_dim = [512, 128, 32, 8, 2]

        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.causal = causal
        self.floor = floor
        self.resample = resample
        self.normalize = normalize
        self.sample_rate = sample_rate

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.encoder_spec_128 = nn.ModuleList()
        self.decoder_spec_128 = nn.ModuleList()
        self.encoder_spec_256 = nn.ModuleList()
        self.decoder_spec_256 = nn.ModuleList()
        self.encoder_spec_512 = nn.ModuleList()
        self.decoder_spec_512 = nn.ModuleList()

        self.fusion_spec_128 = nn.ModuleList()
        self.fusion_spec_256 = nn.ModuleList()
        self.fusion_spec_512 = nn.ModuleList()
        self.linear_spec = nn.ModuleList()

        self.encoder_spec_extract_128 = nn.ModuleList()
        self.encoder_spec_extract_256 = nn.ModuleList()
        self.encoder_spec_extract_512 = nn.ModuleList()

        activation = nn.GLU(1) if glu else nn.ReLU()
        ch_scale = 2 if glu else 1

        for index in range(depth):
            encode = []
            encode += [
                nn.Conv1d(chin, hidden, kernel_size, stride),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden * ch_scale, 1), activation,
            ]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            decode += [
                nn.Conv1d(hidden, ch_scale * hidden, 1), activation,
                nn.ConvTranspose1d(hidden, chout, kernel_size, stride),
            ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))

            # ------------------------------- # 
            # --------- spec infor. --------- #
            encode_spec_128 = []
            encode_spec_256 = []
            encode_spec_512 = []
            
            encode_spec_128 += [spec_encoder(spec_channels[index], spec_channels[index + 1]),]
            encode_spec_256 += [spec_encoder(spec_channels[index], spec_channels[index + 1]),]
            encode_spec_512 += [spec_encoder(spec_channels[index], spec_channels[index + 1]),]
            self.encoder_spec_128.append(nn.Sequential(*encode_spec_128))
            self.encoder_spec_256.append(nn.Sequential(*encode_spec_256))
            self.encoder_spec_512.append(nn.Sequential(*encode_spec_512))

            decode_spec_128 = []
            decode_spec_256 = []
            decode_spec_512 = []

            if(index == 0):
                decode_spec_128 += [spec_decoder(spec_channels[index + 1]*2, spec_channels[index], is_last=True),]
                decode_spec_256 += [spec_decoder(spec_channels[index + 1]*2, spec_channels[index], is_last=True),]
                decode_spec_512 += [spec_decoder(spec_channels[index + 1]*2, spec_channels[index], is_last=True),]
            elif(index == 1):
                decode_spec_128 += [spec_decoder(spec_channels[index + 1]*2, spec_channels[index], paddings=True),]
                decode_spec_256 += [spec_decoder(spec_channels[index + 1]*2, spec_channels[index], paddings=True),]
                decode_spec_512 += [spec_decoder(spec_channels[index + 1]*2, spec_channels[index], paddings=True),]
            else:
                decode_spec_128 += [spec_decoder(spec_channels[index + 1]*2, spec_channels[index]),]
                decode_spec_256 += [spec_decoder(spec_channels[index + 1]*2, spec_channels[index]),]
                decode_spec_512 += [spec_decoder(spec_channels[index + 1]*2, spec_channels[index]),]
            self.decoder_spec_128.insert(0, nn.Sequential(*decode_spec_128))
            self.decoder_spec_256.insert(0, nn.Sequential(*decode_spec_256))
            self.decoder_spec_512.insert(0, nn.Sequential(*decode_spec_512))

            # spec. extraction layers
            encoder_spec_extract_128 = []
            encoder_spec_extract_256 = []
            encoder_spec_extract_512 = []
            encoder_spec_extract_128 += [
                    spec_extraction_encoder(spec_channels[index + 1], int(hidden/2), k=15, s=8, p=6),
                    spec_extraction_encoder(int(hidden/2), hidden, k=3, s=1, p=1),
                    ]
            encoder_spec_extract_256 += [
                    spec_extraction_encoder(spec_channels[index + 1], int(hidden/2), k=7, s=4, p=2),
                    spec_extraction_encoder(int(hidden/2), hidden, k=3, s=1, p=1),
                    ]
            encoder_spec_extract_512 += [
                    spec_extraction_encoder(spec_channels[index + 1], int(hidden/2), k=3, s=2, p=0),
                    spec_extraction_encoder(int(hidden/2), hidden, k=3, s=1, p=1),
                    ] 
            self.encoder_spec_extract_128.append(nn.Sequential(*encoder_spec_extract_128))
            self.encoder_spec_extract_256.append(nn.Sequential(*encoder_spec_extract_256))
            self.encoder_spec_extract_512.append(nn.Sequential(*encoder_spec_extract_512))

            # fusion layers
            fusion_spec_128 = []
            fusion_spec_128 += [
                nn.Linear(fused_in_dim_128[index], fused_out_dim[index]),
                nn.Dropout(p=0.1),
            ]
            self.fusion_spec_128.append(nn.Sequential(*fusion_spec_128))
            fusion_spec_256 = []
            fusion_spec_256 += [
                nn.Linear(fused_in_dim_256[index], fused_out_dim[index]),
                nn.Dropout(p=0.1),
            ]
            self.fusion_spec_256.append(nn.Sequential(*fusion_spec_256))
            fusion_spec_512 = []
            fusion_spec_512 += [
                nn.Linear(fused_in_dim_512[index], fused_out_dim[index]),
                nn.Dropout(p=0.1),
            ]
            self.fusion_spec_512.append(nn.Sequential(*fusion_spec_512))

            # linear layers
            linear_spec = []
            linear_spec += [
                nn.Linear(hidden, hidden*2),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(hidden*2, hidden),
                nn.Dropout(p=0.1),
            ]
            self.linear_spec.append(nn.Sequential(*linear_spec))

            chout = hidden
            chin = hidden
            hidden = min(int(growth * hidden), max_hidden)

        self.lstm = BLSTM(chin, bi=not causal)
        self.lstm_spec_128 = BLSTM(128, bi=not causal)
        self.lstm_spec_256 = BLSTM(384, bi=not causal)
        self.lstm_spec_512 = BLSTM(896, bi=not causal)

        if rescale:
            rescale_module(self, reference=rescale)


    def valid_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.
        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        length = math.ceil(length * self.resample)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length / self.resample))
        return int(length)

    @property
    def total_stride(self):
        return self.stride ** self.depth // self.resample

    def forward(self, mix):
        if mix.dim() == 2:
            mix = mix.unsqueeze(1)
            mix_ini = mix
        elif mix.dim() == 3:
            mix_ini = mix

        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            mix = mix / (self.floor + std)
        else:
            std = 1
        length = mix.shape[-1]
        x = mix
        x = F.pad(x, (0, self.valid_length(length) - length))
        x_ini = F.pad(mix_ini, (0, self.valid_length(length)+1024 - length))

        if self.resample == 2:
            x = upsample2(x)
        elif self.resample == 4:
            x = upsample2(x)
            x = upsample2(x)

        if x_ini.dim() == 3:
            x_ini = x_ini.squeeze(1)

        device = mix.device

        # spectrograms extraction
        x_spec_128 = th.stft(x_ini, 128, 64, 128, th.hann_window(128).to(device), return_complex=True)
        x_mag_128  = x_spec_128.abs()
        x_pha_128  = x_spec_128.angle()
        x_spec_128_mag = x_mag_128.permute(0, 2, 1).unsqueeze(1)

        x_spec_256 = th.stft(x_ini, 256, 128, 256, th.hann_window(256).to(device), return_complex=True)
        x_mag_256  = x_spec_256.abs()
        x_pha_256  = x_spec_256.angle()
        x_spec_256_mag = x_mag_256.permute(0, 2, 1).unsqueeze(1)

        x_spec_512 = th.stft(x_ini, 512, 256, 512, th.hann_window(512).to(device), return_complex=True)
        x_mag_512  = x_spec_512.abs()
        x_pha_512  = x_spec_512.angle()
        x_spec_512_mag = x_mag_512.permute(0, 2, 1).unsqueeze(1)

        # ----------- encoders
        skips = []
        skips_spec_128 = []
        skips_spec_256 = []
        skips_spec_512 = []
        hop_spec = 512
        for _depth in range(self.depth):
            # time domain stream
            x = self.encoder[_depth](x)
            residual = x

            # spec. domain streams
            x_spec_128_mag = self.encoder_spec_128[_depth](x_spec_128_mag)
            x_spec_256_mag = self.encoder_spec_256[_depth](x_spec_256_mag)
            x_spec_512_mag = self.encoder_spec_512[_depth](x_spec_512_mag)

            # time domain reshaping
            b, c, _len = x.shape
            frames = int(np.ceil(_len / hop_spec))

            x_padding = F.pad(x, (0, frames * hop_spec - _len))
            x_padding = x_padding.view(b, c, -1, hop_spec)

            residual_padding = x_padding

            x_spec_128_pad = F.pad(x_spec_128_mag, (0, 0, 0, 14))
            x_spec_256_pad = F.pad(x_spec_256_mag, (0, 0, 0, 6))
            x_spec_512_pad = F.pad(x_spec_512_mag, (0, 0, 0, 2))

            # spec. infor. extration
            x_spec_128_ext = self.encoder_spec_extract_128[_depth](x_spec_128_pad)
            x_spec_256_ext = self.encoder_spec_extract_256[_depth](x_spec_256_pad)
            x_spec_512_ext = self.encoder_spec_extract_512[_depth](x_spec_512_pad)

            x_spec_128_ext = x_spec_128_ext[:, :, :frames, :]
            x_spec_256_ext = x_spec_256_ext[:, :, :frames, :]
            x_spec_512_ext = x_spec_512_ext[:, :, :frames, :]

            # spec. infor. fusion
            x_padding_128 = th.cat((x_padding, x_spec_128_ext), 3)
            x_padding_256 = th.cat((x_padding, x_spec_256_ext), 3)
            x_padding_512 = th.cat((x_padding, x_spec_512_ext), 3)

            x_padding_128 = self.fusion_spec_128[_depth](x_padding_128)
            x_padding_256 = self.fusion_spec_256[_depth](x_padding_256)
            x_padding_512 = self.fusion_spec_512[_depth](x_padding_512)
            
            x_padding = residual_padding + x_padding_128 + x_padding_256 + x_padding_512
            x_padding = x_padding.view(b, c, -1)
            x_padding = x_padding[..., :_len]

            # feed forward layer
            residual_padding = x_padding
            x_padding = x_padding.permute(2, 0, 1)
            x_padding = self.linear_spec[_depth](x_padding)
            x_padding = x_padding.permute(1, 2, 0)

            x = x_padding + residual_padding

            if(_depth < self.depth - 1):
                hop_spec = int(hop_spec/4)
            skips.append(x)
            skips_spec_128.append(x_spec_128_mag)
            skips_spec_256.append(x_spec_256_mag)
            skips_spec_512.append(x_spec_512_mag)

        x = x.permute(2, 0, 1)
        x, _ = self.lstm(x)
        x = x.permute(1, 2, 0)
       
        # spec. lstms
        # spec. lstms
        b, c, t, f = x_spec_128_mag.shape
        x_spec_128_mag = x_spec_128_mag.permute(0, 2, 1, 3)
        x_spec_128_mag = x_spec_128_mag.reshape(b, t, -1)
        x_spec_128_mag, _ = self.lstm_spec_128(x_spec_128_mag)
        x_spec_128_mag = x_spec_128_mag.reshape(b, t, c, f)
        x_spec_128_mag = x_spec_128_mag.permute(0, 2, 1, 3)

        b, c, t, f = x_spec_256_mag.shape
        x_spec_256_mag = x_spec_256_mag.permute(0, 2, 1, 3)
        x_spec_256_mag = x_spec_256_mag.reshape(b, t, -1)
        x_spec_256_mag, _ = self.lstm_spec_256(x_spec_256_mag)
        x_spec_256_mag = x_spec_256_mag.reshape(b, t, c, f)
        x_spec_256_mag = x_spec_256_mag.permute(0, 2, 1, 3)

        b, c, t, f = x_spec_512_mag.shape
        x_spec_512_mag = x_spec_512_mag.permute(0, 2, 1, 3)
        x_spec_512_mag = x_spec_512_mag.reshape(b, t, -1)
        x_spec_512_mag, _ = self.lstm_spec_512(x_spec_512_mag)
        x_spec_512_mag = x_spec_512_mag.reshape(b, t, c, f)
        x_spec_512_mag = x_spec_512_mag.permute(0, 2, 1, 3)

        """
        b, c, t, f = x_spec_128_mag.shape
        x_spec_128_mag = x_spec_128_mag.reshape(b, c*f, t).permute(0, 2, 1)
        x_spec_128_mag, _ = self.lstm_spec_128(x_spec_128_mag)
        x_spec_128_mag = x_spec_128_mag.permute(0, 2, 1).reshape(b, c, t, f)
        b, c, t, f = x_spec_256_mag.shape
        x_spec_256_mag = x_spec_256_mag.reshape(b, c*f, t).permute(0, 2, 1)
        x_spec_256_mag, _ = self.lstm_spec_256(x_spec_256_mag)
        x_spec_256_mag = x_spec_256_mag.permute(0, 2, 1).reshape(b, c, t, f)
        b, c, t, f = x_spec_512_mag.shape
        x_spec_512_mag = x_spec_512_mag.reshape(b, c*f, t).permute(0, 2, 1)
        x_spec_512_mag, _ = self.lstm_spec_512(x_spec_512_mag)
        x_spec_512_mag = x_spec_512_mag.permute(0, 2, 1).reshape(b, c, t, f)
        """

        # decoders
        for _depth in range(self.depth):
            skip = skips.pop(-1)
            x = x + skip[..., :x.shape[-1]]
            x = self.decoder[_depth](x)

            skip_spec_128 = skips_spec_128.pop(-1)
            x_spec_128_mag = th.cat((x_spec_128_mag, skip_spec_128[..., :x_spec_128_mag.shape[-1]]), 1)
            x_spec_128_mag = self.decoder_spec_128[_depth](x_spec_128_mag)

            skip_spec_256 = skips_spec_256.pop(-1)
            x_spec_256_mag = th.cat((x_spec_256_mag, skip_spec_256[..., :x_spec_256_mag.shape[-1]]), 1)
            x_spec_256_mag = self.decoder_spec_256[_depth](x_spec_256_mag)
 
            skip_spec_512 = skips_spec_512.pop(-1)
            x_spec_512_mag = th.cat((x_spec_512_mag, skip_spec_512[..., :x_spec_512_mag.shape[-1]]) ,1)
            x_spec_512_mag = self.decoder_spec_512[_depth](x_spec_512_mag)

        x_spec_128_mag = x_spec_128_mag.squeeze(1).permute(0, 2, 1)
        x_spec_256_mag = x_spec_256_mag.squeeze(1).permute(0, 2, 1)
        x_spec_512_mag = x_spec_512_mag.squeeze(1).permute(0, 2, 1)

        x_spec_128_mag = x_spec_128_mag * x_mag_128
        x_spec_256_mag = x_spec_256_mag * x_mag_256
        x_spec_512_mag = x_spec_512_mag * x_mag_512

        # enhanced frequency infor.
        x_spec_128_enh = x_spec_128_mag * th.exp(x_pha_128*1j)
        x_spec_256_enh = x_spec_256_mag * th.exp(x_pha_256*1j)
        x_spec_512_enh = x_spec_512_mag * th.exp(x_pha_512*1j)

        x_spec_128_enh = th.istft(x_spec_128_enh, 128, 64, 128, th.hann_window(128).to(device)) #, center=False)
        x_spec_256_enh = th.istft(x_spec_256_enh, 256, 128, 256, th.hann_window(256).to(device)) # , center=False)
        x_spec_512_enh = th.istft(x_spec_512_enh, 512, 256, 512, th.hann_window(512).to(device)) #, center=False)

        if self.resample == 2:
            x = downsample2(x)
        elif self.resample == 4:
            x = downsample2(x)
            x = downsample2(x) 

        x = x[..., :length]
        x_spec_128_enh = x_spec_128_enh[..., :length]
        x_spec_256_enh = x_spec_256_enh[..., :length]
        x_spec_512_enh = x_spec_512_enh[..., :length]



        # return: 512 out (t), 1024 out (t), 2048 out (t), 128 out (f), 256 out (f), 512 out (f), 128 out (spec), 256 out (spec), 512 out (spec)
        return std * x, \
               std * x, \
               std * x, \
               x_spec_128_enh, \
               x_spec_256_enh, \
               x_spec_512_enh, \
               x_spec_128_mag, \
               x_spec_256_mag, \
               x_spec_512_mag

    def freeze(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze(self, module):
        for param in module.parameters():
            param.requires_grad = True
