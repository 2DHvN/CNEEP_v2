import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.add_position import add_x_channel

import torch
import torch.nn as nn


class PeriodicPad1d(nn.Module):
    def __init__(self, padding):
        super(PeriodicPad1d, self).__init__()
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

    def forward(self, x):
        # x shape: [Batch, Channel, Length]
        left_pad, right_pad = self.padding

        if left_pad == 0 and right_pad == 0:
            return x

        output = x

        if left_pad > 0:
            left_part = x[:, :, -left_pad:]
            output = torch.cat([left_part, output], dim=-1)

        if right_pad > 0:
            right_part = x[:, :, :right_pad]
            output = torch.cat([output, right_part], dim=-1)

        return output

class CNEEP(nn.Module):
    def __init__(self, opt):
        super(CNEEP, self).__init__()
        self.n_layer = opt.n_layer
        self.init_channel = opt.n_channel
        self.positional = opt.positional
        self.beta = opt.beta

        #
        # encoding layer
        #
        tmp = nn.Sequential()
        if opt.periodic:
            tmp.add_module("periodic_pad", PeriodicPad1d(padding=2))
        else:
            tmp.add_module("pad", nn.ConstantPad1d(padding=2, value=0))
        tmp.add_module("conv",
                       nn.Conv1d(opt.seq_len + (1 if opt.positional else 0), opt.n_channel,
                                 kernel_size=5, stride=1, padding=0))
        tmp.add_module("elu", nn.ELU(inplace=True))
        tmp.add_module("pool",
                       nn.AvgPool1d(kernel_size=2, stride=2))
        setattr(self, "layer1", tmp)

        for i in range(opt.n_layer - 1):
            tmp = nn.Sequential()
            if opt.periodic:
                tmp.add_module("periodic_pad1", PeriodicPad1d(padding=1))
            else:
                tmp.add_module("pad1", nn.ConstantPad1d(padding=1, value=0))
            tmp.add_module("conv1",
                           nn.Conv1d(opt.n_channel * (2 ** i), opt.n_channel * (2 ** i),
                                     kernel_size=3, stride=1, padding=0))
            tmp.add_module("elu1", nn.ELU(inplace=True))

            if opt.periodic:
                tmp.add_module("periodic_pad2", PeriodicPad1d(padding=1))
            else:
                tmp.add_module("pad2", nn.ConstantPad1d(padding=1, value=0))
            tmp.add_module("conv2",
                           nn.Conv1d(opt.n_channel * (2 ** i), opt.n_channel * (2 ** (i + 1)),
                                     kernel_size=3, stride=1, padding=0))
            tmp.add_module("elu2", nn.ELU(inplace=True))

            if i < opt.n_layer - 2:
                tmp.add_module("pool",
                               nn.AvgPool1d(kernel_size=2, stride=2))

            setattr(self, f"layer{i + 2}", tmp)

        #
        # bootstrapping decoder: single bilinear upsample + conv
        #
        bottleneck_channels = opt.n_channel * (2 ** (opt.n_layer - 1))
        upsample_factor = 2 ** (opt.n_layer - 1)

        self.decoder = nn.Sequential(
            nn.Conv1d(bottleneck_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=upsample_factor, mode='linear', align_corners=True),
        )

        # initialize parameters
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def H(self, x):
        for i in range(self.n_layer):
            f = getattr(self, f"layer{i+1}")
            x = f(x)

        x = self.decoder(x)

        return x

    def forward(self, x):
        x_ = x
        _x = torch.flip(x, [1])

        delta = x[:, 0, :] - x[:, 1, :]
        delta = delta.reshape(x.shape[0], 1, x.shape[2])

        if self.positional:
            x_ = add_x_channel(x_)
            _x = add_x_channel(_x)

        x_ = self.H(x_)
        _x = self.H(_x)

        return (x_ + _x) * delta + (x_ - _x) * beta
