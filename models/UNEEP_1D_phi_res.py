import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.add_position import add_x_channel
from utils.periodic_pad import PeriodicPad1d

class CNEEP(nn.Module):
    def __init__(self, opt):
        super(CNEEP, self).__init__()
        self.n_layer = opt.n_layer
        self.init_channel = opt.n_channel
        self.positional = opt.positional

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
        # decoding layer
        #
        for i in list(reversed(range(opt.n_layer - 1))):
            tmp = nn.Sequential()

            if i < opt.n_layer - 2:
                tmp.add_module("upsample",
                               nn.Upsample(scale_factor=2, mode='linear', align_corners=True))

            tmp.add_module("periodic_pad1", PeriodicPad1d(padding=1))
            tmp.add_module("conv1",
                           nn.Conv1d(opt.n_channel * (2 ** (i + 1)), opt.n_channel * (2 ** i),
                                     kernel_size=3, stride=1, padding=0))
            tmp.add_module("elu1", nn.ELU(inplace=True))

            tmp.add_module("periodic_pad2", PeriodicPad1d(padding=1))
            tmp.add_module("conv2",
                           nn.Conv1d(opt.n_channel * (2 ** i), opt.n_channel * (2 ** i),
                                     kernel_size=3, stride=1, padding=0))
            tmp.add_module("elu2", nn.ELU(inplace=True))

            setattr(self, f"r_layer{i+2}", tmp)

        tmp = nn.Sequential()
        tmp.add_module("upsample", nn.Upsample(scale_factor=2, mode='linear', align_corners=True))

        tmp.add_module("periodic_pad", PeriodicPad1d(padding=2))
        tmp.add_module("conv", nn.Conv1d(opt.n_channel, 1, kernel_size=5, stride=1, padding=0))
        setattr(self, "r_layer1", tmp)

        # initialize parameters
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def H(self, x):
        features = []
        for i in range(self.n_layer - 1):
            f = getattr(self, f"layer{i+1}")
            x = f(x)
            features.append(x)
            
        f = getattr(self, f"layer{self.n_layer}")
        x = f(x)

        for i in list(reversed(range(self.n_layer))):
            if i < self.n_layer - 1:
                x = x + features[i]
                
            f = getattr(self, f"r_layer{i+1}")
            x = f(x)

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

        return (x_ + _x) * delta
