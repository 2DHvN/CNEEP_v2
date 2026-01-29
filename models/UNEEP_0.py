import torch
import torch.nn as nn
from utils.add_position import add_xy_channels

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
        tmp.add_module("conv",
                       nn.Conv2d(opt.seq_len + (2 if opt.positional else 0), opt.n_channel,
                                 kernel_size=5, stride=1, padding=2))
        tmp.add_module("relu", nn.ReLU(inplace=True))
        tmp.add_module("maxpool",
                       nn.MaxPool2d(kernel_size=2, stride=2))
        setattr(self, "layer1", tmp)

        for i in range(opt.n_layer - 1):
            tmp = nn.Sequential()
            tmp.add_module("conv1",
                           nn.Conv2d(opt.n_channel * (2 ** i), opt.n_channel * (2 ** i),
                                     kernel_size=3, stride=1, padding=1))
            tmp.add_module("relu1", nn.ReLU(inplace=True))

            tmp.add_module("conv2",
                           nn.Conv2d(opt.n_channel * (2 ** i), opt.n_channel * (2 ** (i + 1)),
                                     kernel_size=3, stride=1, padding=1))
            tmp.add_module("relu2", nn.ReLU(inplace=True))

            if i < opt.n_layer - 2:
                tmp.add_module("maxpool",
                               nn.MaxPool2d(kernel_size=2, stride=2))

            setattr(self, f"layer{i + 2}", tmp)

        #
        # decoding layer
        #
        for i in list(reversed(range(opt.n_layer - 1))):
            tmp = nn.Sequential()

            if i < opt.n_layer - 2:
                tmp.add_module("upsample",
                               nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))

            tmp.add_module("conv1",
                           nn.Conv2d(opt.n_channel * (2 ** (i + 1)), opt.n_channel * (2 ** i),
                                     kernel_size=3, stride=1, padding=1))
            tmp.add_module("relu1", nn.ReLU(inplace=True))

            tmp.add_module("conv2",
                           nn.Conv2d(opt.n_channel * (2 ** i), opt.n_channel * (2 ** i),
                                     kernel_size=3, stride=1, padding=1))
            tmp.add_module("relu2", nn.ReLU(inplace=True))

            setattr(self, f"r_layer{i+2}", tmp)

        tmp = nn.Sequential()
        tmp.add_module("upsample", nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        tmp.add_module("conv", nn.Conv2d(opt.n_channel, 1, kernel_size=5, stride=1, padding=2))
        setattr(self, "r_layer1", tmp)

        # initialize parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def H(self, x):
        for i in range(self.n_layer):
            f = getattr(self, f"layer{i+1}")
            x = f(x)

        for i in list(reversed(range(self.n_layer))):
            f = getattr(self, f"r_layer{i+1}")
            x = f(x)

        return x

    def forward(self, x):
        x_ = x
        _x = torch.flip(x, [1])

        if self.positional:
            x_ = add_xy_channels(x_)
            _x = add_xy_channels(_x)

        x_ = self.H(x_)
        _x = self.H(_x)

        return x_ - _x

