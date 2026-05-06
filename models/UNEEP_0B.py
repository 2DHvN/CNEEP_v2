import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.add_position import add_xy_channels

class PeriodicPad2d(nn.Module):
    def __init__(self, padding):
        super(PeriodicPad2d, self).__init__()
        if isinstance(padding, int):
            self.padding = (padding, padding, padding, padding)
        elif len(padding) == 2:
            self.padding = (padding[0], padding[0], padding[1], padding[1])
        else:
            self.padding = padding

    def forward(self, x):
        # x shape: [Batch, Channel, H, W]
        pad_left, pad_right, pad_top, pad_bottom = self.padding

        if pad_left > 0:
            x = torch.cat([x[:, :, :, -pad_left:], x], dim=-1)
        if pad_right > 0:
            x = torch.cat([x, x[:, :, :, pad_left:pad_left+pad_right]], dim=-1)
        if pad_top > 0:
            x = torch.cat([x[:, :, -pad_top:, :], x], dim=-2)
        if pad_bottom > 0:
            x = torch.cat([x, x[:, :, pad_top:pad_top+pad_bottom, :]], dim=-2)

        return x

class PeriodicUpsample2d(nn.Module):
    def __init__(self, scale_factor=2):
        super(PeriodicUpsample2d, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        # x: (B, C, H, W)
        H, W = x.shape[-2], x.shape[-1]
        target_H, target_W = H * self.scale_factor, W * self.scale_factor
        # Pad first element to enable wrap-around interpolation
        x_padded = torch.cat([x, x[:, :, :1, :]], dim=-2)  # (B, C, H+1, W)
        x_padded = torch.cat([x_padded, x_padded[:, :, :, :1]], dim=-1)  # (B, C, H+1, W+1)
        # Interpolate: align_corners=True ensures output[0]==input[0], output[target_len]==input[0]
        x_up = F.interpolate(x_padded, size=(target_H + 1, target_W + 1), mode='bilinear', align_corners=True)
        # Drop the last point (duplicate of first)
        return x_up[:, :, :-1, :-1]

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
            tmp.add_module("periodic_pad", PeriodicPad2d(padding=2))
        else:
            tmp.add_module("pad", nn.ZeroPad2d(padding=2))
        
        tmp.add_module("conv",
                       nn.Conv2d(opt.seq_len + (2 if opt.positional else 0), opt.n_channel,
                                 kernel_size=5, stride=1, padding=0))
        tmp.add_module("elu", nn.ELU(inplace=True))
        tmp.add_module("pool",
                       nn.AvgPool2d(kernel_size=2, stride=2))
        setattr(self, "layer1", tmp)

        for i in range(opt.n_layer - 1):
            tmp = nn.Sequential()
            if opt.periodic:
                tmp.add_module("periodic_pad1", PeriodicPad2d(padding=1))
            else:
                tmp.add_module("pad1", nn.ZeroPad2d(padding=1))
            tmp.add_module("conv1",
                           nn.Conv2d(opt.n_channel * (2 ** i), opt.n_channel * (2 ** i),
                                     kernel_size=3, stride=1, padding=0))
            tmp.add_module("elu1", nn.ELU(inplace=True))

            if opt.periodic:
                tmp.add_module("periodic_pad2", PeriodicPad2d(padding=1))
            else:
                tmp.add_module("pad2", nn.ZeroPad2d(padding=1))
            tmp.add_module("conv2",
                           nn.Conv2d(opt.n_channel * (2 ** i), opt.n_channel * (2 ** (i + 1)),
                                     kernel_size=3, stride=1, padding=0))
            tmp.add_module("elu2", nn.ELU(inplace=True))

            if i < opt.n_layer - 2:
                tmp.add_module("pool",
                               nn.AvgPool2d(kernel_size=2, stride=2))

            setattr(self, f"layer{i + 2}", tmp)

        #
        # bootstrapping decoder: multi-stage upsample + conv with gradual channel reduction
        #
        bottleneck_channels = opt.n_channel * (2 ** (opt.n_layer - 1))
        
        layers = []
        curr_channels = bottleneck_channels

        # Multi-stage upsampling with gradual channel reduction
        for i in range(opt.n_layer - 1):
            next_channels = opt.n_channel * (2 ** (opt.n_layer - 2 - i))
            
            if opt.periodic:
                layers.append(PeriodicUpsample2d(scale_factor=2))
            else:
                layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            
            if opt.periodic:
                layers.append(PeriodicPad2d(padding=1))
            else:
                layers.append(nn.ZeroPad2d(padding=1))
                
            layers.append(nn.Conv2d(curr_channels, next_channels, kernel_size=3))
            layers.append(nn.ELU(inplace=True))
            
            curr_channels = next_channels

        # Final projection to 1 channel
        layers.append(nn.Conv2d(curr_channels, 1, kernel_size=1))
        self.decoder = nn.Sequential(*layers)

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

        x = self.decoder(x)

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
