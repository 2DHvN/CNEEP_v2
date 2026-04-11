import torch
import torch.nn as nn
from utils.add_position import add_x_channel
from utils.periodic_pad import PeriodicPad1d


class CNEEP(nn.Module):
    """
    XNEEP_1D: Fully learnable decomposition model.

    Output = (J(_x) - J(x_)) * (F(_x) + F(x_))

    where J and F are independent UNet-like encoder-decoder networks.
      - J captures the antisymmetric (current-like) component
      - F captures the symmetric (force-like) component
    This replaces the handcrafted delta_phi with a learnable J network.
    """

    def __init__(self, opt):
        super(CNEEP, self).__init__()
        self.n_layer = opt.n_layer
        self.init_channel = opt.n_channel
        self.positional = opt.positional

        # Build two independent networks: J and F
        self._build_network(opt, prefix="J")
        self._build_network(opt, prefix="F")

        # initialize parameters
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _build_network(self, opt, prefix):
        """
        Build a UNet-like encoder-decoder network with the given prefix.
        Uses bootstrapped decoder (1x1 conv + upsample) like UNEEP_1D_phiB.
        """
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
        setattr(self, f"{prefix}_layer1", tmp)

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

            setattr(self, f"{prefix}_layer{i + 2}", tmp)

        #
        # bootstrapping decoder: single 1x1 conv + upsample
        #
        bottleneck_channels = opt.n_channel * (2 ** (opt.n_layer - 1))
        upsample_factor = 2 ** (opt.n_layer - 1)

        decoder = nn.Sequential(
            nn.Conv1d(bottleneck_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.Upsample(scale_factor=upsample_factor, mode='linear', align_corners=True),
        )
        setattr(self, f"{prefix}_decoder", decoder)

    def _forward_network(self, x, prefix):
        """Forward pass through the network identified by prefix (J or F)."""
        for i in range(self.n_layer):
            f = getattr(self, f"{prefix}_layer{i + 1}")
            x = f(x)

        decoder = getattr(self, f"{prefix}_decoder")
        x = decoder(x)

        return x

    def forward(self, x):
        x_ = x
        _x = torch.flip(x, [1])

        if self.positional:
            x_ = add_x_channel(x_)
            _x = add_x_channel(_x)

        # J network: antisymmetric part -> J(_x) - J(x_)
        J_x_ = self._forward_network(x_, "J")
        J__x = self._forward_network(_x, "J")
        current = J__x - J_x_  # antisymmetric

        # F network: symmetric part -> F(_x) + F(x_)
        F_x_ = self._forward_network(x_, "F")
        F__x = self._forward_network(_x, "F")
        force = F__x + F_x_  # symmetric

        return current * force
