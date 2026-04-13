import torch
import torch.nn as nn
from utils.add_position import add_x_channel

class SNEEP_Discriminator(nn.Module):
    """
    Discriminator for System Entropy Production Pattern.
    Takes a single field state (e.g. phi_t or phi_tp1),
    and outputs a Local Density Ratio Logit field.
    The ratio is used to measure log P(phi_t) - log P(phi_tp1).
    """
    def __init__(self, opt):
        super(SNEEP_Discriminator, self).__init__()
        self.n_layer = opt.n_layer
        self.init_channel = opt.n_channel
        self.positional = opt.positional

        #
        # encoding layer
        #
        tmp = nn.Sequential()
        # Input channel size is opt.seq_len parameter. We set it to 1 to analyze single state fields.
        input_channels = opt.seq_len + (1 if opt.positional else 0)
        tmp.add_module("conv",
                       nn.Conv1d(input_channels, opt.n_channel,
                                 kernel_size=5, stride=1, padding=2))
        tmp.add_module("elu", nn.ELU(inplace=True))
        tmp.add_module("pool",
                       nn.AvgPool1d(kernel_size=2, stride=2))
        setattr(self, "layer1", tmp)

        for i in range(opt.n_layer - 1):
            tmp = nn.Sequential()
            tmp.add_module("conv1",
                           nn.Conv1d(opt.n_channel * (2 ** i), opt.n_channel * (2 ** i),
                                     kernel_size=3, stride=1, padding=1))
            tmp.add_module("elu1", nn.ELU(inplace=True))

            tmp.add_module("conv2",
                           nn.Conv1d(opt.n_channel * (2 ** i), opt.n_channel * (2 ** (i + 1)),
                                     kernel_size=3, stride=1, padding=1))
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

            tmp.add_module("conv1",
                           nn.Conv1d(opt.n_channel * (2 ** (i + 1)), opt.n_channel * (2 ** i),
                                     kernel_size=3, stride=1, padding=1))
            tmp.add_module("elu1", nn.ELU(inplace=True))

            tmp.add_module("conv2",
                           nn.Conv1d(opt.n_channel * (2 ** i), opt.n_channel * (2 ** i),
                                     kernel_size=3, stride=1, padding=1))
            tmp.add_module("elu2", nn.ELU(inplace=True))

            setattr(self, f"r_layer{i+2}", tmp)

        tmp = nn.Sequential()
        tmp.add_module("upsample", nn.Upsample(scale_factor=2, mode='linear', align_corners=True))
        tmp.add_module("conv", nn.Conv1d(opt.n_channel, 1, kernel_size=5, stride=1, padding=2))
        setattr(self, "r_layer1", tmp)

        # initialize parameters
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        x: shape (B, 1, Lx) where 1 is the channel for single state phi
        outputs shape (B, 1, Lx) logit field
        """
        if self.positional:
            x = add_x_channel(x)

        out = x
        for i in range(self.n_layer):
            f = getattr(self, f"layer{i+1}")
            out = f(out)

        for i in list(reversed(range(self.n_layer))):
            f = getattr(self, f"r_layer{i+1}")
            out = f(out)

        return out
