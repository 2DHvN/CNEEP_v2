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