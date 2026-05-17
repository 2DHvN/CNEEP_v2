import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.add_position import add_x_channel


# ──────────────────────────────────────────────────────────────
# Periodic Padding (reused from existing codebase)
# ──────────────────────────────────────────────────────────────
class PeriodicPad1d(nn.Module):
    def __init__(self, padding):
        super(PeriodicPad1d, self).__init__()
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

    def forward(self, x):
        left_pad, right_pad = self.padding

        if left_pad == 0 and right_pad == 0:
            return x

        output = x
        if left_pad > 0:
            output = torch.cat([x[:, :, -left_pad:], output], dim=-1)
        if right_pad > 0:
            output = torch.cat([output, x[:, :, :right_pad]], dim=-1)
        return output


# ──────────────────────────────────────────────────────────────
# Masked 1D Convolution 
# ──────────────────────────────────────────────────────────────
class MaskedConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k: int,
                 bias: bool = True):
        super(MaskedConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.kernel_size = 2 * k + 1

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        mask = torch.zeros(1, 1, self.kernel_size)
        mask[0, 0, 0] = 1.0
        mask[0, 0, k] = 1.0
        mask[0, 0, 2 * k] = 1.0
        self.register_buffer("mask", mask)

        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(x, self.weight * self.mask, self.bias,
                        stride=1, padding=0)


# ──────────────────────────────────────────────────────────────
# Spatial Weighted Sum 1D (Breaks Translation Equivariance)
# ──────────────────────────────────────────────────────────────
class SpatialWeightedSum1d(nn.Module):
    """
    Applies a learnable spatial weight map of shape [1, C, L].
    """
    def __init__(self, in_channels: int, L: int):
        super(SpatialWeightedSum1d, self).__init__()
        self.weight = nn.Parameter(torch.empty(1, in_channels, L))
        self.bias = nn.Parameter(torch.empty(1, 1, L))
        
        nn.init.normal_(self.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sum(x * self.weight, dim=1, keepdim=True) + self.bias


# ──────────────────────────────────────────────────────────────
# Single Correlation Branch 1DF
# ──────────────────────────────────────────────────────────────
class _CorrelationBranch1DF(nn.Module):
    def __init__(self, k: int, in_channels: int, hidden_channels: int,
                 reduce_channels: int, L: int, n_hidden: int = 2):
        super(_CorrelationBranch1DF, self).__init__()
        self.k = k
        self.pad = PeriodicPad1d(padding=k)

        layers = []
        layers.append(MaskedConv1d(in_channels, hidden_channels, k=k))
        layers.append(nn.ELU(inplace=True))

        for i in range(n_hidden - 1):
            layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1))
            layers.append(nn.ELU(inplace=True))

        layers.append(nn.Conv1d(hidden_channels, reduce_channels, kernel_size=1))
        layers.append(nn.ELU(inplace=True))

        self.net = nn.Sequential(*layers)
        self.spatial_weight = SpatialWeightedSum1d(reduce_channels, L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)       # [B, C, L + 2k]
        features = self.net(x)     # [B, C_reduce, L]
        return self.spatial_weight(features)


# ──────────────────────────────────────────────────────────────
# Multi-Scale CNEEP 1DF (full model)
# ──────────────────────────────────────────────────────────────
class MultiScaleCNEEP_1DF(nn.Module):
    def __init__(self, opt):
        super(MultiScaleCNEEP_1DF, self).__init__()

        self.positional = getattr(opt, "positional", False)
        self.beta = getattr(opt, "beta", 1.0)
        self.max_distance = opt.max_distance
        
        if not hasattr(opt, "L"):
            raise ValueError("opt.L is required for XDF models to define the spatial mask.")
        L = opt.L

        in_channels = opt.seq_len + (1 if self.positional else 0)
        hidden_channels = opt.n_channel
        n_hidden = getattr(opt, "n_hidden", 2)
        reduce_channels = getattr(opt, "reduce_channel", 4)

        self.include_k0 = getattr(opt, "include_k0", True)
        start_k = 0 if self.include_k0 else 1
        
        self.branches = nn.ModuleList([
            _CorrelationBranch1DF(
                k=k,
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                reduce_channels=reduce_channels,
                L=L,
                n_hidden=n_hidden,
            )
            for k in range(start_k, self.max_distance + 1)
        ])

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _local_map(self, x: torch.Tensor,
                   branch: _CorrelationBranch1DF) -> torch.Tensor:
        return branch(x)

    def forward(self, x: torch.Tensor, return_maps: bool = False) -> torch.Tensor:
        x_ = x
        _x = torch.flip(x, [1])

        if self.positional:
            x_ = add_x_channel(x_)
            _x = add_x_channel(_x)

        J_list = []
        map_list = []

        for branch in self.branches:
            map_fwd = self._local_map(x_, branch)
            map_rev = self._local_map(_x, branch)

            local_ep = (map_fwd - map_rev)

            if return_maps:
                map_list.append(local_ep)
            else:
                J_k = local_ep.mean(dim=2)
                J_list.append(J_k)

        if return_maps:
            return torch.cat(map_list, dim=1)
        else:
            return torch.cat(J_list, dim=1)

CNEEP = MultiScaleCNEEP_1DF
