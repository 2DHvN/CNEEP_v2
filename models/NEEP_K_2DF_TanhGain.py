import torch
import torch.nn as nn
import torch.nn.functional as F


class PeriodicPad2d(nn.Module):
    """Circular padding for 2D tensors of shape [B, C, Lx, Ly]."""

    def __init__(self, padding):
        super(PeriodicPad2d, self).__init__()
        if isinstance(padding, int):
            self.padding = (padding, padding, padding, padding)
        else:
            self.padding = padding

    def forward(self, x):
        return F.pad(x, self.padding, mode="circular")


class ExclusiveMaskedConv2dNoBias(nn.Module):
    """
    Bias-free exclusive masked convolution.

    k = 0:
        center-only linear map.
    k > 0:
        sum_delta W_delta (X_{i+delta} - X_i) over the Chebyshev shell.
    """

    def __init__(self, in_channels: int, out_channels: int, k: int):
        super(ExclusiveMaskedConv2dNoBias, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.kernel_size = 2 * k + 1

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, self.kernel_size, self.kernel_size)
        )

        mask = torch.zeros(1, 1, self.kernel_size, self.kernel_size)
        if k == 0:
            mask[0, 0, 0, 0] = 1.0
        else:
            mask[0, 0, 0, :] = 1.0
            mask[0, 0, 2 * k, :] = 1.0
            mask[0, 0, :, 0] = 1.0
            mask[0, 0, :, 2 * k] = 1.0
            mask[0, 0, k, k] = 0.0
        self.register_buffer("mask", mask)

        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain("tanh"))

    def forward(self, x: torch.Tensor, x_center: torch.Tensor = None) -> torch.Tensor:
        if self.k == 0:
            return F.conv2d(x, self.weight * self.mask, bias=None)

        B, _, Lx, Ly = x_center.shape
        out = torch.zeros(B, self.out_channels, Lx, Ly, device=x.device, dtype=x.dtype)

        for dy in range(-self.k, self.k + 1):
            for dx in range(-self.k, self.k + 1):
                if max(abs(dy), abs(dx)) == self.k:
                    x_shifted = x[
                        :,
                        :,
                        self.k + dy : self.k + dy + Lx,
                        self.k + dx : self.k + dx + Ly,
                    ]
                    x_rel = x_shifted - x_center
                    w_delta = self.weight[:, :, self.k + dy, self.k + dx]
                    w_delta = w_delta.unsqueeze(-1).unsqueeze(-1)
                    out = out + F.conv2d(x_rel, w_delta, bias=None)

        return out


class _TanhGainKBranch2DF(nn.Module):
    """One bias-free tanh branch with a learnable scalar gain."""

    def __init__(
        self,
        k: int,
        in_channels: int,
        hidden_channels: int,
        n_hidden: int = 2,
        n_components: int = 1,
        gain_init: float = 1.0,
    ):
        super(_TanhGainKBranch2DF, self).__init__()
        self.k = k
        self.pad = PeriodicPad2d(k) if k > 0 else None
        self.gain = nn.Parameter(torch.tensor(float(gain_init)))

        self.masked_conv = ExclusiveMaskedConv2dNoBias(in_channels, hidden_channels, k=k)

        post_layers = []
        for _ in range(max(n_hidden - 1, 0)):
            post_layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False))
            post_layers.append(nn.Tanh())
        post_layers.append(nn.Conv2d(hidden_channels, n_components, kernel_size=1, bias=False))
        self.post_net = nn.Sequential(*post_layers)

        for m in self.post_net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("tanh"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_center = x
        x_padded = self.pad(x) if self.pad is not None else x
        h = torch.tanh(self.masked_conv(x_padded, x_center=x_center))
        return self.gain * self.post_net(h)


def add_spatial_channels(x: torch.Tensor) -> torch.Tensor:
    """Append normalized x/y coordinate channels to [B, C, Lx, Ly]."""
    B, _, Lx, Ly = x.shape
    device = x.device
    dtype = x.dtype

    xx, yy = torch.meshgrid(
        torch.arange(Lx, device=device, dtype=dtype),
        torch.arange(Ly, device=device, dtype=dtype),
        indexing="ij",
    )
    xx = xx / max(Lx - 1, 1)
    yy = yy / max(Ly - 1, 1)

    xx = xx.unsqueeze(0).unsqueeze(0).expand(B, 1, Lx, Ly)
    yy = yy.unsqueeze(0).unsqueeze(0).expand(B, 1, Lx, Ly)
    return torch.cat([x, xx, yy], dim=1)


class MultiScaleK_2DF_TanhGain(nn.Module):
    """
    K2DF variant with bias-free convolutions, tanh activations, and
    a learnable scalar gain for each distance branch.

    The forward API matches MultiScaleK_2DF:
        model(x) -> [B, K]
        model(x, return_maps=True) -> [B, K, Lx, Ly]
    """

    def __init__(self, opt):
        super(MultiScaleK_2DF_TanhGain, self).__init__()

        self.positional = opt.positional
        self.beta = opt.beta
        self.max_distance = opt.max_distance
        self.n_components = getattr(opt, "n_components", 1)
        self.include_k0 = getattr(opt, "include_k0", True)

        in_channels = opt.seq_len * self.n_components + (2 if opt.positional else 0)
        hidden_channels = opt.n_channel
        n_hidden = getattr(opt, "n_hidden", 2)
        gain_init = getattr(opt, "branch_gain_init", 1.0)

        start_k = 0 if self.include_k0 else 1
        self.branches = nn.ModuleList(
            [
                _TanhGainKBranch2DF(
                    k=k,
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    n_hidden=n_hidden,
                    n_components=self.n_components,
                    gain_init=gain_init,
                )
                for k in range(start_k, self.max_distance + 1)
            ]
        )

    def _local_map(self, x: torch.Tensor, branch: _TanhGainKBranch2DF) -> torch.Tensor:
        return branch(x)

    def forward(self, x: torch.Tensor, return_maps: bool = False) -> torch.Tensor:
        if self.n_components > 1:
            B, S, C, Lx, Ly = x.shape
            x_ = x.reshape(B, S * C, Lx, Ly)
            _x = torch.flip(x, [1]).reshape(B, S * C, Lx, Ly)
            dx = x[:, 1, :, :, :] - x[:, 0, :, :, :]
            _dx = -dx
        else:
            x_ = x
            _x = torch.flip(x, [1])
            dx = (x[:, 1, :, :] - x[:, 0, :, :]).unsqueeze(1)
            _dx = -dx

        if self.positional:
            x_ = add_spatial_channels(x_)
            _x = add_spatial_channels(_x)

        J_list = []
        map_list = []

        for branch in self.branches:
            A_fwd = self._local_map(x_, branch)
            A_rev = self._local_map(_x, branch)

            map_fwd = (A_fwd * dx).sum(dim=1, keepdim=True)
            map_rev = (A_rev * _dx).sum(dim=1, keepdim=True)
            local_ep = map_fwd - map_rev

            if return_maps:
                map_list.append(local_ep)
            else:
                J_list.append(local_ep.mean(dim=(2, 3)))

        if return_maps:
            return torch.cat(map_list, dim=1)
        return torch.cat(J_list, dim=1)
