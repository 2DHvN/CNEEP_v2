import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.add_position import add_x_channel


# ──────────────────────────────────────────────────────────────
# Periodic Padding (reused from existing codebase)
# ──────────────────────────────────────────────────────────────
class PeriodicPad1d(nn.Module):
    """Circular (periodic) padding for 1D tensors of shape [B, C, L]."""

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
# Masked 1D Convolution  (Requirement 1)
# ──────────────────────────────────────────────────────────────
class MaskedConv1d(nn.Module):
    """Conv1d whose kernel is masked so that only the two endpoints
    (index 0 and 2k) and the center (index k) are learnable.

    For a given distance parameter *k*, the kernel size is ``2*k + 1``.
    The mask is::

        [1, 0, 0, ..., 0, 1, 0, ..., 0, 0, 1]
         ^                 ^                 ^
         0                 k                2k

    This forces the convolution to look only at positions exactly
    *k* sites away and at the center — extracting the correlation
    information at distance *r = k*.
    """

    def __init__(self, in_channels: int, out_channels: int, k: int,
                 bias: bool = True):
        super(MaskedConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.kernel_size = 2 * k + 1

        # Learnable weight & bias (same shape as a normal Conv1d)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        # Build the binary mask: 1 at positions {0, k, 2k}, 0 elsewhere
        mask = torch.zeros(1, 1, self.kernel_size)
        mask[0, 0, 0] = 1.0
        mask[0, 0, k] = 1.0
        mask[0, 0, 2 * k] = 1.0
        self.register_buffer("mask", mask)

        # Kaiming initialisation
        nn.init.kaiming_normal_(self.weight, mode="fan_out",
                                nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, C_in, L]  (already periodically padded)."""
        return F.conv1d(x, self.weight * self.mask, self.bias,
                        stride=1, padding=0)


# ──────────────────────────────────────────────────────────────
# Single correlation-distance branch  (Requirements 4 & 5)
# ──────────────────────────────────────────────────────────────
class _CorrelationBranch(nn.Module):
    """One branch for a specific distance *k*.

    Pipeline (per branch)::

        PeriodicPad1d(k) → MaskedConv1d(kernel=2k+1) → ELU
        → [Conv1d(1×1) → ELU] × (n_hidden - 1)
        → Conv1d(1×1)  →  Local EP 1D Map  [B, 1, L]
    """

    def __init__(self, k: int, in_channels: int, hidden_channels: int,
                 n_hidden: int = 2):
        super(_CorrelationBranch, self).__init__()
        self.k = k
        self.pad = PeriodicPad1d(padding=k)

        layers = []

        # 1) Masked convolution — spatial feature extraction at distance k
        layers.append(MaskedConv1d(in_channels, hidden_channels, k=k))
        layers.append(nn.ELU(inplace=True))

        # 2) Point-wise (1×1) MLP layers — keeps receptive field isolated
        for i in range(n_hidden - 1):
            layers.append(nn.Conv1d(hidden_channels, hidden_channels,
                                    kernel_size=1))
            layers.append(nn.ELU(inplace=True))

        # 3) Final 1×1 projection → 1 channel (local EP map)
        layers.append(nn.Conv1d(hidden_channels, 1, kernel_size=1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, C, L]  — input *before* periodic padding.

        Returns
        -------
        [B, 1, L]  — local EP 1D map for this branch.
        """
        x = self.pad(x)       # [B, C, L + 2k]
        return self.net(x)     # [B, 1, L]


# ──────────────────────────────────────────────────────────────
# Multi-Scale CNEEP 1D  (full model)
# ──────────────────────────────────────────────────────────────
class MultiScaleCNEEP1D(nn.Module):
    """Parallel-branch CNEEP that decomposes entropy production (EP)
    by correlation length *r*.

    For each distance k ∈ {1, …, max_distance}, an independent branch
    extracts correlation features at exactly that distance and produces a
    local EP map.  Time-reversal antisymmetry is applied at the map level
    before Global Average Pooling yields a scalar EP estimate J_k per
    distance.

    Parameters
    ----------
    opt : namespace / object
        Must contain at least:
        - ``seq_len``        : int — number of temporal frames in the input.
        - ``positional``     : bool — whether to append a positional channel.
        - ``beta``           : float — coefficient for the antisymmetric term.
        - ``max_distance``   : int — largest correlation distance K.
        - ``n_channel``      : int — hidden channel width inside each branch.
        - ``n_hidden``       : int (optional, default 2) — number of hidden
          1×1-conv layers per branch.
    """

    def __init__(self, opt):
        super(MultiScaleCNEEP1D, self).__init__()

        self.positional = opt.positional
        self.beta = opt.beta
        self.max_distance = opt.max_distance

        in_channels = opt.seq_len + (1 if opt.positional else 0)
        hidden_channels = opt.n_channel
        n_hidden = getattr(opt, "n_hidden", 2)

        # Build K independent branches (k = 1 … max_distance)
        self.branches = nn.ModuleList([
            _CorrelationBranch(
                k=k,
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                n_hidden=n_hidden,
            )
            for k in range(1, self.max_distance + 1)
        ])

        # Kaiming init for all Conv1d inside branches
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ----- shared backbone: produces the local EP map ----- #
    def _local_map(self, x: torch.Tensor,
                   branch: _CorrelationBranch) -> torch.Tensor:
        """Run one branch on input *x* and return [B, 1, L] local EP map."""
        return branch(x)

    # ----- forward ----- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, seq_len, L]

        Returns
        -------
        J : [B, K]  where J[:, k-1] is the estimated EP at distance k.
        """
        # Time-forward and time-reversed inputs
        x_ = x                          # forward
        _x = torch.flip(x, [1])         # reverse time dimension

        # Δφ (state difference used for the symmetric term)
        delta = (x[:, 0, :] - x[:, 1, :]).unsqueeze(1)  # [B, 1, L]

        # Optionally append positional channel
        if self.positional:
            x_ = add_x_channel(x_)
            _x = add_x_channel(_x)

        J_list = []

        for branch in self.branches:
            # Local EP maps from forward / reversed inputs  [B, 1, L]
            map_fwd = self._local_map(x_, branch)
            map_rev = self._local_map(_x, branch)

            # Time-reversal antisymmetry at the map level (Req. 6)
            #   symmetric  part: (H(x_) + H(_x)) * delta
            #   antisymmetric  : (H(x_) - H(_x)) * beta
            local_ep = (map_fwd + map_rev) * delta \
                     + (map_fwd - map_rev) * self.beta   # [B, 1, L]

            # Global Average Pooling over the spatial dimension (Req. 7)
            J_k = local_ep.mean(dim=2)          # [B, 1]
            J_list.append(J_k)

        # Stack into [B, K]
        return torch.cat(J_list, dim=1)


# ──────────────────────────────────────────────────────────────
# Convenience alias so that external training scripts can
# instantiate the model with the same ``CNEEP(opt)`` interface.
# ──────────────────────────────────────────────────────────────
CNEEP = MultiScaleCNEEP1D
