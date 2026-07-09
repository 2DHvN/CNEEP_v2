"""Euclidean-annulus ShellForce CNEEP for continuous ABP fields.

This module mirrors ``NEEP_ShellForce_2D.py`` but replaces exclusive
Chebyshev shells with Euclidean annuli.  It is intended for particle fields
obtained from continuous ABP simulations, where distance from a particle center
is radial rather than square-shell based.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def add_spatial_channels(x: torch.Tensor) -> torch.Tensor:
    """Append normalized x/y coordinate channels to ``[B, C, Lx, Ly]``."""
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
    return torch.cat(
        [
            x,
            xx.view(1, 1, Lx, Ly).expand(B, 1, Lx, Ly),
            yy.view(1, 1, Lx, Ly).expand(B, 1, Lx, Ly),
        ],
        dim=1,
    )


def euclidean_annulus_offsets(
    k: int,
    *,
    shell_width: float = 1.0,
    shell_offset: float = 0.0,
) -> Tuple[List[Tuple[int, int]], float, float]:
    """Return integer offsets whose Euclidean radius lies in annulus k.

    For ``k > 0`` the annulus is

        shell_offset + (k - 1/2) shell_width < r <=
        shell_offset + (k + 1/2) shell_width.

    The origin is excluded.  ``k=0`` is handled by the local branch and should
    not call this helper.
    """
    if k <= 0:
        raise ValueError("Euclidean annuli are defined for k > 0.")
    if shell_width <= 0:
        raise ValueError("shell_width must be positive.")

    r_inner = max(0.0, shell_offset + (k - 0.5) * shell_width)
    r_outer = shell_offset + (k + 0.5) * shell_width
    radius = int(math.ceil(r_outer))
    offsets: List[Tuple[int, int]] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            r = math.sqrt(float(dy * dy + dx * dx))
            if r_inner < r <= r_outer + 1e-12:
                offsets.append((dy, dx))

    if not offsets:
        raise ValueError(
            f"Empty Euclidean annulus for k={k}, shell_width={shell_width}, "
            f"shell_offset={shell_offset}."
        )
    return offsets, r_inner, r_outer


def annulus_kernel_from_offsets(offsets: List[Tuple[int, int]]):
    """Build a mask kernel and radius from integer offsets."""
    radius = max(max(abs(dy), abs(dx)) for dy, dx in offsets)
    kernel_size = 2 * radius + 1
    mask = torch.zeros(1, 1, kernel_size, kernel_size, dtype=torch.float32)
    for dy, dx in offsets:
        mask[0, 0, radius + dy, radius + dx] = 1.0
    return mask, radius


class EuclideanShellRelativeConv2d(nn.Module):
    """Learned convolution over a Euclidean annulus of relative differences."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int,
        *,
        shell_width: float = 1.0,
        shell_offset: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        offsets, r_inner, r_outer = euclidean_annulus_offsets(
            k, shell_width=shell_width, shell_offset=shell_offset
        )
        mask, radius = annulus_kernel_from_offsets(offsets)
        self.k = k
        self.r_inner = r_inner
        self.r_outer = r_outer
        self.radius = radius
        self.kernel_size = 2 * radius + 1

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, self.kernel_size, self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.register_buffer("mask", mask)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x_center: torch.Tensor) -> torch.Tensor:
        x_pad = F.pad(x_center, (self.radius, self.radius, self.radius, self.radius), mode="circular")
        masked_weight = self.weight * self.mask
        neighbor_sum = F.conv2d(x_pad, masked_weight, bias=None)
        center_weight = masked_weight.sum(dim=(2, 3), keepdim=True)
        center_sum = F.conv2d(x_center, center_weight, bias=None)
        out = neighbor_sum - center_sum
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out


class EuclideanShellAbsoluteConv2d(nn.Module):
    """Learned convolution over a Euclidean annulus of absolute field values."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int,
        *,
        shell_width: float = 1.0,
        shell_offset: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        offsets, r_inner, r_outer = euclidean_annulus_offsets(
            k, shell_width=shell_width, shell_offset=shell_offset
        )
        mask, radius = annulus_kernel_from_offsets(offsets)
        self.k = k
        self.r_inner = r_inner
        self.r_outer = r_outer
        self.radius = radius
        self.kernel_size = 2 * radius + 1

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, self.kernel_size, self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.register_buffer("mask", mask)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x_center: torch.Tensor) -> torch.Tensor:
        x_pad = F.pad(x_center, (self.radius, self.radius, self.radius, self.radius), mode="circular")
        masked_weight = self.weight * self.mask
        return F.conv2d(x_pad, masked_weight, bias=self.bias)


class FixedEuclideanShellWeightedSum2d(nn.Module):
    """Fixed channelwise weighted sum over a Euclidean annulus."""

    def __init__(
        self,
        k: int,
        *,
        shell_width: float = 1.0,
        shell_offset: float = 0.0,
        weight_normalization: str = "none",
        relative: bool = False,
    ):
        super().__init__()
        offsets, r_inner, r_outer = euclidean_annulus_offsets(
            k, shell_width=shell_width, shell_offset=shell_offset
        )
        mask, radius = annulus_kernel_from_offsets(offsets)
        self.k = k
        self.r_inner = r_inner
        self.r_outer = r_outer
        self.radius = radius
        self.relative = relative

        weight = mask.clone()
        if weight_normalization in {"mean", "average"}:
            weight = weight / weight.sum().clamp_min(1.0)
        elif weight_normalization in {"none", "sum", "raw"}:
            pass
        else:
            raise ValueError("weight_normalization must be 'none'/'sum'/'raw' or 'mean'/'average'.")

        self.register_buffer("kernel", weight)
        self.register_buffer("weight_sum", weight.sum())
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

    def forward(self, x_center: torch.Tensor) -> torch.Tensor:
        channels = x_center.shape[1]
        x_pad = F.pad(x_center, (self.radius, self.radius, self.radius, self.radius), mode="circular")
        kernel = self.kernel.expand(channels, 1, -1, -1)
        out = F.conv2d(x_pad, kernel, groups=channels)
        if self.relative:
            out = out - self.weight_sum * x_center
        return out


class LearnedEuclideanShellWeightedSum2d(nn.Module):
    """Learned depthwise weighted sum over a Euclidean annulus."""

    def __init__(
        self,
        in_channels: int,
        k: int,
        *,
        shell_width: float = 1.0,
        shell_offset: float = 0.0,
        weight_normalization: str = "none",
        relative: bool = False,
    ):
        super().__init__()
        offsets, r_inner, r_outer = euclidean_annulus_offsets(
            k, shell_width=shell_width, shell_offset=shell_offset
        )
        mask, radius = annulus_kernel_from_offsets(offsets)
        self.in_channels = in_channels
        self.k = k
        self.r_inner = r_inner
        self.r_outer = r_outer
        self.radius = radius
        self.relative = relative

        base = mask.clone()
        if weight_normalization in {"mean", "average"}:
            base = base / base.sum().clamp_min(1.0)
        elif weight_normalization in {"none", "sum", "raw"}:
            pass
        else:
            raise ValueError("weight_normalization must be 'none'/'sum'/'raw' or 'mean'/'average'.")

        kernel = base.expand(in_channels, 1, -1, -1).clone()
        self.weight = nn.Parameter(kernel)
        self.register_buffer("mask", mask)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

    def forward(self, x_center: torch.Tensor) -> torch.Tensor:
        x_pad = F.pad(x_center, (self.radius, self.radius, self.radius, self.radius), mode="circular")
        masked_weight = self.weight * self.mask
        out = F.conv2d(x_pad, masked_weight, groups=self.in_channels)
        if self.relative:
            weight_sum = masked_weight.sum(dim=(2, 3)).view(1, self.in_channels, 1, 1)
            out = out - weight_sum * x_center
        return out


class _ABPEuclideanShellForceBranch2D(nn.Module):
    """One local or Euclidean-annulus shell-force branch."""

    def __init__(
        self,
        k: int,
        n_components: int,
        center_channels: int,
        hidden_channels: int,
        *,
        n_hidden: int = 2,
        shell_center_mode: str = "relative_only",
        shell_force_bias: bool = False,
        shell_relative_mode: str = "learned_absolute",
        shell_weight_normalization: str = "none",
        shell_width: float = 1.0,
        shell_offset: float = 0.0,
        activation: str = "tanh",
    ):
        super().__init__()
        self.k = k
        self.shell_center_mode = shell_center_mode
        self.shell_relative_mode = shell_relative_mode

        if k == 0 or shell_center_mode in {"add", "gated"}:
            self.center_proj = nn.Conv2d(
                center_channels,
                hidden_channels,
                kernel_size=1,
                bias=(k == 0 or shell_force_bias),
            )
        else:
            self.center_proj = None

        relative_learned_modes = {"learned", "learned_relative", "full_learned_relative"}
        absolute_learned_modes = {"learned_absolute", "absolute_learned", "full_learned_absolute"}
        relative_fixed_modes = {"fixed_sum", "fixed", "relative_sum", "relative"}
        absolute_fixed_modes = {"absolute_sum", "absolute", "abs_sum", "abs"}
        relative_learned_sum_modes = {
            "learned_sum",
            "learned_relative_sum",
            "learned_channelwise_sum",
            "learned_channelwise",
        }
        absolute_learned_sum_modes = {
            "learned_absolute_sum",
            "learned_abs_sum",
            "learned_channelwise_absolute_sum",
            "learned_channelwise_absolute",
        }

        self.r_inner = 0.0
        self.r_outer = 0.0
        if k > 0 and shell_relative_mode in absolute_learned_modes:
            self.rel_proj = EuclideanShellAbsoluteConv2d(
                n_components,
                hidden_channels,
                k=k,
                shell_width=shell_width,
                shell_offset=shell_offset,
                bias=shell_force_bias,
            )
            self.rel_lift = None
        elif k > 0 and shell_relative_mode in relative_learned_modes:
            self.rel_proj = EuclideanShellRelativeConv2d(
                n_components,
                hidden_channels,
                k=k,
                shell_width=shell_width,
                shell_offset=shell_offset,
                bias=False,
            )
            self.rel_lift = None
        elif k > 0 and shell_relative_mode in absolute_fixed_modes:
            self.rel_proj = FixedEuclideanShellWeightedSum2d(
                k,
                shell_width=shell_width,
                shell_offset=shell_offset,
                weight_normalization=shell_weight_normalization,
                relative=False,
            )
            self.rel_lift = nn.Conv2d(n_components, hidden_channels, kernel_size=1, bias=shell_force_bias)
        elif k > 0 and shell_relative_mode in relative_fixed_modes:
            self.rel_proj = FixedEuclideanShellWeightedSum2d(
                k,
                shell_width=shell_width,
                shell_offset=shell_offset,
                weight_normalization=shell_weight_normalization,
                relative=True,
            )
            self.rel_lift = nn.Conv2d(n_components, hidden_channels, kernel_size=1, bias=shell_force_bias)
        elif k > 0 and shell_relative_mode in absolute_learned_sum_modes:
            self.rel_proj = LearnedEuclideanShellWeightedSum2d(
                n_components,
                k,
                shell_width=shell_width,
                shell_offset=shell_offset,
                weight_normalization=shell_weight_normalization,
                relative=False,
            )
            self.rel_lift = nn.Conv2d(n_components, hidden_channels, kernel_size=1, bias=shell_force_bias)
        elif k > 0 and shell_relative_mode in relative_learned_sum_modes:
            self.rel_proj = LearnedEuclideanShellWeightedSum2d(
                n_components,
                k,
                shell_width=shell_width,
                shell_offset=shell_offset,
                weight_normalization=shell_weight_normalization,
                relative=True,
            )
            self.rel_lift = nn.Conv2d(n_components, hidden_channels, kernel_size=1, bias=shell_force_bias)
        elif k > 0:
            raise ValueError(
                "shell_relative_mode must include one of learned_absolute, learned, "
                "absolute_sum, fixed_sum, learned_absolute_sum, or learned_sum."
            )
        else:
            self.rel_proj = None
            self.rel_lift = None

        if self.rel_proj is not None:
            self.r_inner = float(self.rel_proj.r_inner)
            self.r_outer = float(self.rel_proj.r_outer)

        self.center_gate = (
            nn.Conv2d(center_channels, hidden_channels, kernel_size=1, bias=True)
            if k > 0 and shell_center_mode == "gated"
            else None
        )

        if activation == "elu":
            self.act = nn.ELU(inplace=True)
            post_act = lambda: nn.ELU(inplace=True)
        elif activation == "relu":
            self.act = nn.ReLU(inplace=True)
            post_act = lambda: nn.ReLU(inplace=True)
        elif activation == "tanh":
            self.act = nn.Tanh()
            post_act = nn.Tanh
        elif activation in {"identity", "linear", "none"}:
            self.act = nn.Identity()
            post_act = nn.Identity
        else:
            raise ValueError("activation must be one of 'elu', 'relu', 'tanh', or 'identity'.")

        layers = []
        for _ in range(max(n_hidden - 1, 0)):
            layers.append(
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=1,
                    bias=(k == 0 or shell_force_bias),
                )
            )
            layers.append(post_act())
        layers.append(
            nn.Conv2d(
                hidden_channels,
                n_components,
                kernel_size=1,
                bias=(k == 0 or shell_force_bias),
            )
        )
        self.force_head = nn.Sequential(*layers)

    def forward(self, x_mid: torch.Tensor, center_features: torch.Tensor) -> torch.Tensor:
        if self.k == 0:
            h = self.center_proj(center_features)
        else:
            h = self.rel_proj(x_mid)
            if self.rel_lift is not None:
                h = self.rel_lift(h)
            if self.shell_center_mode == "add":
                h = h + self.center_proj(center_features)
            elif self.shell_center_mode == "gated":
                h = h * (1.0 + torch.tanh(self.center_gate(center_features)))
        h = self.act(h)
        return self.force_head(h)


class ABPEuclideanShellForceCNEEP2D(nn.Module):
    """Short-time ShellForce CNEEP with Euclidean annuli for ABP fields."""

    def __init__(self, opt):
        super().__init__()
        self.positional = getattr(opt, "positional", False)
        self.beta = getattr(opt, "beta", 1.0)
        self.max_distance = opt.max_distance
        self.include_k0 = getattr(opt, "include_k0", True)
        self.n_components = getattr(opt, "n_components", 1)
        self.hidden_channels = opt.n_channel
        self.n_hidden = getattr(opt, "n_hidden", 2)
        self.start_k = 0 if self.include_k0 else 1

        self.shell_center_mode = getattr(opt, "shell_center_mode", "relative_only")
        self.shell_force_bias = getattr(opt, "shell_force_bias", False)
        self.shell_relative_mode = getattr(opt, "shell_relative_mode", "learned_absolute")
        self.shell_weight_normalization = getattr(opt, "shell_weight_normalization", "none")
        self.shell_width = float(getattr(opt, "shell_width", 1.0))
        self.shell_offset = float(getattr(opt, "shell_offset", 0.0))
        self.activation = getattr(opt, "shell_force_activation", "tanh")

        if self.shell_center_mode not in {"relative_only", "add", "gated"}:
            raise ValueError("shell_center_mode must be one of {'relative_only', 'add', 'gated'}.")

        center_channels = self.n_components + (2 if self.positional else 0)
        self.branches = nn.ModuleList(
            [
                _ABPEuclideanShellForceBranch2D(
                    k=k,
                    n_components=self.n_components,
                    center_channels=center_channels,
                    hidden_channels=self.hidden_channels,
                    n_hidden=self.n_hidden,
                    shell_center_mode=self.shell_center_mode,
                    shell_force_bias=self.shell_force_bias,
                    shell_relative_mode=self.shell_relative_mode,
                    shell_weight_normalization=self.shell_weight_normalization,
                    shell_width=self.shell_width,
                    shell_offset=self.shell_offset,
                    activation=self.activation,
                )
                for k in range(self.start_k, self.max_distance + 1)
            ]
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def shell_bounds(self):
        """Return annulus bounds in pixel units for each branch."""
        bounds = []
        for branch in self.branches:
            if branch.k == 0:
                bounds.append((0.0, 0.0))
            else:
                bounds.append((branch.r_inner, branch.r_outer))
        return bounds

    def _split_short_time_pair(self, x: torch.Tensor):
        if x.shape[1] != 2:
            raise ValueError("ABPEuclideanShellForceCNEEP2D requires seq_len=2.")
        if self.n_components > 1:
            if x.dim() != 5:
                raise ValueError("Expected x with shape [B, 2, C, Lx, Ly].")
            x0 = x[:, 0]
            x1 = x[:, 1]
        else:
            if x.dim() != 4:
                raise ValueError("Expected x with shape [B, 2, Lx, Ly].")
            x0 = x[:, 0].unsqueeze(1)
            x1 = x[:, 1].unsqueeze(1)
        return 0.5 * (x0 + x1), x1 - x0

    def forward(
        self,
        x: torch.Tensor,
        return_maps: bool = False,
        return_forces: bool = False,
    ) -> torch.Tensor:
        x_mid, dx = self._split_short_time_pair(x)
        center_features = add_spatial_channels(x_mid) if self.positional else x_mid

        J_list = []
        map_list = []
        force_list = []
        for branch in self.branches:
            force = branch(x_mid, center_features)
            local_ep = (force * dx).sum(dim=1, keepdim=True)
            if return_forces:
                force_list.append(force.unsqueeze(1))
            if return_maps:
                map_list.append(local_ep)
            else:
                J_list.append(local_ep.mean(dim=(2, 3)))

        if return_maps:
            maps = torch.cat(map_list, dim=1)
            if return_forces:
                return maps, torch.cat(force_list, dim=1)
            return maps

        J = torch.cat(J_list, dim=1)
        if return_forces:
            return torch.cat(force_list, dim=1)
        return J


class ABPEuclideanShellForceCNEEP2D_Tanh(ABPEuclideanShellForceCNEEP2D):
    """Euclidean-annulus ShellForce variant with tanh branch activations."""

    def __init__(self, opt):
        opt.shell_force_activation = "tanh"
        super().__init__(opt)
