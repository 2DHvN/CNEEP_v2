import torch
import torch.nn as nn
import torch.nn.functional as F


class PeriodicPad2d(nn.Module):
    """Circular padding for tensors of shape [B, C, Lx, Ly]."""

    def __init__(self, padding):
        super(PeriodicPad2d, self).__init__()
        if isinstance(padding, int):
            self.padding = (padding, padding, padding, padding)
        else:
            self.padding = padding

    def forward(self, x):
        return F.pad(x, self.padding, mode="circular")


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


class ShellRelativeConv2d(nn.Module):
    """
    Efficient exclusive Chebyshev-shell relative convolution.

    For k > 0, this computes
        sum_delta W_delta (x[i + delta] - x[i])
    over offsets with max(abs(dy), abs(dx)) == k.
    """

    def __init__(self, in_channels: int, out_channels: int, k: int, bias: bool = True):
        super(ShellRelativeConv2d, self).__init__()
        if k <= 0:
            raise ValueError("ShellRelativeConv2d is only defined for k > 0.")

        self.k = k
        self.kernel_size = 2 * k + 1
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, self.kernel_size, self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        mask = torch.zeros(1, 1, self.kernel_size, self.kernel_size)
        mask[0, 0, 0, :] = 1.0
        mask[0, 0, 2 * k, :] = 1.0
        mask[0, 0, :, 0] = 1.0
        mask[0, 0, :, 2 * k] = 1.0
        mask[0, 0, k, k] = 0.0
        self.register_buffer("mask", mask)

        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x_center: torch.Tensor) -> torch.Tensor:
        x_pad = F.pad(x_center, (self.k, self.k, self.k, self.k), mode="circular")
        masked_weight = self.weight * self.mask

        neighbor_sum = F.conv2d(x_pad, masked_weight, bias=None)
        center_weight = masked_weight.sum(dim=(2, 3), keepdim=True)
        center_sum = F.conv2d(x_center, center_weight, bias=None)
        out = neighbor_sum - center_sum

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out


class ShellAbsoluteConv2d(nn.Module):
    """
    Efficient exclusive Chebyshev-shell absolute convolution.

    For k > 0, this computes
        sum_delta W_delta x[i + delta]
    over offsets with max(abs(dy), abs(dx)) == k, without subtracting the
    center value.
    """

    def __init__(self, in_channels: int, out_channels: int, k: int, bias: bool = True):
        super(ShellAbsoluteConv2d, self).__init__()
        if k <= 0:
            raise ValueError("ShellAbsoluteConv2d is only defined for k > 0.")

        self.k = k
        self.kernel_size = 2 * k + 1
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, self.kernel_size, self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        mask = torch.zeros(1, 1, self.kernel_size, self.kernel_size)
        mask[0, 0, 0, :] = 1.0
        mask[0, 0, 2 * k, :] = 1.0
        mask[0, 0, :, 0] = 1.0
        mask[0, 0, :, 2 * k] = 1.0
        mask[0, 0, k, k] = 0.0
        self.register_buffer("mask", mask)

        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x_center: torch.Tensor) -> torch.Tensor:
        x_pad = F.pad(x_center, (self.k, self.k, self.k, self.k), mode="circular")
        masked_weight = self.weight * self.mask
        return F.conv2d(x_pad, masked_weight, bias=self.bias)


class FixedShellWeightedSum2d(nn.Module):
    """
    Fixed scalar-weight shell operator in the original field channels.

    If relative=True, computes
        sum_delta w_delta (x[i + delta] - x[i]).
    If relative=False, computes
        sum_delta w_delta x[i + delta].
    over offsets with max(abs(dy), abs(dx)) == k.
    """

    def __init__(self, k: int, weight_normalization: str = "none", relative: bool = True):
        super(FixedShellWeightedSum2d, self).__init__()
        if k <= 0:
            raise ValueError("FixedShellWeightedSum2d is only defined for k > 0.")
        self.k = k
        self.weight_normalization = weight_normalization
        self.relative = relative

        offsets = [
            (dy, dx)
            for dy in range(-k, k + 1)
            for dx in range(-k, k + 1)
            if max(abs(dy), abs(dx)) == k
        ]
        weight = torch.ones(len(offsets), dtype=torch.float32)
        if weight_normalization in {"mean", "average"}:
            weight = weight / weight.sum()
        elif weight_normalization in {"none", "sum", "raw"}:
            pass
        else:
            raise ValueError("weight_normalization must be one of {'none', 'sum', 'raw', 'mean', 'average'}.")

        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        self.register_buffer("weight", weight)
        kernel = torch.zeros(1, 1, 2 * k + 1, 2 * k + 1, dtype=torch.float32)
        for idx, (dy, dx) in enumerate(offsets):
            kernel[0, 0, k + dy, k + dx] = weight[idx]
        self.register_buffer("kernel", kernel)
        self.register_buffer("weight_sum", weight.sum())

    def forward(self, x_center: torch.Tensor) -> torch.Tensor:
        channels = x_center.shape[1]
        x_pad = F.pad(x_center, (self.k, self.k, self.k, self.k), mode="circular")
        kernel = self.kernel.expand(channels, 1, -1, -1)
        out = F.conv2d(x_pad, kernel, groups=channels)
        if self.relative:
            out = out - self.weight_sum * x_center
        return out


class LearnedShellWeightedSum2d(nn.Module):
    """
    Learnable depthwise shell operator in the original field channels.

    This keeps the output channel count equal to the input channel count, but
    learns a separate shell-offset weighted sum for each channel.
    """

    def __init__(
        self,
        in_channels: int,
        k: int,
        weight_normalization: str = "none",
        relative: bool = True,
    ):
        super(LearnedShellWeightedSum2d, self).__init__()
        if k <= 0:
            raise ValueError("LearnedShellWeightedSum2d is only defined for k > 0.")
        self.in_channels = in_channels
        self.k = k
        self.weight_normalization = weight_normalization
        self.relative = relative

        offsets = [
            (dy, dx)
            for dy in range(-k, k + 1)
            for dx in range(-k, k + 1)
            if max(abs(dy), abs(dx)) == k
        ]
        base_weight = torch.ones(len(offsets), dtype=torch.float32)
        if weight_normalization in {"mean", "average"}:
            base_weight = base_weight / base_weight.sum()
        elif weight_normalization in {"none", "sum", "raw"}:
            pass
        else:
            raise ValueError("weight_normalization must be one of {'none', 'sum', 'raw', 'mean', 'average'}.")

        kernel = torch.zeros(in_channels, 1, 2 * k + 1, 2 * k + 1, dtype=torch.float32)
        mask = torch.zeros(1, 1, 2 * k + 1, 2 * k + 1, dtype=torch.float32)
        for idx, (dy, dx) in enumerate(offsets):
            kernel[:, 0, k + dy, k + dx] = base_weight[idx]
            mask[0, 0, k + dy, k + dx] = 1.0

        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        self.register_buffer("mask", mask)
        self.weight = nn.Parameter(kernel)

    def forward(self, x_center: torch.Tensor) -> torch.Tensor:
        x_pad = F.pad(x_center, (self.k, self.k, self.k, self.k), mode="circular")
        masked_weight = self.weight * self.mask
        out = F.conv2d(x_pad, masked_weight, groups=self.in_channels)
        if self.relative:
            weight_sum = masked_weight.sum(dim=(2, 3)).view(1, self.in_channels, 1, 1)
            out = out - weight_sum * x_center
        return out


class _ShellForceBranch2D(nn.Module):
    """
    One short-time shell-force branch.

    k = 0 learns a local force F_0(x_mid).
    k > 0 learns a nonlinear force from exclusive shell messages.
    """

    def __init__(
        self,
        k: int,
        n_components: int,
        center_channels: int,
        hidden_channels: int,
        n_hidden: int = 2,
        shell_center_mode: str = "relative_only",
        shell_force_bias: bool = False,
        shell_relative_mode: str = "fixed_sum",
        shell_weight_normalization: str = "none",
        activation: str = "elu",
    ):
        super(_ShellForceBranch2D, self).__init__()
        self.k = k
        self.shell_center_mode = shell_center_mode
        self.shell_relative_mode = shell_relative_mode
        self.activation_name = activation

        if k == 0 or shell_center_mode in {"add", "gated"}:
            self.center_proj = nn.Conv2d(
                center_channels,
                hidden_channels,
                kernel_size=1,
                bias=(k == 0 or shell_force_bias),
            )
        else:
            self.center_proj = None

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

        relative_learned_modes = {"learned", "learned_relative", "full_learned_relative"}
        absolute_learned_modes = {"learned_absolute", "absolute_learned", "full_learned_absolute"}

        if k > 0 and shell_relative_mode in relative_learned_modes:
            self.rel_proj = ShellRelativeConv2d(n_components, hidden_channels, k=k, bias=False)
            self.rel_lift = None
        elif k > 0 and shell_relative_mode in absolute_learned_modes:
            self.rel_proj = ShellAbsoluteConv2d(n_components, hidden_channels, k=k, bias=shell_force_bias)
            self.rel_lift = None
        elif k > 0 and shell_relative_mode in relative_fixed_modes:
            self.rel_proj = FixedShellWeightedSum2d(
                k=k,
                weight_normalization=shell_weight_normalization,
                relative=True,
            )
            self.rel_lift = nn.Conv2d(
                n_components,
                hidden_channels,
                kernel_size=1,
                bias=shell_force_bias,
            )
        elif k > 0 and shell_relative_mode in absolute_fixed_modes:
            self.rel_proj = FixedShellWeightedSum2d(
                k=k,
                weight_normalization=shell_weight_normalization,
                relative=False,
            )
            self.rel_lift = nn.Conv2d(
                n_components,
                hidden_channels,
                kernel_size=1,
                bias=shell_force_bias,
            )
        elif k > 0 and shell_relative_mode in relative_learned_sum_modes:
            self.rel_proj = LearnedShellWeightedSum2d(
                in_channels=n_components,
                k=k,
                weight_normalization=shell_weight_normalization,
                relative=True,
            )
            self.rel_lift = nn.Conv2d(
                n_components,
                hidden_channels,
                kernel_size=1,
                bias=shell_force_bias,
            )
        elif k > 0 and shell_relative_mode in absolute_learned_sum_modes:
            self.rel_proj = LearnedShellWeightedSum2d(
                in_channels=n_components,
                k=k,
                weight_normalization=shell_weight_normalization,
                relative=False,
            )
            self.rel_lift = nn.Conv2d(
                n_components,
                hidden_channels,
                kernel_size=1,
                bias=shell_force_bias,
            )
        elif k > 0:
            raise ValueError(
                "shell_relative_mode must be one of "
                "{'fixed_sum', 'relative_sum', 'absolute_sum', 'learned', 'learned_absolute', "
                "'learned_sum', 'learned_absolute_sum'}."
            )
        else:
            self.rel_proj = None
            self.rel_lift = None

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
            raise ValueError("activation must be one of {'elu', 'relu', 'tanh', 'identity'}.")

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


class MultiScaleShellForceCNEEP2D(nn.Module):
    """
    Short-time shell-force CNEEP model.

    The model decomposes entropy production into shell-wise force increments:
        x_mid = 0.5 * (x_t + x_{t+dt})
        dx    = x_{t+dt} - x_t
        dS_k  = sum_i F_k(x_mid)_i dot dx_i

    This makes every branch antisymmetric under time reversal by construction
    and gives J_k the interpretation of the EP increment generated by the
    learned k-shell force.
    """

    def __init__(self, opt):
        super(MultiScaleShellForceCNEEP2D, self).__init__()

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
        self.shell_relative_mode = getattr(opt, "shell_relative_mode", "fixed_sum")
        self.shell_weight_normalization = getattr(opt, "shell_weight_normalization", "none")
        self.activation = getattr(opt, "shell_force_activation", "elu")
        if self.shell_center_mode not in {"relative_only", "add", "gated"}:
            raise ValueError("shell_center_mode must be one of {'relative_only', 'add', 'gated'}.")

        center_channels = self.n_components + (2 if self.positional else 0)
        self.branches = nn.ModuleList(
            [
                _ShellForceBranch2D(
                    k=k,
                    n_components=self.n_components,
                    center_channels=center_channels,
                    hidden_channels=self.hidden_channels,
                    n_hidden=self.n_hidden,
                    shell_center_mode=self.shell_center_mode,
                    shell_force_bias=self.shell_force_bias,
                    shell_relative_mode=self.shell_relative_mode,
                    shell_weight_normalization=self.shell_weight_normalization,
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

    def _split_short_time_pair(self, x: torch.Tensor):
        if x.shape[1] != 2:
            raise ValueError("MultiScaleShellForceCNEEP2D requires seq_len=2.")

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

        x_mid = 0.5 * (x0 + x1)
        dx = x1 - x0
        return x_mid, dx

    def forward(
        self,
        x: torch.Tensor,
        return_maps: bool = False,
        return_forces: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            [B, 2, C, Lx, Ly] when n_components > 1, else [B, 2, Lx, Ly].
        return_maps:
            If True, return [B, K, Lx, Ly] local EP maps.
            If False, return [B, K] shell EP increments averaged over space.
        return_forces:
            If True, return learned force maps [B, K, C, Lx, Ly]. If both
            return_maps and return_forces are True, returns (maps, forces).
        """
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
                J_k = local_ep.mean(dim=(2, 3))
                J_list.append(J_k)

        if return_maps:
            maps = torch.cat(map_list, dim=1)
            if return_forces:
                return maps, torch.cat(force_list, dim=1)
            return maps

        J = torch.cat(J_list, dim=1)
        if return_forces:
            return torch.cat(force_list, dim=1)
        return J


class MultiScaleShellForceCNEEP2D_Tanh(MultiScaleShellForceCNEEP2D):
    """ShellForce variant with tanh activations in every branch."""

    def __init__(self, opt):
        opt.shell_force_activation = "tanh"
        super(MultiScaleShellForceCNEEP2D_Tanh, self).__init__(opt)
