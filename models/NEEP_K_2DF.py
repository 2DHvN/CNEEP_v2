import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────
# Periodic padding for 2D inputs
# ──────────────────────────────────────────────────────────────
class PeriodicPad2d(nn.Module):
    """Circular (periodic) padding for 2D tensors of shape [B, C, Lx, Ly]."""

    def __init__(self, padding):
        super(PeriodicPad2d, self).__init__()
        if isinstance(padding, int):
            self.padding = (padding, padding, padding, padding)
        else:
            self.padding = padding

    def forward(self, x):
        return F.pad(x, self.padding, mode='circular')


# ──────────────────────────────────────────────────────────────
# Exclusive Masked 2D Convolution
# ──────────────────────────────────────────────────────────────
class ExclusiveMaskedConv2d(nn.Module):
    """Conv2d whose kernel is masked so that:
      - k=0: only the center pixel is learnable.
      - k>0: only the border (perimeter) of the (2k+1)x(2k+1) kernel
              is learnable; the center pixel is EXCLUDED.

    When x_center is provided (k>0), computes the *relative* convolution:
        out_i = Σ_δ w_δ (X_{i+δ} - X_i)
    by decomposing into:
        Σ_δ w_δ X_{i+δ}  −  (Σ_δ w_δ) · X_i
    i.e.  masked_conv(x_pad)  −  S · x_center,
    where S[co,ci] = Σ_{h,w} mask·weight  is an effective 1×1 projection.

    This ensures each k-branch captures exclusively the *relative*
    contribution from its specific correlation distance.
    """

    def __init__(self, in_channels: int, out_channels: int, k: int,
                 bias: bool = True):
        super(ExclusiveMaskedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.kernel_size = 2 * k + 1

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, self.kernel_size, self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        # Build the exclusive binary mask
        mask = torch.zeros(1, 1, self.kernel_size, self.kernel_size)
        if k == 0:
            # k=0: center only
            mask[0, 0, 0, 0] = 1.0
        else:
            # k>0: border only, center excluded
            mask[0, 0, 0, :] = 1.0        # Top edge
            mask[0, 0, 2 * k, :] = 1.0    # Bottom edge
            mask[0, 0, :, 0] = 1.0        # Left edge
            mask[0, 0, :, 2 * k] = 1.0    # Right edge
            mask[0, 0, k, k] = 0.0        # Explicitly exclude center

        self.register_buffer("mask", mask)

        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor,
                x_center: torch.Tensor = None) -> torch.Tensor:
        """Compute relative masked convolution.

        Parameters
        ----------
        x : [B, C_in, Lx+2k, Ly+2k]  periodically padded input (k>0)
            or [B, C_in, Lx, Ly] for k=0.
        x_center : [B, C_in, Lx, Ly]  original unpadded input.
        """
        if self.k == 0:
            # k=0: just normal conv2d on the center pixel
            out = F.conv2d(x, self.weight, self.bias)
            return out

        # k > 0: Relative convolution: Σ_δ w_δ (X_{i+δ} - X_i)
        B, C_in, Lx, Ly = x_center.shape
        out = torch.zeros(B, self.out_channels, Lx, Ly, device=x.device, dtype=x.dtype)

        # Loop over the border offsets of the (2k+1)x(2k+1) kernel
        for dy in range(-self.k, self.k + 1):
            for dx in range(-self.k, self.k + 1):
                # Only border offsets (Chebyshev distance == k)
                if max(abs(dy), abs(dx)) == self.k:
                    # Slice of the padded input corresponding to offset (dy, dx)
                    x_shifted = x[:, :, self.k + dy : self.k + dy + Lx, self.k + dx : self.k + dx + Ly]
                    # Compute relative displacement (X_{i+δ} - X_i)
                    x_rel = x_shifted - x_center
                    # Extract the corresponding 1x1 weight slice [C_out, C_in, 1, 1]
                    w_delta = self.weight[:, :, self.k + dy, self.k + dx].unsqueeze(-1).unsqueeze(-1)
                    # Accumulate: out_i += w_δ * (X_{i+δ} - X_i)
                    out = out + F.conv2d(x_rel, w_delta)

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)

        return out


# ──────────────────────────────────────────────────────────────
# Single K-branch (exclusive distance)
# ──────────────────────────────────────────────────────────────
class _KBranch2D(nn.Module):
    """One branch for a specific distance *k* with exclusive masking.

    Pipeline (per branch, k>0):
        x_center = x
        PeriodicPad2d(k) → ExclusiveMaskedConv2d(x_pad, x_center)
        → ELU → [Conv2d(1×1) → ELU] × (n_hidden - 1)
        → Conv2d(1×1)  →  Local EP 2D Map  [B, 1, Lx, Ly]

    For k>0 the masked conv computes the relative interaction
        Σ_δ w_δ (X_{i+δ} − X_i)
    so each branch sees only displacement from the center.
    """

    def __init__(self, k: int, in_channels: int, hidden_channels: int,
                 n_hidden: int = 2, n_components: int = 1):
        super(_KBranch2D, self).__init__()
        self.k = k

        # Periodic padding (only for k > 0)
        self.pad = PeriodicPad2d(k) if k > 0 else None

        # First layer: exclusive masked conv (supports relative mode)
        self.masked_conv = ExclusiveMaskedConv2d(
            in_channels, hidden_channels, k=k,
        )
        self.act = nn.ELU(inplace=True)

        # Remaining 1×1 conv layers
        post_layers: list = []
        for _ in range(n_hidden - 1):
            post_layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1))
            post_layers.append(nn.ELU(inplace=True))

        # Output to n_components channel (Force vector per position)
        post_layers.append(nn.Conv2d(hidden_channels, n_components, kernel_size=1))
        self.post_net = nn.Sequential(*post_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, C_in, Lx, Ly]"""
        x_center = x                                     # unpadded center
        x_padded = self.pad(x) if self.pad is not None else x

        # Relative masked convolution: Σ_δ w_δ (X_{i+δ} − X_i)
        h = self.masked_conv(x_padded, x_center=x_center)
        h = self.act(h)

        return self.post_net(h)


# ──────────────────────────────────────────────────────────────
# Spatial channel helper (optional)
# ──────────────────────────────────────────────────────────────
def add_spatial_channels(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B, C, Lx, Ly]
    Appends normalized x and y coordinate channels.
    Returns: [B, C+2, Lx, Ly]
    """
    B, _, Lx, Ly = x.shape
    device = x.device

    xx, yy = torch.meshgrid(torch.arange(Lx, device=device),
                            torch.arange(Ly, device=device),
                            indexing='ij')

    xx = xx.float() / (Lx - 1)
    yy = yy.float() / (Ly - 1)

    xx = xx.unsqueeze(0).unsqueeze(0).expand(B, 1, Lx, Ly)
    yy = yy.unsqueeze(0).unsqueeze(0).expand(B, 1, Lx, Ly)

    return torch.cat([x, xx, yy], dim=1)


# ──────────────────────────────────────────────────────────────
# 2D Multi-Scale K_2D Model (Exclusive Branches)
# ──────────────────────────────────────────────────────────────
class MultiScaleK_2DF(nn.Module):
    """
    Parallel branch architecture for 2D spatial entropy production estimation
    with exclusive distance masks.

    Unlike MultiScaleCNEEP2D where each k-branch sees both the center
    and the border, here each k>0 branch sees ONLY its distance-k border
    pixels.  This removes information overlap so each branch captures
    exclusively the marginal contribution from its correlation distance.
    """

    def __init__(self, opt):
        super(MultiScaleK_2DF, self).__init__()

        self.positional = opt.positional
        self.beta = opt.beta
        self.max_distance = opt.max_distance

        self.n_components = getattr(opt, "n_components", 1)
        in_channels = opt.seq_len * self.n_components + (2 if opt.positional else 0)
        hidden_channels = opt.n_channel
        n_hidden = getattr(opt, "n_hidden", 2)

        self.include_k0 = getattr(opt, "include_k0", True)
        start_k = 0 if self.include_k0 else 1

        # Build K independent branches with exclusive masks
        self.branches = nn.ModuleList([
            _KBranch2D(
                k=k,
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                n_hidden=n_hidden,
                n_components=self.n_components
            )
            for k in range(start_k, self.max_distance + 1)
        ])

        # Kaiming init for all Conv2d inside branches
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ----- shared backbone: produces the local EP map ----- #
    def _local_map(self, x: torch.Tensor,
                   branch: _KBranch2D) -> torch.Tensor:
        """Run one branch on input *x* and return [B, 1, Lx, Ly] local EP map."""
        return branch(x)

    # ----- forward ----- #
    def forward(self, x: torch.Tensor, return_maps: bool = False) -> torch.Tensor:
        """
        Parameters
        ----------
        x : If n_components == 1: [B, seq_len, Lx, Ly]
            If n_components > 1: [B, seq_len, n_components, Lx, Ly]
        return_maps : bool (optional, default False)
            If True, returns the full local EP maps [B, K, Lx, Ly].
            If False, returns the scalar EP per distance [B, K].

        Returns
        -------
        If return_maps is False:
            J : [B, K+1]  where J[:, k] is the estimated EP at distance k.
        If return_maps is True:
            maps : [B, K+1, Lx, Ly] where maps[:, k, :, :] is the local EP map at distance k.
        """
        if self.n_components > 1:
            # x is [B, seq_len, C, Lx, Ly]
            B, S, C, Lx, Ly = x.shape
            x_ = x.reshape(B, S * C, Lx, Ly)
            _x = torch.flip(x, [1]).reshape(B, S * C, Lx, Ly)
            
            # Extract displacement vector
            dx = x[:, 1, :, :, :] - x[:, 0, :, :, :]  # [B, C, Lx, Ly]
            _dx = -dx                                 # Time reversed displacement
        else:
            # Time-forward and time-reversed inputs
            x_ = x                          # forward
            _x = torch.flip(x, [1])         # reverse time dimension
            
            # Extract displacement vector
            dx = (x[:, 1, :, :] - x[:, 0, :, :]).unsqueeze(1)  # [B, 1, Lx, Ly]
            _dx = -dx

        # Optionally append positional channels
        if self.positional:
            x_ = add_spatial_channels(x_)
            _x = add_spatial_channels(_x)

        J_list = []
        map_list = []

        for branch in self.branches:
            # Branch outputs Force vector: [B, C, Lx, Ly]
            A_fwd = self._local_map(x_, branch)
            A_rev = self._local_map(_x, branch)

            # Local EP maps = Force · dx
            map_fwd = (A_fwd * dx).sum(dim=1, keepdim=True)   # [B, 1, Lx, Ly]
            map_rev = (A_rev * _dx).sum(dim=1, keepdim=True)  # [B, 1, Lx, Ly]

            # Time-reversal antisymmetry at the map level
            local_ep = map_fwd - map_rev   # [B, 1, Lx, Ly]

            if return_maps:
                map_list.append(local_ep)
            else:
                # Global Average Pooling over the spatial dimensions
                J_k = local_ep.mean(dim=(2, 3))          # [B, 1]
                J_list.append(J_k)

        if return_maps:
            # Stack into [B, K+1, Lx, Ly]
            return torch.cat(map_list, dim=1)
        else:
            # Stack into [B, K+1]
            return torch.cat(J_list, dim=1)
