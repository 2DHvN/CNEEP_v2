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
# Masked 2D Convolution  (Requirement 1)
# ──────────────────────────────────────────────────────────────
class MaskedConv2d(nn.Module):
    """Conv2d whose kernel is masked so that only the center and
    the 4 orthogonal points at Manhattan/grid distance k are learnable.

    For k=0, only the center is learnable.
    """

    def __init__(self, in_channels: int, out_channels: int, k: int,
                 bias: bool = True):
        super(MaskedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.kernel_size = 2 * k + 1

        # Learnable weight & bias
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, self.kernel_size, self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        # Build the binary mask
        mask = torch.zeros(1, 1, self.kernel_size, self.kernel_size)
        mask[0, 0, k, k] = 1.0  # Center point
        if k > 0:
            # Set the entire border/perimeter to 1.0
            mask[0, 0, 0, :] = 1.0        # Top edge
            mask[0, 0, 2 * k, :] = 1.0    # Bottom edge
            mask[0, 0, :, 0] = 1.0        # Left edge
            mask[0, 0, :, 2 * k] = 1.0    # Right edge
            
        self.register_buffer("mask", mask)

        # Kaiming initialisation
        nn.init.kaiming_normal_(self.weight, mode="fan_out",
                                nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, C_in, Lx, Ly]  (already periodically padded)."""
        return F.conv2d(x, self.weight * self.mask, self.bias,
                        stride=1, padding=0)


# ──────────────────────────────────────────────────────────────
# Single correlation-distance branch
# ──────────────────────────────────────────────────────────────
class _CorrelationBranch2D(nn.Module):
    """One branch for a specific distance *k*.

    Pipeline (per branch):
        PeriodicPad2d(k) → MaskedConv2d(kernel=2k+1) → ELU
        → [Conv2d(1×1) → ELU] × (n_hidden - 1)
        → Conv2d(1×1)  →  Local EP 2D Map  [B, 1, Lx, Ly]
    """

    def __init__(self, k: int, in_channels: int, hidden_channels: int,
                 n_hidden: int = 2):
        super(_CorrelationBranch2D, self).__init__()
        self.k = k

        layers = []
        if k > 0:
            layers.append(PeriodicPad2d(k))
            
        layers.append(MaskedConv2d(in_channels, hidden_channels, k=k))
        layers.append(nn.ELU(inplace=True))

        for _ in range(n_hidden - 1):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1))
            layers.append(nn.ELU(inplace=True))

        # Output to 1 channel (local EP scalar per position)
        layers.append(nn.Conv2d(hidden_channels, 1, kernel_size=1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, seq_len, Lx, Ly]"""
        return self.net(x)


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
# 2D Multi-Scale CNEEP Model
# ──────────────────────────────────────────────────────────────
class MultiScaleCNEEP2D(nn.Module):
    """
    Parallel branch architecture for 2D spatial entropy production estimation.
    Extracts EP contributions separated by correlation distance.
    """

    def __init__(self, opt):
        super(MultiScaleCNEEP2D, self).__init__()

        self.positional = opt.positional
        self.beta = opt.beta
        self.max_distance = opt.max_distance

        in_channels = opt.seq_len + (2 if opt.positional else 0)
        hidden_channels = opt.n_channel
        n_hidden = getattr(opt, "n_hidden", 2)

        # Build K independent branches (k = 0 … max_distance)
        self.branches = nn.ModuleList([
            _CorrelationBranch2D(
                k=k,
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                n_hidden=n_hidden,
            )
            for k in range(0, self.max_distance + 1)
        ])

        # Kaiming init for all Conv2d inside branches
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ----- shared backbone: produces the local EP map ----- #
    def _local_map(self, x: torch.Tensor,
                   branch: _CorrelationBranch2D) -> torch.Tensor:
        """Run one branch on input *x* and return [B, 1, Lx, Ly] local EP map."""
        return branch(x)

    # ----- forward ----- #
    def forward(self, x: torch.Tensor, return_maps: bool = False) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, seq_len, Lx, Ly]
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
        # Time-forward and time-reversed inputs
        x_ = x                          # forward
        _x = torch.flip(x, [1])         # reverse time dimension

        # Δφ (state difference used for the symmetric term)
        delta = (x[:, 0, :, :] - x[:, 1, :, :]).unsqueeze(1)  # [B, 1, Lx, Ly]

        # Optionally append positional channels
        if self.positional:
            x_ = add_spatial_channels(x_)
            _x = add_spatial_channels(_x)

        J_list = []
        map_list = []

        for branch in self.branches:
            # Local EP maps from forward / reversed inputs  [B, 1, Lx, Ly]
            map_fwd = self._local_map(x_, branch)
            map_rev = self._local_map(_x, branch)

            # Time-reversal antisymmetry at the map level
            local_ep = (map_fwd - map_rev)   # [B, 1, Lx, Ly]

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
