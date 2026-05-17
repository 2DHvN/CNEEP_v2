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
# Masked 2D Convolution 
# ──────────────────────────────────────────────────────────────
class MaskedConv2d(nn.Module):
    """Conv2d whose kernel is masked so that only the center and
    the 4 orthogonal points at Manhattan/grid distance k are learnable.
    """

    def __init__(self, in_channels: int, out_channels: int, k: int,
                 bias: bool = True):
        super(MaskedConv2d, self).__init__()
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

        mask = torch.zeros(1, 1, self.kernel_size, self.kernel_size)
        mask[0, 0, k, k] = 1.0
        if k > 0:
            mask[0, 0, 0, :] = 1.0
            mask[0, 0, 2 * k, :] = 1.0
            mask[0, 0, :, 0] = 1.0
            mask[0, 0, :, 2 * k] = 1.0
            
        self.register_buffer("mask", mask)

        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight * self.mask, self.bias,
                        stride=1, padding=0)


# ──────────────────────────────────────────────────────────────
# Spatial Weighted Sum (Breaks Translation Equivariance)
# ──────────────────────────────────────────────────────────────
class SpatialWeightedSum2d(nn.Module):
    """
    Applies a learnable spatial weight map of shape [1, C, Lx, Ly].
    This explicitly breaks translation equivariance.
    """
    def __init__(self, in_channels: int, Lx: int, Ly: int):
        super(SpatialWeightedSum2d, self).__init__()
        self.weight = nn.Parameter(torch.empty(1, in_channels, Lx, Ly))
        self.bias = nn.Parameter(torch.empty(1, 1, Lx, Ly))
        
        # Initialize with small values
        nn.init.normal_(self.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, Lx, Ly]
        # output: [B, 1, Lx, Ly]
        return torch.sum(x * self.weight, dim=1, keepdim=True) + self.bias


# ──────────────────────────────────────────────────────────────
# Single 2DF Branch
# ──────────────────────────────────────────────────────────────
class _CorrelationBranch2DF(nn.Module):
    """
    Pipeline (per branch):
        PeriodicPad2d(k) → MaskedConv2d(kernel=2k+1) → ELU
        → [Conv2d(1×1) → ELU] × (n_hidden - 1)
        → Conv2d(1×1) to reduce_channels → ELU
        → SpatialWeightedSum2d [B, 1, Lx, Ly]
    """

    def __init__(self, k: int, in_channels: int, hidden_channels: int,
                 reduce_channels: int, Lx: int, Ly: int, n_hidden: int = 2):
        super(_CorrelationBranch2DF, self).__init__()
        self.k = k

        layers = []
        if k > 0:
            layers.append(PeriodicPad2d(k))
            
        layers.append(MaskedConv2d(in_channels, hidden_channels, k=k))
        layers.append(nn.ELU(inplace=True))

        for _ in range(n_hidden - 1):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1))
            layers.append(nn.ELU(inplace=True))

        # 1x1 conv to moderately reduce channels before spatial masking
        layers.append(nn.Conv2d(hidden_channels, reduce_channels, kernel_size=1))
        layers.append(nn.ELU(inplace=True))

        self.net = nn.Sequential(*layers)
        
        # Spatial weighted sum (C x W x H)
        self.spatial_weight = SpatialWeightedSum2d(reduce_channels, Lx, Ly)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : [B, seq_len, Lx, Ly]"""
        features = self.net(x)
        return self.spatial_weight(features)


# ──────────────────────────────────────────────────────────────
# Spatial channel helper
# ──────────────────────────────────────────────────────────────
def add_spatial_channels(x: torch.Tensor) -> torch.Tensor:
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
# 2D Multi-Scale CNEEP 2DF Model
# ──────────────────────────────────────────────────────────────
class MultiScaleCNEEP_2DF(nn.Module):
    """
    Parallel branch architecture with Spatial Masking (2DF) for 2D.
    Breaks translation equivariance via location-specific weights.
    """

    def __init__(self, opt):
        super(MultiScaleCNEEP_2DF, self).__init__()

        self.positional = getattr(opt, "positional", False)
        self.beta = getattr(opt, "beta", 1.0)
        self.max_distance = opt.max_distance
        
        # Extract spatial dimensions from opt.input_shape
        if not hasattr(opt, "input_shape"):
            raise ValueError("opt.input_shape (e.g., (64, 64)) is required for XDF models to define the spatial mask.")
        Lx, Ly = opt.input_shape

        in_channels = opt.seq_len + (2 if self.positional else 0)
        hidden_channels = opt.n_channel
        n_hidden = getattr(opt, "n_hidden", 2)
        reduce_channels = getattr(opt, "reduce_channel", 4) # Default to 4 if not specified

        self.include_k0 = getattr(opt, "include_k0", True)
        start_k = 0 if self.include_k0 else 1

        self.branches = nn.ModuleList([
            _CorrelationBranch2DF(
                k=k,
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                reduce_channels=reduce_channels,
                Lx=Lx, Ly=Ly,
                n_hidden=n_hidden,
            )
            for k in range(start_k, self.max_distance + 1)
        ])

        # Kaiming init for standard layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _local_map(self, x: torch.Tensor,
                   branch: _CorrelationBranch2DF) -> torch.Tensor:
        return branch(x)

    def forward(self, x: torch.Tensor, return_maps: bool = False) -> torch.Tensor:
        x_ = x
        _x = torch.flip(x, [1])

        if self.positional:
            x_ = add_spatial_channels(x_)
            _x = add_spatial_channels(_x)

        J_list = []
        map_list = []

        for branch in self.branches:
            map_fwd = self._local_map(x_, branch)
            map_rev = self._local_map(_x, branch)

            local_ep = (map_fwd - map_rev)

            if return_maps:
                map_list.append(local_ep)
            else:
                J_k = local_ep.mean(dim=(2, 3))
                J_list.append(J_k)

        if return_maps:
            return torch.cat(map_list, dim=1)
        else:
            return torch.cat(J_list, dim=1)
