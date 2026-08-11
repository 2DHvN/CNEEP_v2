import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def kneep_spatial_mean_score(branch_maps: torch.Tensor) -> torch.Tensor:
    """Return the intensive NEEP score used by the training objective.

    ``branch_maps`` has shape ``[B, K, Lx, Ly]``.  Training follows
    ``utils.train``/``utils.validate``: average over space first, then sum the
    branch contributions.  The saved-frame time interval is already encoded
    in the transition and must not be divided out here.
    """
    if branch_maps.ndim != 4:
        raise ValueError(f"Expected branch maps [B,K,Lx,Ly], got {tuple(branch_maps.shape)}")
    return branch_maps.mean(dim=(-2, -1)).sum(dim=1)


def kneep_normalize_branch_maps(branch_maps: torch.Tensor) -> torch.Tensor:
    """Convert raw KNEEP branch maps to local EP maps by dividing by Lx*Ly.

    KNEEP is trained with a spatial-mean score.  Consequently the raw
    ``return_maps=True`` tensor carries the global score scale at every pixel;
    every predicted map must be divided by the number of spatial sites before
    it is interpreted or plotted locally.
    """
    if branch_maps.ndim != 4:
        raise ValueError(f"Expected branch maps [B,K,Lx,Ly], got {tuple(branch_maps.shape)}")
    n_sites = branch_maps.shape[-2] * branch_maps.shape[-1]
    return branch_maps / float(n_sites)


def kneep_local_ep_increment(branch_maps: torch.Tensor) -> torch.Tensor:
    """Normalize raw branch maps by Lx*Ly, then sum their local increments."""
    return kneep_normalize_branch_maps(branch_maps).sum(dim=1)


def kneep_total_ep_increment(branch_maps: torch.Tensor) -> torch.Tensor:
    """Return total EP from the raw maps' spatial means, without map scaling.

    Total EP follows the same spatial-mean score used by NEEP training.  The
    ``1/(Lx*Ly)`` normalization belongs only to local-map visualization and is
    deliberately not part of this computation.
    """
    return kneep_spatial_mean_score(branch_maps)


def kneep_total_from_spatial_mean(
    branch_scores: torch.Tensor,
    spatial_shape,
) -> torch.Tensor:
    """Return the total EP increment represented by ``model(x)``.

    ``branch_scores`` are already spatial means of the raw maps, hence already
    have the global NEEP score scale.  Do not multiply by Lx*Ly again.
    ``spatial_shape`` is retained to validate the caller's convention.
    """
    if branch_scores.ndim != 2:
        raise ValueError(f"Expected branch scores [B,K], got {tuple(branch_scores.shape)}")
    lx, ly = (int(spatial_shape[0]), int(spatial_shape[1]))
    if lx <= 0 or ly <= 0:
        raise ValueError(f"Invalid spatial shape {(lx, ly)}")
    return branch_scores.sum(dim=1)

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


def _canonical_kernel_geometry(kernel_geometry: str) -> str:
    geometry = str(kernel_geometry).lower()
    aliases = {
        "chebyshev": "chebyshev",
        "square": "chebyshev",
        "linf": "chebyshev",
        "l_inf": "chebyshev",
        "euclidean": "euclidean",
        "annulus": "euclidean",
        "l2": "euclidean",
        "l_2": "euclidean",
    }
    if geometry not in aliases:
        raise ValueError("kernel_geometry must be 'chebyshev' or 'euclidean'.")
    return aliases[geometry]


def _exclusive_offsets(
    k: int,
    kernel_geometry: str,
    shell_width: float = 1.0,
    shell_offset: float = 0.0,
):
    """Return offsets and display bounds for one exclusive K shell."""
    if k < 0:
        raise ValueError("k must be non-negative.")

    geometry = _canonical_kernel_geometry(kernel_geometry)
    if k == 0:
        return [(0, 0)], 0, 0.0, 0.0

    if geometry == "chebyshev":
        offsets = [
            (dy, dx)
            for dy in range(-k, k + 1)
            for dx in range(-k, k + 1)
            if max(abs(dy), abs(dx)) == k
        ]
        return offsets, k, float(max(k - 1, 0)), float(k)

    if shell_width <= 0:
        raise ValueError("shell_width must be positive for Euclidean kernels.")
    r_inner = max(0.0, shell_offset + (k - 0.5) * shell_width)
    r_outer = shell_offset + (k + 0.5) * shell_width
    radius = int(math.ceil(r_outer))
    offsets = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            r = math.sqrt(float(dy * dy + dx * dx))
            if r_inner < r <= r_outer + 1e-12:
                offsets.append((dy, dx))
    if not offsets:
        raise ValueError(
            f"Empty Euclidean K shell for k={k}, shell_width={shell_width}, "
            f"shell_offset={shell_offset}."
        )
    return offsets, radius, r_inner, r_outer


# ──────────────────────────────────────────────────────────────
# Exclusive Masked 2D Convolution
# ──────────────────────────────────────────────────────────────
class ExclusiveMaskedConv2d(nn.Module):
    """Conv2d whose kernel is masked so that:
      - k=0: only the center pixel is learnable.
      - k>0: only one exclusive Chebyshev perimeter or Euclidean annulus
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

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int,
        bias: bool = True,
        kernel_geometry: str = "chebyshev",
        shell_width: float = 1.0,
        shell_offset: float = 0.0,
    ):
        super(ExclusiveMaskedConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.kernel_geometry = _canonical_kernel_geometry(kernel_geometry)
        self.shell_width = float(shell_width)
        self.shell_offset = float(shell_offset)

        offsets, radius, r_inner, r_outer = _exclusive_offsets(
            k,
            self.kernel_geometry,
            shell_width=self.shell_width,
            shell_offset=self.shell_offset,
        )
        self.radius = radius
        self.r_inner = float(r_inner)
        self.r_outer = float(r_outer)
        self._offsets_list = offsets
        self.kernel_size = 2 * radius + 1

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, self.kernel_size, self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        # Build the exclusive binary mask
        mask = torch.zeros(1, 1, self.kernel_size, self.kernel_size)
        for dy, dx in offsets:
            mask[0, 0, radius + dy, radius + dx] = 1.0

        self.register_buffer("mask", mask)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

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

        if x_center is None:
            raise ValueError("x_center is required for k > 0 relative convolution.")

        # k > 0: Relative convolution: Σ_δ w_δ (X_{i+δ} - X_i)
        B, C_in, Lx, Ly = x_center.shape
        out = torch.zeros(B, self.out_channels, Lx, Ly, device=x.device, dtype=x.dtype)

        # Loop over the selected exclusive offsets.
        for dy, dx in self._offsets_list:
            x_shifted = x[
                :,
                :,
                self.radius + dy : self.radius + dy + Lx,
                self.radius + dx : self.radius + dx + Ly,
            ]
            x_rel = x_shifted - x_center
            w_delta = self.weight[:, :, self.radius + dy, self.radius + dx].unsqueeze(-1).unsqueeze(-1)
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

    def __init__(
        self,
        k: int,
        in_channels: int,
        hidden_channels: int,
        n_hidden: int = 2,
        n_components: int = 1,
        kernel_geometry: str = "chebyshev",
        shell_width: float = 1.0,
        shell_offset: float = 0.0,
    ):
        super(_KBranch2D, self).__init__()
        self.k = k

        # First layer: exclusive masked conv (supports relative mode)
        self.masked_conv = ExclusiveMaskedConv2d(
            in_channels,
            hidden_channels,
            k=k,
            kernel_geometry=kernel_geometry,
            shell_width=shell_width,
            shell_offset=shell_offset,
        )
        self.kernel_geometry = self.masked_conv.kernel_geometry
        self.radius = self.masked_conv.radius
        self.r_inner = self.masked_conv.r_inner
        self.r_outer = self.masked_conv.r_outer

        # Periodic padding (only for nonlocal shells)
        self.pad = PeriodicPad2d(self.radius) if self.radius > 0 else None
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
        # Input components may contain contextual observables that should not
        # themselves enter the final force·increment contraction.  By default
        # retain the historical behaviour and contract every component.
        component_indices = getattr(
            opt, "ep_component_indices", tuple(range(self.n_components))
        )
        self.ep_component_indices = tuple(int(index) for index in component_indices)
        if not self.ep_component_indices:
            raise ValueError("ep_component_indices must contain at least one component.")
        if len(set(self.ep_component_indices)) != len(self.ep_component_indices):
            raise ValueError("ep_component_indices must not contain duplicates.")
        if any(index < 0 or index >= self.n_components for index in self.ep_component_indices):
            raise ValueError(
                f"ep_component_indices={self.ep_component_indices} is invalid "
                f"for n_components={self.n_components}."
            )
        self.n_ep_components = len(self.ep_component_indices)
        in_channels = opt.seq_len * self.n_components + (2 if opt.positional else 0)
        hidden_channels = opt.n_channel
        n_hidden = getattr(opt, "n_hidden", 2)
        self.k_kernel_geometry = _canonical_kernel_geometry(getattr(opt, "k_kernel_geometry", "chebyshev"))
        self.shell_width = float(getattr(opt, "shell_width", 1.0))
        self.shell_offset = float(getattr(opt, "shell_offset", 0.0))

        self.include_k0 = getattr(opt, "include_k0", True)
        start_k = 0 if self.include_k0 else 1

        # Build K independent branches with exclusive masks
        self.branches = nn.ModuleList([
            _KBranch2D(
                k=k,
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                n_hidden=n_hidden,
                n_components=self.n_ep_components,
                kernel_geometry=self.k_kernel_geometry,
                shell_width=self.shell_width,
                shell_offset=self.shell_offset,
            )
            for k in range(start_k, self.max_distance + 1)
        ])

        # Kaiming init for all Conv2d inside branches
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ----- shared backbone: produces the force field ----- #
    def _local_map(self, x: torch.Tensor,
                   branch: _KBranch2D) -> torch.Tensor:
        """Run one branch on input *x* and return [B, C, Lx, Ly] force."""
        return branch(x)

    def shell_bounds(self):
        """Return display bounds in pixel units for each exclusive branch."""
        return [(branch.r_inner, branch.r_outer) for branch in self.branches]

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
            J : [B, K+1], the spatial mean of each branch's local EP
                increment map.  This is the intensive score used for stable
                NEEP training and the total-EP output scale.  Do not divide
                this score by Lx*Ly again.
        If return_maps is True:
            maps : [B, K+1, Lx, Ly], the raw branch maps.  Divide these maps
                   by Lx*Ly (or use ``kneep_normalize_branch_maps`` /
                   ``kneep_local_ep_increment``) before local interpretation.
                   No dt division is applied.
        """
        if self.n_components > 1:
            # x is [B, seq_len, C, Lx, Ly]
            B, S, C, Lx, Ly = x.shape
            x_ = x.reshape(B, S * C, Lx, Ly)
            _x = torch.flip(x, [1]).reshape(B, S * C, Lx, Ly)
            
            # Use the configured generalized increments in the final
            # contraction.  All input components remain available to infer A.
            dx = x[:, 1, self.ep_component_indices, :, :] - x[:, 0, self.ep_component_indices, :, :]
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
