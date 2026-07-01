import torch
import torch.nn as nn

from models.NEEP_K_2DF import _KBranch2D, add_spatial_channels


def rotate90_2d(x: torch.Tensor) -> torch.Tensor:
    """Apply R=[[0,-1],[1,0]] to a [B, 2, Lx, Ly] field."""
    return torch.stack((-x[:, 1], x[:, 0]), dim=1)


class MultiScaleK_2DF_LocalRot(nn.Module):
    """
    K2DF variant with an explicit but soft local rotational EP branch.

    k = 0 is anchored to

        dS_0(i) = g0 * [R x_mid(i)] dot dx(i)

    plus an optional learnable center-only residual branch initialized with a
    small gain. k > 0 branches are the original K2DF relative-shell branches.
    This anchors the local SAOU gauge without fully freezing the local model.
    """

    def __init__(self, opt):
        super(MultiScaleK_2DF_LocalRot, self).__init__()

        self.positional = opt.positional
        self.beta = opt.beta
        self.max_distance = opt.max_distance
        self.n_components = getattr(opt, "n_components", 1)
        self.include_k0 = getattr(opt, "include_k0", True)

        if self.n_components != 2:
            raise ValueError("MultiScaleK_2DF_LocalRot requires n_components=2.")
        if getattr(opt, "seq_len", None) != 2:
            raise ValueError("MultiScaleK_2DF_LocalRot requires seq_len=2.")

        gain_init = float(getattr(opt, "local_rot_gain_init", 1.0))
        gain = torch.tensor(gain_init, dtype=torch.float32)
        if getattr(opt, "local_rot_trainable", True):
            self.local_rot_gain = nn.Parameter(gain)
        else:
            self.register_buffer("local_rot_gain", gain)
        self.use_local_residual = getattr(opt, "local_rot_residual", True)
        residual_gain_init = float(getattr(opt, "local_residual_gain_init", 0.0))
        self.local_residual_gain = nn.Parameter(torch.tensor(residual_gain_init, dtype=torch.float32))

        in_channels = opt.seq_len * self.n_components + (2 if opt.positional else 0)
        hidden_channels = opt.n_channel
        n_hidden = getattr(opt, "n_hidden", 2)

        self.local_residual_branch = (
            _KBranch2D(
                k=0,
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                n_hidden=n_hidden,
                n_components=self.n_components,
            )
            if self.use_local_residual and self.include_k0
            else None
        )

        # k=0 is anchored separately; learn shell branches for k > 0.
        self.shell_ks = list(range(1, self.max_distance + 1))
        self.branches = nn.ModuleList(
            [
                _KBranch2D(
                    k=k,
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    n_hidden=n_hidden,
                    n_components=self.n_components,
                )
                for k in self.shell_ks
            ]
        )

    def _local_rotation_map(self, x: torch.Tensor, dx: torch.Tensor) -> torch.Tensor:
        x_mid = 0.5 * (x[:, 0] + x[:, 1])
        local_force = self.local_rot_gain * rotate90_2d(x_mid)
        return (local_force * dx).sum(dim=1, keepdim=True)

    def forward(self, x: torch.Tensor, return_maps: bool = False) -> torch.Tensor:
        """
        x: [B, 2, 2, Lx, Ly]

        If return_maps=False, returns [B, max_distance+1] when include_k0=True.
        If return_maps=True, returns [B, max_distance+1, Lx, Ly].
        """
        B, S, C, Lx, Ly = x.shape
        if S != 2 or C != 2:
            raise ValueError("Expected x with shape [B, 2, 2, Lx, Ly].")

        x_ = x.reshape(B, S * C, Lx, Ly)
        _x = torch.flip(x, [1]).reshape(B, S * C, Lx, Ly)
        dx = x[:, 1] - x[:, 0]
        _dx = -dx

        if self.positional:
            x_ = add_spatial_channels(x_)
            _x = add_spatial_channels(_x)

        map_list = []
        J_list = []

        if self.include_k0:
            local_ep = self._local_rotation_map(x, dx)
            if self.local_residual_branch is not None:
                A_fwd = self.local_residual_branch(x_)
                A_rev = self.local_residual_branch(_x)
                map_fwd = (A_fwd * dx).sum(dim=1, keepdim=True)
                map_rev = (A_rev * _dx).sum(dim=1, keepdim=True)
                local_ep = local_ep + self.local_residual_gain * (map_fwd - map_rev)
            if return_maps:
                map_list.append(local_ep)
            else:
                J_list.append(local_ep.mean(dim=(2, 3)))

        for branch in self.branches:
            A_fwd = branch(x_)
            A_rev = branch(_x)

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
