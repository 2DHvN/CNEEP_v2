"""Quick smoke test for continuous ABP utilities."""

import os
import sys
from argparse import Namespace

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.ABP import ABPFieldizer, ABPParams, ContinuousABP, recommended_center_grid_size
from models.NEEP_ABP_ShellForce_2D import ABPEuclideanShellForceCNEEP2D_Tanh


def main():
    params = ABPParams(
        N=16,
        L=8.0,
        sigma=1.0,
        v0=2.0,
        Dr=1.0,
        Dt=0.01,
        dt=1.0e-3,
        seed=1,
        force_chunk_size=8,
        device="cpu",
    )
    sim = ContinuousABP(params)
    grid_size = recommended_center_grid_size(params.L, params.sigma)
    fieldizer = ABPFieldizer(params.L, grid_size, params.sigma, mode="center", clip_occupancy=False)
    result = sim.simulate(
        B=2,
        burn_in=1,
        n_steps=4,
        save_interval=2,
        fieldizer=fieldizer,
        show_progress=False,
    )

    assert result["positions"].shape == (3, 2, params.N, 2)
    assert result["fields"].shape == (3, 2, 1, grid_size, grid_size)
    field_sums = result["fields"][:, :, 0].sum(dim=(-2, -1))
    assert torch.allclose(field_sums, torch.full_like(field_sums, float(params.N)), atol=1e-5)
    diag = fieldizer.diagnostics_dict(result["positions"][-1])
    assert diag["center_exclusive_rule_ok"]
    assert diag["multi_center_pixels"] == 0

    gaussian = ABPFieldizer(
        params.L,
        grid_size,
        params.sigma,
        mode="gaussian",
        clip_occupancy=False,
        gaussian_sigma=0.5 * params.sigma,
    )
    gaussian_field = gaussian.encode(result["positions"][-1])
    gaussian_sums = gaussian_field[:, 0].sum(dim=(-2, -1))
    assert torch.allclose(gaussian_sums, torch.full_like(gaussian_sums, float(params.N)), atol=1e-4)

    opt = Namespace(
        max_distance=3,
        include_k0=True,
        n_components=1,
        n_channel=8,
        n_hidden=2,
        positional=False,
        shell_relative_mode="learned_absolute",
        shell_width=1.0,
        shell_offset=0.0,
        shell_center_mode="relative_only",
        shell_force_bias=False,
    )
    model = ABPEuclideanShellForceCNEEP2D_Tanh(opt)
    video = result["fields"][:, :, 0].permute(1, 0, 2, 3).contiguous().float()
    x = torch.stack([video[:, 0], video[:, 1]], dim=1)
    J = model(x)
    maps = model(x, return_maps=True)
    assert J.shape == (2, 4)
    assert maps.shape == (2, 4, grid_size, grid_size)
    print("ABP smoke test passed.")


if __name__ == "__main__":
    main()
