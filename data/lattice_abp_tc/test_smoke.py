"""Quick smoke test for thermodynamically consistent lattice ABP."""

import math
import os
import sys
from dataclasses import replace

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.lattice_abp_tc import ThermodynamicLatticeABP, ThermodynamicLatticeABPParams
from data.lattice_abp_tc._numba_backend import NUMBA_AVAILABLE


def main():
    params = ThermodynamicLatticeABPParams(
        N=8,
        L=6.0,
        grid_size=24,
        sigma=1.0,
        epsilon=5.0,
        v0=4.0,
        Dr=1.0,
        Dt=0.5,
        dt=1.0e-3,
        seed=3,
        device="cpu",
    )
    sim = ThermodynamicLatticeABP(params)
    result = sim.simulate(
        B=2,
        burn_in=1,
        n_steps=4,
        save_interval=2,
        show_progress=False,
        save_exact_medium_ep=True,
        save_ep_maps=True,
    )

    assert result["sites"].shape == (3, 2, params.N, 2)
    assert result["positions"].shape == (3, 2, params.N, 2)
    assert result["occupancy"].shape == (3, 2, params.grid_size, params.grid_size)
    assert result["exact_medium_ep"].shape == (2, 2)
    assert result["exact_medium_ep_maps"].shape == (2, 2, params.grid_size, params.grid_size)
    assert torch.isfinite(result["exact_medium_ep"]).all()
    assert (result["occupancy"].sum(dim=(-2, -1)) == params.N).all()
    assert torch.allclose(
        result["exact_active_medium_ep"] + result["exact_wca_medium_ep"],
        result["exact_medium_ep"],
        rtol=1.0e-6,
        atol=1.0e-5,
    )
    assert torch.allclose(
        result["exact_medium_ep_maps"].sum(dim=(-2, -1)).T,
        result["exact_medium_ep"],
        rtol=1.0e-6,
        atol=1.0e-5,
    )
    compact_summary = sim.mips_summary(result["occupancy"][-1])
    assert "coarse_std" not in compact_summary
    coarse_summary = sim.mips_summary(
        result["occupancy"][-1],
        include_coarse=True,
    )
    assert "coarse_std" in coarse_summary

    free_params = ThermodynamicLatticeABPParams(
        N=1,
        L=8.0,
        grid_size=32,
        epsilon=0.0,
        v0=2.0,
        Dt=0.5,
        dt=1.0e-4,
        prefactor="cv",
        seed=1,
        device="cpu",
    )
    free_sim = ThermodynamicLatticeABP(free_params)
    sites, theta = free_sim.init_state(B=1)
    theta[:] = 0.0
    probs = free_sim.compute_particle_hop_probabilities(sites, theta, 0)["probabilities"][0]
    expected_ratio = math.exp(free_params.v0 * free_params.dl / free_params.Dt)
    assert abs((probs[0] / probs[1]).item() - expected_ratio) < 1.0e-6

    if NUMBA_AVAILABLE:
        exact_keys = (
            "sites",
            "positions",
            "theta",
            "occupancy",
            "potential",
            "accepted_hops",
            "exact_medium_ep",
            "exact_active_medium_ep",
            "exact_wca_medium_ep",
            "exact_medium_ep_maps",
        )
        equivalence_cases = (
            dict(
                dtype="float64",
                prefactor="cv",
                epsilon=3.0,
                v0=3.0,
                Dt=0.5,
                dt=5.0e-4,
            ),
            dict(
                dtype="float32",
                prefactor="c0",
                epsilon=0.1,
                v0=0.4,
                Dt=1.0,
                dt=1.0e-4,
            ),
        )
        for case_idx, case in enumerate(equivalence_cases):
            equivalence_params = ThermodynamicLatticeABPParams(
                N=10,
                L=7.0,
                grid_size=28,
                sigma=1.0,
                Dr=0.7,
                seed=19 + case_idx,
                device="cpu",
                backend="torch",
                **case,
            )
            reference = ThermodynamicLatticeABP(
                equivalence_params
            ).simulate(
                B=2,
                n_steps=12,
                save_interval=3,
                show_progress=False,
                save_exact_medium_ep=True,
                save_ep_maps=True,
            )
            accelerated = ThermodynamicLatticeABP(
                replace(equivalence_params, backend="numba")
            ).simulate(
                B=2,
                n_steps=12,
                save_interval=3,
                show_progress=False,
                save_exact_medium_ep=True,
                save_ep_maps=True,
            )
            for key in exact_keys:
                assert torch.equal(reference[key], accelerated[key]), (
                    case["dtype"],
                    case["prefactor"],
                    key,
                )

    print("Thermodynamic lattice ABP smoke test passed.")


if __name__ == "__main__":
    main()
