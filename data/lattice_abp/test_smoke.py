"""Quick smoke test for LatticeABP simulation."""
import sys, os
sys.path.insert(0, '.')
os.makedirs("output", exist_ok=True)
from core import LatticeABP

sim = LatticeABP(L=16, density=0.4, seed=42)
res = sim.simulate(B=2, n_steps=200, burn_in=100, save_interval=50, show_progress=False)

print("O_traj:", res["O_traj"].shape)
print("E_traj:", res["E_traj"].shape)
print("times:", res["times"].shape)

# Test visualization
from visualization import visualize_state
jammed = sim.compute_jammed_mask(res["O_final"], res["E_final"])
fig, ax = visualize_state(
    res["O_final"], res["E_final"], jammed,
    ensemble_idx=0, title="Test", save_path="output/test_state.png",
)
print("All tests passed!")
