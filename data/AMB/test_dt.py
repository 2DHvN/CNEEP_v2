import numpy as np
import torch
from generate_trajectories_1d import ActiveModelB1D

Lx = 100
dx = 1.0
n_steps = 100000

print(f"Testing dt dependency with n_steps={n_steps}")

for dt in [0.01, 0.005, 0.001]:
    # Fix n_steps to keep total simulation time consistent?
    # If the user means the *rate* changed, let's keep n_steps constant to see the rate
    # But to get steady state mean, it's better to keep total time T constant.
    # Actually let's just use compute_mean_epr_on_the_fly
    steps_for_dt = int(100.0 / dt) 
    burn_in_for_dt = int(100.0 / dt)
    
    model = ActiveModelB1D(Lx=Lx, dx=dx, dt=dt, backend='torch', use_gpu=True, fd_order=4)
    np.random.seed(42)
    torch.manual_seed(42)
    epr = model.compute_mean_epr_on_the_fly(n_trajectories=1, n_steps=steps_for_dt, burn_in=burn_in_for_dt, show_progress=False)
    print(f"dt = {dt:5.3f}, steps = {steps_for_dt:6d}, EPR density max = {epr.max():.6f}, min = {epr.min():.6f}, mean = {epr.mean():.6f}")
