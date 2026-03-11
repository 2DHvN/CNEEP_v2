#!/usr/bin/env python
# coding: utf-8

# # AMB 1D Simulation Sanity Check
# This notebook verifies that the 1D Cahn-Hilliard active model simulation (Active Model B) integrates strictly using Flux (J) operations and Dense Matrix ($N \times M$) interactions instead of FFT.
# It computes and visualizes the **Mean Local EPR Density** over a very short trajectory.

# In[1]:


import sys, os

# In[6]:


import numpy as np
import matplotlib.pyplot as plt
import torch

from generate_trajectories_1d import ActiveModelB1D

# Core Simulation Parameters
kwargs = {
    'Lx': 5000,            # Domain length
    'dx': 0.1,             # Spatial resolution
    'a': 0.125,            # Active model parameter A
    'b': 0.125,            # Active model parameter B
    'kappa': 8.0,          # Gradient coefficient
    'lam': 2.0,            # Active parameter (Lambda)
    'D': 0.001,            # Noise strength (Diffusivity)
    'dt': 0.001,           # Time step
    'smooth': True,        # Smoothing for stability
    'backend': 'torch',
    'use_gpu': torch.cuda.is_available()
}

print(f"Using Backend: {kwargs['backend']} | GPU Enabled: {kwargs['use_gpu']}")

model = ActiveModelB1D(**kwargs)


n_seeds = 500
n_steps = 1000
burn_in = 2000

# Metric Evaluator (Mean EPR Density)
print('Simulating Ensemble and Computing EPR Density On-the-Fly...')
ensemble_epr_density = model.compute_mean_epr_on_the_fly(
    n_trajectories=n_seeds, n_steps=n_steps, burn_in=burn_in, show_progress=True
)

# Print Total Mean EPR
total_epr = np.sum(ensemble_epr_density) * model.dx
print(f'Total Mean EPR across ensemble: {total_epr:.6e}')


# In[5]:


# Visualization: Mean Local EPR Density Only
x = np.arange(model.Lx) * model.dx

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, ensemble_epr_density, lw=2, color='crimson', label=f'Ensemble Mean EPR Density ({n_seeds} seeds)')
ax.axhline(0, color='black', ls='--', lw=0.8)

ax.set_title('Time & Ensemble-Averaged Local EPR Density (Sanity Check)')
ax.set_xlabel('Spatial Coordinate (x)')
ax.set_ylabel('$<\sigma(x)>_{t, \mathrm{ens}}$')
ax.set_xlim(0, kwargs['Lx'] * model.dx)
ax.legend()

plt.tight_layout()
plt.show()

