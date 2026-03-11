import nbformat as nbf

nb = nbf.v4.new_notebook()

md_intro = """# Phase Separated States in 1D (Single Domain Wall)
This notebook studies phase separated states in one dimension by considering a single domain wall in the center of the system and imposing $\\nabla\\phi = 0$ (Neumann) on the distant boundaries.

It strictly uses finite difference methods with mid-point spatial discretisation (2nd order) and explicit first order Euler integration, exactly following the reference protocol for $D=0$ (or low $D$)."""

code_setup = """import sys, os
import numpy as np
import matplotlib.pyplot as plt
import torch
from generate_trajectories_1d import ActiveModelB1D

# Simulation Parameters for a single domain wall
kwargs = {
    'Lx': 500,            # Domain length
    'dx': 1.0,             # Spatial resolution
    'a': 0.125,            
    'b': 0.125,            
    'kappa': 8.0,          
    'lam': 2.0,            
    'D': 0.0,              # Start with deterministic or very low noise
    'dt': 0.005,           # Time step
    'smooth': True,        
    'backend': 'torch',
    'use_gpu': torch.cuda.is_available(),
    'bc': 'neumann'        # Neumann boundaries for single wall
}

print(f"Using Backend: {kwargs['backend']} | GPU Enabled: {kwargs['use_gpu']}")
model = ActiveModelB1D(**kwargs)
"""

code_sim = """# Generate deterministic trajectory
print("Relaxing the single domain wall...")

# We'll run a single trajectory for a few steps to see the wall stabilization
n_steps = 10000

# Burn-in can be 0 to see the initial relaxation from the tanh profile
trajectories = model.generate_trajectories(
    n_trajectories=1,
    n_steps=n_steps,
    burn_in=0,
    show_progress=True
)

# trajectories shape: (1, n_steps, Lx)
traj = trajectories[0]
"""

code_plot = """# Visualization of the Domain Wall
x = np.arange(model.Lx) * model.dx

fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Initial vs Final Density Profile
ax[0].plot(x, traj[0], '--', label='Initial (t=0)')
ax[0].plot(x, traj[-1], '-', lw=2, label=f'Final (t={n_steps*kwargs["dt"]:.1f})')
ax[0].set_title('Density Profile $\\phi(x)$')
ax[0].set_xlabel('x')
ax[0].set_ylabel('$\\phi$')
ax[0].legend()

# Plot 2: Kymograph
im = ax[1].imshow(traj.T, aspect='auto', origin='lower', cmap='magma', 
                  extent=[0, n_steps*kwargs["dt"], 0, model.Lx*model.dx])
plt.colorbar(im, ax=ax[1], label='$\\phi$')
ax[1].set_title('Space-Time Kymograph')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('x')

plt.tight_layout()
plt.show()
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(md_intro),
    nbf.v4.new_code_cell(code_setup),
    nbf.v4.new_code_cell(code_sim),
    nbf.v4.new_code_cell(code_plot)
]

with open(r'c:\Users\ldh04\OneDrive\문서\CANONEQal\repos\CNEEP_v2\notebooks\sanity_check_amb_domain_wall.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook generated.")
