import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from generate_trajectories_1d import ActiveModelB1D

kwargs = {
    'Lx': 512,            # Domain length
    'dx': 1.0,             # Spatial resolution
    'a': 0.125,            # Active model parameter A
    'b': 0.125,            # Active model parameter B
    'kappa': 8.0,          # Gradient coefficient
    'lam': 2.0,            # Active parameter (Lambda)
    'dt': 0.01,           # Time step
    'smooth': True,        # Smoothing for stability
    'backend': 'torch',
    'use_gpu': torch.cuda.is_available()
}

D_values = [0.001, 0.002, 0.02, 0.04]
n_steps = 10000
burn_in = 20000
n_seeds = 1000

plt.figure(figsize=(10, 6))
colors = ['blue', 'green', 'orange', 'red']

x = np.arange(kwargs['Lx']) * kwargs['dx']

for i, D in enumerate(D_values):
    print(f"Simulating for D = {D}...")
    model = ActiveModelB1D(**kwargs, D=D)
    
    # Run ensemble simulation on the fly
    np.random.seed(42)
    torch.manual_seed(42)
    
    # We use compute_mean_epr_on_the_fly instead to save memory
    epr_density = model.compute_mean_epr_on_the_fly(
        n_trajectories=n_seeds, 
        n_steps=n_steps, 
        burn_in=burn_in, 
        show_progress=True
    )
    
    mean_total_epr = epr_density.sum() * kwargs['dx']
    plt.plot(x, epr_density, label=f"D = {D} (Total EPR={mean_total_epr:.4f})", color=colors[i], alpha=0.8)

plt.axhline(0, color='k', linestyle='--', alpha=0.5)
plt.xlabel('x')
plt.ylabel(r"$\langle\sigma(x)\rangle_t$")
plt.title(f"Time-averaged Local EPR Density for varying Diffusivity D\n(Steps={n_steps}, a={kwargs['a']}, b={kwargs['b']}, \u03ba={kwargs['kappa']}, \u03bb={kwargs['lam']}, dt={kwargs['dt']})")
plt.legend()
plt.tight_layout()

# Save the plot
output_filename = "sanity_check_D_variation.png"
plt.savefig(output_filename, dpi=150)
print(f"Plot saved to {output_filename}")
