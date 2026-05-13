"""
Lattice Active Brownian Particle (ABP) simulation package
==========================================================
Based on: "Phase separation and large deviations of lattice active matter"
          Whitelam, Klymko, Mandal (2018), J. Chem. Phys. 148, 154902

Modules:
    core           — LatticeABP simulator, BoundaryCondition, InteractionModule
    visualization  — State rendering with jammed/free color coding
    run_demo       — CLI demo runner
"""

from .core import LatticeABP, BoundaryCondition, InteractionModule
from .visualization import visualize_state, visualize_density_evolution, visualize_jammed_fraction
