import json
import os

notebook_path = r'c:\Users\ldh04\OneDrive\문서\CANONEQal\repos\CNEEP_v2\data\AMB\sanity_check_amb_1d.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The content to append
code_lines = [
    "D_values = [0.001, 0.002, 0.02, 0.04]\n",
    "n_steps = 10000\n",
    "burn_in = 20000\n",
    "n_seeds = 1000\n",
    "\n",
    "plt.figure(figsize=(10, 6))\n",
    "colors = ['blue', 'green', 'orange', 'red']\n",
    "\n",
    "x = np.arange(kwargs['Lx']) * kwargs['dx']\n",
    "\n",
    "for i, D in enumerate(D_values):\n",
    "    print(f\"Simulating for D = {D}...\")\n",
    "    model = ActiveModelB1D(**kwargs, D=D)\n",
    "    \n",
    "    np.random.seed(42)\n",
    "    if torch is not None:\n",
    "        torch.manual_seed(42)\n",
    "    \n",
    "    epr_density = model.compute_mean_epr_on_the_fly(\n",
    "        n_trajectories=n_seeds, \n",
    "        n_steps=n_steps, \n",
    "        burn_in=burn_in, \n",
    "        show_progress=True\n",
    "    )\n",
    "    \n",
    "    mean_total_epr = epr_density.sum() * kwargs['dx']\n",
    "    plt.plot(x, epr_density, label=f\"D = {D} (Total EPR={mean_total_epr:.4f})\", color=colors[i], alpha=0.8)\n",
    "\n",
    "plt.axhline(0, color='k', linestyle='--', alpha=0.5)\n",
    "plt.xlabel('x')\n",
    "plt.ylabel(r\"$\\langle\\sigma(x)\\rangle_t$\")\n",
    "plt.title(f\"Time-averaged Local EPR Density for varying Diffusivity D\\n(Steps={n_steps}, a={kwargs['a']}, b={kwargs['b']}, \\u03ba={kwargs['kappa']}, \\u03bb={kwargs['lam']}, dt={kwargs['dt']})\")\n",
    "plt.legend()\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
]

nb['cells'].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": code_lines
})

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Successfully appended plotting code with new parameters to sanity_check_amb_1d.ipynb")
