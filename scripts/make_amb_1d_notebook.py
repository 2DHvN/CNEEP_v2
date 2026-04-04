import json
import os

with open(r'c:\Users\ldh04\OneDrive\문서\CANONEQal\repos\CNEEP_v2\notebooks\AMB.ipynb', 'r') as f:
    nb = json.load(f)

# Change title
nb['cells'][0]['source'][0] = "# Active Model B 1D — CNEEP_1D Notebook"

# Imports
src = nb['cells'][1]['source']
new_src = []
for line in src:
    if 'google.colab' in line or 'drive.mount' in line:
        continue
    if "CNEEP_V2_ROOT =" in line:
        new_src.append("CNEEP_V2_ROOT = os.path.abspath('..')\n")
    elif 'from models.train import train' in line:
        new_src.append('from models.train_1d import train_1d as train\n')
    elif 'from models.validate import validate' in line:
        new_src.append('from models.validate_1d import validate_1d as validate\n')
    elif 'from generate_trajectories import ActiveModelB' in line:
        new_src.append('from generate_trajectories_1d import ActiveModelB1D\n')
    else:
        new_src.append(line)
nb['cells'][1]['source'] = new_src

# cell 2 - Hyper parameters
src = nb['cells'][2]['source']
new_src = []
for line in src:
    if 'opt.input_shape =' in line:
        new_src.append('opt.input_shape = (256,)    # AMB grid size\n')
    elif 'Lx=64, Ly=64' in line:
        new_src.append('    Lx=256, dx=1.0,\n')
    elif 'init_mode =' in line:
        new_src.append('init_mode = "wall"\n')
        new_src.append('init_type = "wall"\n')
        new_src.append('epr_mu_active_only = True\n')
    else:
        new_src.append(line)
nb['cells'][2]['source'] = new_src

# cell 3 - AMB parameters
src = nb['cells'][3]['source']
new_src = []
for line in src:
    if "'bc': \"periodic\"" in line:
        new_src.append("    'bc': \"periodic\",\n")
        new_src.append("    'epr_mu_active_only': True\n")
    else:
        new_src.append(line)
nb['cells'][3]['source'] = new_src

# cell 4 - train generation
src = nb['cells'][4]['source']
for i, line in enumerate(src):
    if 'ActiveModelB(' in line:
        src[i] = src[i].replace('ActiveModelB', 'ActiveModelB1D')
    if 'init_mode=' in line:
        src[i] = src[i].replace(', init_mode=init_mode', '')

# cell 6 - test generation
src = nb['cells'][6]['source']
for i, line in enumerate(src):
    if 'ActiveModelB(' in line:
        src[i] = src[i].replace('ActiveModelB', 'ActiveModelB1D')
    if 'init_mode=' in line:
        src[i] = src[i].replace(', init_mode=init_mode', '')

# cell 7 - ground truth EPR
src = nb['cells'][7]['source']
for i, line in enumerate(src):
    if 'gt_epr_maps  = np.zeros' in line:
        src[i] = 'gt_epr_maps  = np.zeros((opt.L_test - 1, amb_params["Lx"]))\n'
    if 'model_amb_test.compute_local_epr_density' in line:
        # no change needed initially, but let's check
        pass
    if 'gt_total_epr[t] = ' in line:
        # dx**2 to dx
        src[i] = '    gt_total_epr[t] = np.sum(epr_map) * model_amb_test.dx\n'

# cell 9 - Prepare video
src = nb['cells'][9]['source']
for i, line in enumerate(src):
    if 'unsqueeze(2)' in line:
        src[i] = line.replace('Lx, Ly', 'Lx')

# cell 11 - Build model
src = nb['cells'][11]['source']
for i, line in enumerate(src):
    if 'from models.CNEEP_0 import CNEEP' in line:
        src[i] = 'from models.CNEEP_1D import CNEEP\n'

# cell 14 - Validation
src = nb['cells'][14]['source']
for i, line in enumerate(src):
    if "pred_maps = pred_maps / (" in line:
        src[i] = "pred_maps = pred_maps / amb_params['Lx']\n"
    if "amb_params['Lx'] * amb_params['Ly']" in line:
        src[i] = "pred_maps = pred_maps / amb_params['Lx']\n"

# Leave visualization as is, mostly 1D or suitable for 1D. (We might need minimal manual tweaks later)
num_cells = len(nb['cells'])
# For cell 18 (Mean EPR density map), imshow won't work perfectly for 1D, will change to plot
for i in range(num_cells):
    if 'imshow(gt_mean_map' in nb['cells'][i]['source'][0] or 'Mean local EPR density' in nb['cells'][i]['source'][0]:
        nb['cells'][i]['source'] = [
            "# Mean local EPR density: GT vs Predicted\n",
            "gt_mean_map   = gt_epr_maps[:min_len].mean(axis=0)\n",
            "pred_mean_map = pred_maps[:min_len].mean(axis=0)\n",
            "\n",
            "plt.figure(figsize=(10, 4))\n",
            "plt.plot(gt_mean_map, label='GT Mean Local EPR Density', color='steelblue')\n",
            "plt.plot(pred_mean_map, label='Predicted Mean Local EPR Density', color='crimson')\n",
            "plt.legend()\n",
            "plt.title('Time-averaged local EPR density')\n",
            "plt.xlabel('x')\n",
            "plt.ylabel('EPR Density')\n",
            "plt.tight_layout()\n",
            "plt.savefig(f'{current_result_folder}/mean_epr_map_comparison.png', dpi=150)\n",
            "plt.show()\n"
        ]

# Cell 20 (Animation), imshow -> plot
for i in range(num_cells):
    if len(nb['cells'][i]['source']) > 0 and 'Overlay animation:' in nb['cells'][i]['source'][0]:
        nb['cells'][i]['source'] = [
            "# Overlay animation: density field + predicted EP map\n",
            "n_frames = min(min_len, 400)\n",
            "\n",
            "pred_map_norm = (pred_maps[:n_frames] - pred_maps[:n_frames].min()) / \\\n",
            "                (pred_maps[:n_frames].max() - pred_maps[:n_frames].min() + 1e-8)\n",
            "density_np = traj_test[:n_frames]\n",
            "\n",
            "fig, ax1 = plt.subplots(figsize=(8, 4))\n",
            "ax2 = ax1.twinx()\n",
            "\n",
            "line1, = ax1.plot(density_np[0], color='k', label='Density')\n",
            "line2, = ax2.plot(pred_map_norm[0], color='r', alpha=0.5, label='Predicted EP (norm)')\n",
            "ax1.set_ylim(density_np.min(), density_np.max())\n",
            "ax2.set_ylim(-0.1, 1.1)\n",
            "\n",
            "def update(frame):\n",
            "    line1.set_ydata(density_np[frame])\n",
            "    line2.set_ydata(pred_map_norm[frame])\n",
            "    return line1, line2\n",
            "\n",
            "ani = FuncAnimation(fig, update, frames=n_frames, blit=True)\n",
            "ani.save(f'{current_result_folder}/epr_overlay.mp4', fps=10)\n",
            "plt.close()\n",
            "print(f'Animation saved ({n_frames} frames)')\n"
        ]

with open(r'c:\Users\ldh04\OneDrive\문서\CANONEQal\repos\CNEEP_v2\notebooks\AMB_1D.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print("Done generating AMB_1D.ipynb!")
