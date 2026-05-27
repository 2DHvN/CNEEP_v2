import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = 32
mag = np.ones((1, L, L))
angle = np.zeros((1, L, L))
u = np.ones((1, L, L))
v = np.zeros((1, L, L))

arrow_stride = 2
arrow_scale = 25.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
im_mag = ax1.imshow(mag[0], vmin=0, vmax=1, cmap="magma", origin="lower", extent=[0, L, 0, L])
im_dir = ax2.imshow(angle[0], vmin=-np.pi, vmax=np.pi, cmap="twilight", origin="lower", extent=[0, L, 0, L])

y, x = np.meshgrid(np.arange(L) + 0.5, np.arange(L) + 0.5)
x_sub = x[::arrow_stride, ::arrow_stride]
y_sub = y[::arrow_stride, ::arrow_stride]

u_init = u[0] / (mag[0] + 1e-8)
v_init = v[0] / (mag[0] + 1e-8)
u_sub_init = u_init[::arrow_stride, ::arrow_stride]
v_sub_init = v_init[::arrow_stride, ::arrow_stride]

q = ax2.quiver(x_sub, y_sub, u_sub_init, v_sub_init, pivot='middle', scale=arrow_scale, scale_units='width')

print("Before tight_layout - ax2 limits:", ax2.get_xlim(), ax2.get_ylim())
fig.tight_layout(pad=3.0)
print("After tight_layout - ax2 limits:", ax2.get_xlim(), ax2.get_ylim())

fig.savefig("test_quiver_limits.png")
