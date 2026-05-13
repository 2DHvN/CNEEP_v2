import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """\
# Lattice ABP Sanity Check

이 노트북은 `core.py`에 구현된 **Lattice Active Brownian Particle (ABP)** 엔진이 정상적으로 동작하는지 확인하고,
생성된 MIPS(Motility-Induced Phase Separation) 시뮬레이션 결과를 인라인으로 빠르게 렌더링하고 시각화하는 용도로 작성되었습니다.
"""

code_imports = """\
import sys
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Patch
from IPython.display import Video, display

# 현재 디렉토리 모듈을 찾기 위해 경로 추가
sys.path.insert(0, os.path.abspath('.'))
from core import LatticeABP

# 출력 디렉토리 생성
os.makedirs("output", exist_ok=True)
"""

code_sim = """\
# 시각화용 RGB 배열 생성 헬퍼 함수
def build_rgb(occ, jam, L):
    rgb = np.zeros((L, L, 3))
    rgb[:, :, 0] = 0.05; rgb[:, :, 1] = 0.07; rgb[:, :, 2] = 0.14
    free = (occ == 1) & (~jam)
    rgb[free, 0] = 0.20; rgb[free, 1] = 0.60; rgb[free, 2] = 0.86
    jm = (occ == 1) & jam
    rgb[jm, 0] = 0.91; rgb[jm, 1] = 0.30; rgb[jm, 2] = 0.24
    return rgb

# 빠른 Sanity check를 위한 작은 시스템 구성 (L=64, 빠른 시뮬레이션)
L = 64
density = 0.55
v_plus, v_zero, v_minus, D_rot = 10.0, 0.5, 0.1, 0.2
n_frames = 200
steps_per_frame = 50

print(f"Initializing LatticeABP (L={L}, ρ={density}) ...")
sim = LatticeABP(
    L=L, v_plus=v_plus, v_zero=v_zero, v_minus=v_minus,
    D_rot=D_rot, density=density, bc_mode="periodic",
    device="auto", seed=123
)
O, E = sim.init_state(B=1)

frames_occ = []
frames_jammed = []

# 초기 상태
jammed_0 = sim.compute_jammed_mask(O, E)
frames_occ.append(O[0].cpu().numpy().copy())
frames_jammed.append(jammed_0[0].cpu().numpy().copy())

# PyTorch 최적화 (CPU 멀티스레딩 최대로 사용 및 Gradient 계산 비활성화)
torch.set_num_threads(os.cpu_count())

print("Running simulation (Gillespie steps)...")
t0 = time.time()

with torch.inference_mode():
    for f in range(1, n_frames):
        for _ in range(steps_per_frame):
            O, E, _ = sim.gillespie_step(O, E)
        jammed = sim.compute_jammed_mask(O, E)
        frames_occ.append(O[0].cpu().numpy().copy())
        frames_jammed.append(jammed[0].cpu().numpy().copy())

print(f"Simulation done in {time.time() - t0:.2f} seconds.")
"""

code_anim = """\
# 애니메이션 렌더링 및 저장
output_path = "output/sanity_check_mips.mp4"
print(f"Rendering animation to {output_path}...")

fig, ax = plt.subplots(figsize=(6, 6), facecolor="#0d1117")
ax.set_facecolor("#0d1117")
ax.set_xticks([]); ax.set_yticks([])

im = ax.imshow(build_rgb(frames_occ[0], frames_jammed[0], L), interpolation="nearest")

title = ax.set_title("Sanity Check MIPS", color="white", fontsize=12)
ax.legend(
    handles=[Patch(facecolor="#e74c3c", label="Jammed"), Patch(facecolor="#3498db", label="Free")],
    loc="upper right", fontsize=9, facecolor="#1a1a2e", edgecolor="#444", labelcolor="white",
)

def update(idx):
    im.set_data(build_rgb(frames_occ[idx], frames_jammed[idx], L))
    title.set_text(f"Step {idx * steps_per_frame}")
    return [im, title]

anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=50, blit=False)
writer = animation.FFMpegWriter(fps=20, bitrate=1500)
anim.save(output_path, writer=writer, dpi=100, savefig_kwargs={"facecolor": fig.get_facecolor()})
plt.close(fig)

print("Rendering complete!")
"""

code_display = """\
# 결과 영상 인라인 재생
Video(output_path, embed=True, width=500)
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_code_cell(code_sim),
    nbf.v4.new_code_cell(code_anim),
    nbf.v4.new_code_cell(code_display)
]

with open('sanity_check.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
