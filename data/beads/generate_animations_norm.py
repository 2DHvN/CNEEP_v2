"""
N-Beads Brownian Movie — Per-bead Distribution Normalization

기존 generate_animations.py의 문제점:
  - 모든 bead에 동일한 global scale을 적용
  - 온도가 높은 bead의 진폭이 scale을 지배 → 낮은 온도 bead의 움직임이 압축됨

개선:
  - 각 bead x_i 의 정상 상태(steady-state) 분포를 이용하여 개별 정규화
  - x_i → (x_i - mean_i) / std_i 후 pixel 좌표로 변환
  - 결과적으로 모든 bead가 cell 내에서 동일한 시각적 dynamic range를 가짐
  - CNN이 각 bead의 상대적 움직임 차이를 더 잘 학습할 수 있음
"""

import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from generate_trajectories import NBeadsModel


# ============================================================
# Per-bead normalized frame generation
# ============================================================

def generate_brownian_frames_norm(
    trajectory: np.ndarray,
    cell_size: int,
    skip_frames: int = 1,
    sigma: float = 2.0,
    n_sigma_range: float = 3.0,
    stats: dict = None,
):
    """
    Generate raw pixel frames with per-bead distribution normalization.

    각 bead의 trajectory를 자체 분포(mean, std)로 정규화하여
    모든 bead가 cell 내에서 동일한 시각적 범위를 갖도록 합니다.

    Parameters
    ----------
    trajectory : (T, N) ndarray
        Bead positions over time.
    cell_size : int
        Pixel size of each bead's cell. Image = (cell_size, N * cell_size).
    skip_frames : int
        Frame subsampling interval.
    sigma : float
        Gaussian blob size (pixels).
    n_sigma_range : float
        Normalized coordinate ±n_sigma_range 가 cell 폭의 ±(A/4)에 매핑됨.
        기본값 3.0 → 약 99.7%의 분포가 cell 내에 수용.
    stats : dict or None
        사전에 계산된 통계. {'mean': (N,), 'std': (N,)}.
        None이면 입력 trajectory에서 자동 계산.

    Returns
    -------
    frames : (T', H, W, 3) uint8
    stats_used : dict
        실제 사용된 {'mean': ..., 'std': ...} — 테스트 시 동일 통계 재사용 가능.
    """
    n_steps, n_beads = trajectory.shape
    A = cell_size

    H = A
    W = n_beads * A

    # ---- per-bead statistics ----
    if stats is not None:
        means = np.asarray(stats['mean'])
        stds = np.asarray(stats['std'])
    else:
        means = trajectory.mean(axis=0)       # (N,)
        stds = trajectory.std(axis=0)          # (N,)
        stds = np.where(stds < 1e-8, 1.0, stds)  # 0-division guard

    # pixel scale: normalized ±n_sigma_range → ±(A/4) pixels
    pixel_scale = (A / 4.0) / n_sigma_range   # pixels per unit of normalized coord

    cell_centers_x = np.array([A // 2 + i * A for i in range(n_beads)])
    cell_center_y = A // 2

    yy, xx = np.meshgrid(
        np.arange(H),
        np.arange(W),
        indexing="ij"
    )

    frames = []

    for positions in trajectory[::skip_frames]:
        frame = np.zeros((H, W, 3), dtype=np.float32)

        for i, x in enumerate(positions):
            # per-bead normalization
            x_norm = (x - means[i]) / stds[i]

            cx = cell_centers_x[i] + x_norm * pixel_scale
            cy = cell_center_y

            blob = np.exp(
                -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)
            )

            frame[..., 1] += blob   # green channel

        if frame.max() > 0:
            frame /= frame.max()

        frame = (255 * frame).astype(np.uint8)
        frames.append(frame)

    stats_used = {'mean': means, 'std': stds}
    return np.stack(frames), stats_used


# ============================================================
# MP4 generation via matplotlib
# ============================================================

def save_frames_as_mp4(
    frames: np.ndarray,
    fps: int,
    output_path: str
):
    """Save frames to mp4 using matplotlib."""
    T, H, W, _ = frames.shape

    fig, ax = plt.subplots()
    ax.axis("off")

    im = ax.imshow(
        frames[0],
        interpolation="nearest",
        animated=True
    )

    def update(i):
        im.set_array(frames[i])
        return (im,)

    ani = animation.FuncAnimation(
        fig, update, frames=T,
        interval=1000 / fps, blit=True
    )

    writer = animation.FFMpegWriter(fps=fps)
    ani.save(output_path, writer=writer)
    plt.close(fig)

    print(f"[OK] MP4 saved → {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="N-Beads Brownian Movie (per-bead normalized)"
    )

    parser.add_argument("--n_beads", type=int, default=2)
    parser.add_argument("--n_steps", type=int, default=10000)
    parser.add_argument("--burn_in", type=int, default=2000)

    parser.add_argument("--k", type=float, default=1.0)
    parser.add_argument("--T_hot", type=float, default=10.0)
    parser.add_argument("--T_cold", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.01)

    parser.add_argument("--cell_size", type=int, default=20)
    parser.add_argument("--skip_frames", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--n_sigma_range", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=20)

    parser.add_argument("--output", type=str, default="brownian_movie_norm.mp4")
    parser.add_argument("--save_npz", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    np.random.seed(args.seed)

    print(f"[INFO] Generating {args.n_beads}-beads trajectory")

    model = NBeadsModel(
        n_beads=args.n_beads,
        k=args.k,
        T_hot=args.T_hot,
        T_cold=args.T_cold,
        dt=args.dt
    )

    trajectory = model.generate_trajectory(
        n_steps=args.n_steps,
        burn_in=args.burn_in
    )

    ep_rate = model.compute_mean_entropy_production(trajectory)
    print(f"[INFO] Mean EP rate: {ep_rate:.6f}")
    print(f"[INFO] Trajectory shape: {trajectory.shape}")

    # per-bead stats
    means = trajectory.mean(axis=0)
    stds = trajectory.std(axis=0)
    print(f"[INFO] Per-bead mean: {means}")
    print(f"[INFO] Per-bead std : {stds}")

    frames, stats_used = generate_brownian_frames_norm(
        trajectory=trajectory,
        cell_size=args.cell_size,
        skip_frames=args.skip_frames,
        sigma=args.sigma,
        n_sigma_range=args.n_sigma_range,
    )

    print(f"[INFO] Frames shape: {frames.shape}")

    if args.save_npz:
        npz_path = os.path.splitext(args.output)[0] + ".npz"
        np.savez(
            npz_path,
            frames=frames,
            trajectory=trajectory,
            cell_size=args.cell_size,
            skip_frames=args.skip_frames,
            sigma=args.sigma,
            norm_mean=stats_used['mean'],
            norm_std=stats_used['std'],
        )
        print(f"[OK] Saved dataset → {npz_path}")

    save_frames_as_mp4(
        frames=frames,
        fps=args.fps,
        output_path=args.output
    )

    print("Done.")


if __name__ == "__main__":
    main()
