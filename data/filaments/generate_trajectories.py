import numpy as np

class SquareLatticeFilamentModel:
    def __init__(
        self,
        Lx,
        Ly,
        k=1.0,
        gamma=5.0,
        T_hot=10.0,
        T_cold=1.0,
        dt=0.01,
        hot_fraction=0.2,
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.N = Lx * Ly
        self.k = k
        self.gamma = gamma
        self.dt = dt

        # lattice spacing
        self.a = 1.0

        # base lattice positions
        self.r0 = np.zeros((self.N, 2))
        for i in range(self.N):
            ix = i // Ly
            iy = i % Ly
            self.r0[i] = (ix * self.a, iy * self.a)

        # bonds (nearest neighbors)
        self.bonds = []
        for ix in range(Lx):
            for iy in range(Ly):
                i = ix * Ly + iy
                if ix + 1 < Lx and iy != 0 and iy != Ly-1:
                    self.bonds.append((i, (ix + 1) * Ly + iy))
                if iy + 1 < Ly and ix != 0 and ix != Lx-1:
                    self.bonds.append((i, ix * Ly + iy + 1))
        self.bonds = np.array(self.bonds)

        # fixed boundary sites
        self.fixed = np.zeros(self.N, dtype=bool)
        for i in range(self.N):
            ix = i // Ly
            iy = i % Ly
            if ix == 0 or ix == Lx - 1 or iy == 0 or iy == Ly - 1:
                self.fixed[i] = True

        # temperature field (random hot / cold)
        self.T = np.full(self.N, T_cold)
        hot_mask = np.random.rand(self.N) < hot_fraction
        self.T[hot_mask] = T_hot

    def generate_trajectory(self, n_steps, burn_in=0):
        r = self.r0.copy()
        traj = []

        for t in range(n_steps + burn_in):
            F = np.zeros_like(r)

            # elastic forces
            for i, j in self.bonds:
                rij = r[j] - r[i]
                dist = np.linalg.norm(rij) + 1e-12
                f = self.k * (dist - self.a) * rij / dist
                F[i] += f
                F[j] -= f

            # Langevin step
            noise = np.sqrt(
                2 * self.T[:, None] * self.dt / self.gamma
            ) * np.random.randn(self.N, 2)

            r += self.dt * F / self.gamma + noise

            # enforce fixed boundary
            r[self.fixed] = self.r0[self.fixed]

            if t >= burn_in:
                traj.append(r.copy())

        return np.stack(traj)
