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

            # linearized elastic forces: f_ij = k * (u_j - u_i)
            # u = r - r0 (displacement from equilibrium)
            for i, j in self.bonds:
                du = (r[j] - self.r0[j]) - (r[i] - self.r0[i])
                f = self.k * du       # (2,) vector, x and y independent
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

    # ================================================================
    # Entropy Production Rate (EPR) computation
    # ================================================================

    def compute_forces_at(self, positions):
        """
        Compute linearized spring forces on all sites.

        F_ij = k * (u_j - u_i),  u = positions - r0

        Parameters
        ----------
        positions : (N, 2) ndarray

        Returns
        -------
        F : (N, 2) ndarray — net force on each site
        """
        F = np.zeros_like(positions)
        for i, j in self.bonds:
            du = (positions[j] - self.r0[j]) - (positions[i] - self.r0[i])
            f = self.k * du
            F[i] += f
            F[j] -= f
        return F

    def compute_heat_per_site(self, trajectory):
        """
        Compute heat flow into each site at each time step (Stratonovich convention).

        δQ_i(t) = F_i(r_mid) · Δr_i ,  r_mid = (r_t + r_{t+1}) / 2

        Parameters
        ----------
        trajectory : (T, N, 2) ndarray

        Returns
        -------
        heat : (T-1, N) ndarray — heat absorbed by each site per step
        """
        n_steps = trajectory.shape[0]
        heat = np.zeros((n_steps - 1, self.N))

        for t in range(n_steps - 1):
            r_curr = trajectory[t]
            r_next = trajectory[t + 1]
            r_mid = (r_curr + r_next) / 2.0
            dr = r_next - r_curr

            F_mid = self.compute_forces_at(r_mid)

            # δQ_i = F_i · dr_i  (dot product over x,y components)
            heat[t] = np.sum(F_mid * dr, axis=1)

        return heat

    def compute_entropy_production_per_site(self, trajectory):
        """
        Entropy production decomposed by site: σ_i = δQ_i / T_i

        Parameters
        ----------
        trajectory : (T, N, 2) ndarray

        Returns
        -------
        ep_per_site : (T-1, N) ndarray
        """
        heat = self.compute_heat_per_site(trajectory)
        return heat / self.T[np.newaxis, :]

    def compute_entropy_production_rate(self, trajectory):
        """
        Total instantaneous entropy production rate: σ(t) = Σ_i σ_i(t)

        Parameters
        ----------
        trajectory : (T, N, 2) ndarray

        Returns
        -------
        ep_rate : (T-1,) ndarray
        """
        ep_per_site = self.compute_entropy_production_per_site(trajectory)
        return np.sum(ep_per_site, axis=1)

    def compute_mean_entropy_production(self, trajectory):
        """
        Mean entropy production rate over the entire trajectory.

        Parameters
        ----------
        trajectory : (T, N, 2) ndarray

        Returns
        -------
        mean_ep : float
        """
        return float(np.mean(self.compute_entropy_production_rate(trajectory)))
