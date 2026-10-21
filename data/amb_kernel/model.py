"""Scalar nonlocal AMB models M00--M05 on a periodic h=1 square lattice.

K is stored at FFT origin and acts as (K f)[i] = sum_s K[s] f[i-s].
All simulation arithmetic is float64. Kernels retain their complete periodic
support; max_distance only groups diagnostic channels into a tail bucket.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

SPEC_VERSION = "1.0"
MODEL_NAMES = {
    "M00": "Reciprocal interaction + local AMB",
    "M01": "Isotropic nonlinear source",
    "M02": "Nematic nonlinear source",
    "M03": "Polar linear kernel",
    "M04": "Mixed polar nonlinear source",
    "M05": "Nonlinear receiver",
}


def laplacian(field: np.ndarray) -> np.ndarray:
    """Positive lattice Laplacian L=-Delta, acting on the final two axes."""
    f = np.asarray(field)
    return (4 * f - np.roll(f, 1, -1) - np.roll(f, -1, -1)
            - np.roll(f, 1, -2) - np.roll(f, -1, -2))


def gradients(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Central differences (Gx phi, Gy phi), with h=1."""
    return ((np.roll(field, -1, -1) - np.roll(field, 1, -1)) / 2,
            (np.roll(field, -1, -2) - np.roll(field, 1, -2)) / 2)


def transpose_kernel(kernel: np.ndarray) -> np.ndarray:
    """K^T[s] = K[-s mod n], including self-inverse Nyquist offsets."""
    return np.roll(np.flip(kernel, axis=(-2, -1)), (1, 1), axis=(-2, -1))


def convolve(field: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Periodic convolution with an FFT-origin real kernel (not correlation)."""
    return np.fft.ifft2(np.fft.fft2(field) * np.fft.fft2(kernel)).real


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = "M01"
    n: int = 32
    phi_bar: float = 0.0
    a: float = 1.0
    b: float = 1.0
    kappa: float = 2.0
    g1: float = 0.25
    g2: float = 0.10
    lambda0: float = 0.0
    eta: float = 0.0
    D: float = 0.30
    ell_x: float = 1.2
    ell_y: float = 1.2
    vx: float = 0.0
    vy: float = 0.0
    kernel_family: str = "green_even"
    reciprocal_linear: bool = False

    def __post_init__(self) -> None:
        if self.model_id not in MODEL_NAMES:
            raise ValueError(f"Unknown model_id {self.model_id!r}; choose M00 through M05.")
        if isinstance(self.n, bool) or not isinstance(self.n, (int, np.integer)) or self.n < 3:
            raise ValueError("n must be an integer >= 3.")
        numeric = (self.phi_bar, self.a, self.b, self.kappa, self.g1, self.g2,
                   self.lambda0, self.eta, self.D, self.ell_x, self.ell_y, self.vx, self.vy)
        if not np.all(np.isfinite(numeric)):
            raise ValueError("All physical parameters must be finite.")
        if self.D <= 0 or self.kappa <= 0 or self.b < 0:
            raise ValueError("D and kappa must be positive, and b must be nonnegative.")
        if self.ell_x < 0 or self.ell_y < 0:
            raise ValueError("Screening lengths must be nonnegative.")
        families = {"green_even", "green_even_anisotropic", "green_polar_upwind",
                    "polar_even", "delta", "uniform"}
        if self.kernel_family not in families:
            raise ValueError(f"kernel_family must be one of {sorted(families)}.")
        if self.model_id in {"M00", "M03"} and (self.g2 != 0 or self.eta != 0):
            raise ValueError("M00/M03 require g2=eta=0; use pilot_config(model_id).")
        if self.model_id != "M00" and self.lambda0 != 0:
            raise ValueError("Only M00 includes local AMB activity lambda0.")
        if self.kernel_family in {"green_even", "green_even_anisotropic"} and (self.vx or self.vy):
            raise ValueError("Nonzero transport requires green_polar_upwind or polar_even.")

    @property
    def Nx(self) -> int:
        return self.n

    @property
    def Ny(self) -> int:
        return self.n

    @property
    def h(self) -> float:
        return 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def pilot_config(model_id: str = "M01", **overrides: Any) -> ModelConfig:
    """Spec pilot starting values, not a claim of stationarity or significant EPR.

    M02 accepts ell_geo/chi/director_axis convenience arguments. Other models
    accept ell to set both screening lengths. Nx=Ny is accepted as an alias for n.
    """
    model_id = model_id.upper()
    values: dict[str, Any] = {"model_id": model_id}
    if model_id == "M00":
        values.update(g2=0.0, lambda0=0.10)
    elif model_id == "M02":
        values.update(kernel_family="green_even_anisotropic", ell_x=1.2*np.sqrt(2), ell_y=1.2/np.sqrt(2))
    elif model_id == "M03":
        values.update(b=0.0, g2=0.0, vx=1.2, kernel_family="green_polar_upwind")
    elif model_id == "M04":
        values.update(vx=1.2, kernel_family="green_polar_upwind")
    if "Nx" in overrides or "Ny" in overrides:
        nx = overrides.pop("Nx", overrides.get("Ny", overrides.get("n", 32)))
        ny = overrides.pop("Ny", nx)
        if nx != ny or ("n" in overrides and overrides["n"] != nx):
            raise ValueError("The protocol requires Nx=Ny=n.")
        overrides["n"] = nx
    if "ell" in overrides:
        ell = overrides.pop("ell")
        overrides.update(ell_x=ell, ell_y=ell)
    if any(key in overrides for key in ("ell_geo", "chi", "director_axis")):
        ell_geo = float(overrides.pop("ell_geo", 1.2))
        chi = float(overrides.pop("chi", 2.0 if model_id == "M02" else 1.0))
        axis = overrides.pop("director_axis", "x")
        if not np.isfinite(chi) or chi <= 0 or axis not in {"x", "y"}:
            raise ValueError("chi must be positive and director_axis must be x or y.")
        ex, ey = ell_geo*np.sqrt(chi), ell_geo/np.sqrt(chi)
        overrides.update(ell_x=ex if axis == "x" else ey, ell_y=ey if axis == "x" else ex)
    values.update(overrides)
    return ModelConfig(**values)


def control_configs(model_id: str = "M01", **overrides: Any) -> dict[str, ModelConfig]:
    """Primary pilot plus the specification's essential matched controls.

    These are configurations for separate runs. Their EPR differences do not
    replace same-trajectory mechanism attribution.
    """
    primary = pilot_config(model_id, **overrides)
    model_id = primary.model_id
    controls = {"primary": primary}
    if model_id == "M00":
        controls.update(passive=replace(primary, lambda0=0.0),
                        local_amb=replace(primary, g1=0.0),
                        delta=replace(primary, kernel_family="delta"))
    elif model_id in {"M01", "M02"}:
        controls.update(no_source=replace(primary, g2=0.0),
                        reciprocal=replace(primary, eta=1.0),
                        delta=replace(primary, kernel_family="delta"),
                        uniform=replace(primary, kernel_family="uniform"))
        if model_id == "M02":
            ell_geo = np.sqrt(primary.ell_x*primary.ell_y)
            controls.update(isotropic=replace(primary, ell_x=ell_geo, ell_y=ell_geo),
                            axis_swap=replace(primary, ell_x=primary.ell_y, ell_y=primary.ell_x))
    elif model_id == "M03":
        controls.update(no_flow=replace(primary, vx=0.0, vy=0.0),
                        no_interaction=replace(primary, g1=0.0),
                        even_kernel=replace(primary, kernel_family="polar_even"),
                        reflected=replace(primary, vx=-primary.vx, vy=-primary.vy))
    elif model_id == "M04":
        controls.update(directional_only=replace(primary, g2=0.0),
                        source_only=replace(primary, vx=0.0, vy=0.0),
                        full_reciprocal=replace(primary, eta=1.0, reciprocal_linear=True),
                        reflected=replace(primary, vx=-primary.vx, vy=-primary.vy))
        if primary.g2 != 0:
            controls["linear_cancellation"] = replace(primary, phi_bar=-primary.g1/(2*primary.g2))
    elif model_id == "M05":
        controls.update(no_receiver=replace(primary, g2=0.0),
                        reciprocal=replace(primary, eta=1.0),
                        delta=replace(primary, kernel_family="delta"),
                        uniform=replace(primary, kernel_family="uniform"),
                        matched_source=replace(primary, model_id="M01"))
    return controls


class MidpointConvergenceError(RuntimeError):
    """The fixed-noise implicit midpoint solve failed; restart the run at smaller dt."""


class AMBKernelModel:
    spec_version = SPEC_VERSION

    def __init__(self, config: ModelConfig, kernel: np.ndarray | None = None):
        self.config = config
        self.shape = (config.n, config.n)
        self.N = config.n**2
        q = 2 * np.pi * np.fft.fftfreq(config.n)
        qx, qy = np.meshgrid(q, q)
        self.Lxhat = 4*np.sin(qx/2)**2
        self.Lyhat = 4*np.sin(qy/2)**2
        self.Lhat = self.Lxhat + self.Lyhat
        if kernel is None:
            family = config.kernel_family
            if family == "delta":
                khat = np.ones(self.shape, complex)
            elif family == "uniform":
                khat = np.zeros(self.shape, complex)
                khat[0, 0] = 1
            else:
                denominator = (1 + config.ell_x**2*self.Lxhat
                               + config.ell_y**2*self.Lyhat).astype(complex)
                if family in {"green_polar_upwind", "polar_even"}:
                    denominator += (abs(config.vx)*(1-np.cos(qx)) + abs(config.vy)*(1-np.cos(qy))
                                    + 1j*(config.vx*np.sin(qx)+config.vy*np.sin(qy)))
                khat = 1/denominator
                if family == "polar_even":
                    khat = khat.real
            self.K = np.fft.ifft2(khat).real
        else:
            self.K = np.array(kernel, dtype=np.float64, copy=True)
            if self.K.shape != self.shape or not np.all(np.isfinite(self.K)):
                raise ValueError("Custom kernel must be finite and have shape (n,n).")
        if not np.isclose(self.K.sum(), 1.0, rtol=0, atol=1e-12):
            raise ValueError("Kernel must sum to one; it will not be silently renormalized.")
        kt = transpose_kernel(self.K)
        self.Ke = (self.K + kt)/2
        self.Ko = (self.K - kt)/2
        if config.model_id in {"M00", "M01", "M02", "M05"} and np.max(np.abs(self.Ko)) > 1e-13:
            raise ValueError(f"{config.model_id} requires a reciprocal even kernel.")
        self.Khat = np.fft.fft2(self.K)
        self.Kehat = np.fft.fft2(self.Ke)
        self.Kohat = np.fft.fft2(self.Ko)
        offsets = np.rint(np.fft.fftfreq(config.n)*config.n).astype(int)
        self.offset_x, self.offset_y = np.meshgrid(offsets, offsets)
        self.shell_labels = np.maximum(np.abs(self.offset_x), np.abs(self.offset_y))
        self.radii = np.arange(config.n//2+1)
        masks = self.shell_labels[None] == self.radii[:, None, None]
        self.K_shells = masks*self.K
        self.Ke_shells = masks*self.Ke
        self.Ko_shells = masks*self.Ko
        self._shell_hats = {"K": np.fft.fft2(self.K_shells),
                            "Ke": np.fft.fft2(self.Ke_shells),
                            "Ko": np.fft.fft2(self.Ko_shells)}
        self._lk_shell_origin = laplacian(self.K_shells)[:, 0, 0]
        self._lk_origin = float(laplacian(self.K)[0, 0])

    def _field(self, phi: np.ndarray) -> np.ndarray:
        phi = np.asarray(phi, dtype=np.float64)
        if phi.shape != self.shape or not np.all(np.isfinite(phi)):
            raise ValueError(f"phi must be finite with shape {self.shape}.")
        return phi

    def convolve(self, phi: np.ndarray, kernel_hat: np.ndarray | None = None) -> np.ndarray:
        return np.fft.ifft2(np.fft.fft2(phi) * (self.Khat if kernel_hat is None else kernel_hat)).real

    def mu(self, phi: np.ndarray) -> np.ndarray:
        """Full chemical potential, including the specified reciprocal completion."""
        p = self._field(phi)
        c = self.config
        linear_hat = self.Kehat if c.reciprocal_linear else self.Khat
        result = c.a*p + c.b*p**3 + c.kappa*laplacian(p) + c.g1*self.convolve(p, linear_hat)
        if c.model_id == "M00":
            gx, gy = gradients(p)
            result += c.lambda0*(gx*gx + gy*gy)
        elif c.model_id in {"M01", "M02", "M04"}:
            result += c.g2*self.convolve(p*p)
            if c.eta:
                result += 2*c.eta*c.g2*p*self.convolve(p, self.Khat.conj())
        elif c.model_id == "M05":
            u = self.convolve(p)
            result += c.g2*u*u
            if c.eta:
                result += 2*c.eta*c.g2*self.convolve(p*u, self.Khat.conj())
        return result

    chemical_potential = mu

    def drift(self, phi: np.ndarray) -> np.ndarray:
        return -laplacian(self.mu(phi))

    def mu_jvp(self, phi: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Exact chemical-potential Jacobian-vector product; no stored dense Jacobian."""
        p, d, c = self._field(phi), self._field(direction), self.config
        linear_hat = self.Kehat if c.reciprocal_linear else self.Khat
        out = (c.a + 3*c.b*p*p)*d + c.kappa*laplacian(d) + c.g1*self.convolve(d, linear_hat)
        if c.model_id == "M00":
            gx, gy = gradients(p)
            dx, dy = gradients(d)
            out += 2*c.lambda0*(gx*dx+gy*dy)
        elif c.model_id in {"M01", "M02", "M04"}:
            out += 2*c.g2*self.convolve(p*d)
            if c.eta:
                out += 2*c.eta*c.g2*(d*self.convolve(p, self.Khat.conj())
                                             + p*self.convolve(d, self.Khat.conj()))
        elif c.model_id == "M05":
            u, du = self.convolve(p), self.convolve(d)
            out += 2*c.g2*u*du
            if c.eta:
                out += 2*c.eta*c.g2*self.convolve(d*u+p*du, self.Khat.conj())
        return out

    @staticmethod
    def _group(mechanism: str, arrays: np.ndarray, max_distance: int | None) -> dict[str, Any]:
        last = len(arrays)-1
        if max_distance is not None:
            if isinstance(max_distance, bool) or int(max_distance) != max_distance or max_distance < 0:
                raise ValueError("max_distance must be a nonnegative integer.")
            last = min(last, int(max_distance))
        out = {f"{mechanism}:r{r}": arrays[r] for r in range(last+1)}
        if last < len(arrays)-1:
            out[f"{mechanism}:tail"] = arrays[last+1:].sum(axis=0)
        return out

    def active_channels(self, phi: np.ndarray, max_distance: int | None = None) -> dict[str, np.ndarray]:
        """Model-defined signed work channels. M05 uses the u*u_r convention."""
        p, c = self._field(phi), self.config
        out: dict[str, np.ndarray] = {}
        if c.model_id == "M00":
            gx, gy = gradients(p)
            key = "local:tail" if max_distance is not None and max_distance < 1 else "local:r1"
            return {key: c.lambda0*(gx*gx+gy*gy)}
        if c.model_id in {"M03", "M04"}:
            amplitude = 0.0 if c.reciprocal_linear else c.g1
            fields = amplitude*self.convolve(p, self._shell_hats["Ko"])
            out.update(self._group("directional", fields, max_distance))
        if c.model_id in {"M01", "M02", "M04"}:
            fields = (1-c.eta)*c.g2*self.convolve(p*p, self._shell_hats["K"])
            out.update(self._group("source", fields, max_distance))
        if c.model_id == "M05":
            fields = (1-c.eta)*c.g2*self.convolve(p)*self.convolve(p, self._shell_hats["K"])
            out.update(self._group("receiver", fields, max_distance))
        return out

    def angular_channels(self, phi: np.ndarray, max_distance: int | None = None) -> dict[str, np.ndarray]:
        """Source H/V sector diagnostic, with each diagonal split equally.

        The two sectors sum to each source shell exactly (to roundoff). These
        simulator observables do not change the neural-network input or masks.
        """
        if self.config.model_id not in {"M01", "M02", "M04"}:
            raise ValueError("Angular source sectors apply to M01, M02 and M04.")
        p = self._field(phi)
        x, y = np.abs(self.offset_x), np.abs(self.offset_y)
        horizontal = (x > y).astype(float)+0.5*(x == y)
        coefficient = (1-self.config.eta)*self.config.g2
        fields_h = coefficient*self.convolve(p*p, np.fft.fft2(self.K_shells*horizontal))
        fields_v = coefficient*self.convolve(p*p, np.fft.fft2(self.K_shells*(1-horizontal)))
        return {**self._group("source_H", fields_h, max_distance),
                **self._group("source_V", fields_v, max_distance)}

    def active_mu(self, phi: np.ndarray) -> np.ndarray:
        return sum(self.active_channels(phi).values(), np.zeros(self.shape))

    def passive_channels(self, phi: np.ndarray, max_distance: int | None = None) -> dict[str, np.ndarray]:
        """Quadratic reciprocal shell chemical potentials, for boundary-work checks."""
        return self._group("passive", self.config.g1*self.convolve(phi, self._shell_hats["Ke"]), max_distance)

    def active_channel_traces(self, phi: np.ndarray, max_distance: int | None = None) -> dict[str, float]:
        """Tr(L J_channel), including receiver derivatives and nonzero field mean.

        Receiver: J_r=c[diag(u_r)K+diag(u)K_r]. Cyclic trace and constant
        convolution diagonals give c[(LK)_00 sum(u_r)+(LK_r)_00 sum(u)].
        For local AMB, Tr(L diag(G phi)G)=0 since the diagonal of GL is zero.
        """
        p, c = self._field(phi), self.config
        out: dict[str, Any] = {}
        if c.model_id == "M00":
            return {key: 0.0 for key in self.active_channels(p, max_distance)}
        if c.model_id in {"M03", "M04"}:
            out.update(self._group("directional", np.zeros(len(self.radii)), max_distance))
        coefficient = (1-c.eta)*c.g2
        if c.model_id in {"M01", "M02", "M04"}:
            values = 2*coefficient*self._lk_shell_origin*p.sum()
            out.update(self._group("source", values, max_distance))
        if c.model_id == "M05":
            u = self.convolve(p)
            ur = self.convolve(p, self._shell_hats["K"])
            values = coefficient*(self._lk_origin*ur.sum(axis=(-2,-1))+self._lk_shell_origin*u.sum())
            out.update(self._group("receiver", values, max_distance))
        return {key: float(value) for key, value in out.items()}

    def passive_free_energy(self, phi: np.ndarray) -> float:
        """F_E whose derivative is mu minus active_mu (also for asymmetric K)."""
        p, c = self._field(phi), self.config
        energy = np.sum(c.a*p*p/2+c.b*p**4/4) + c.kappa*np.sum(p*laplacian(p))/2
        energy += c.g1*np.sum(p*self.convolve(p, self.Kehat))/2
        if c.model_id in {"M01", "M02", "M04"}:
            energy += c.eta*c.g2*np.sum(p*self.convolve(p*p))
        elif c.model_id == "M05":
            energy += c.eta*c.g2*np.sum(p*self.convolve(p)**2)
        return float(energy)

    def linear_symbol(self, phi_bar: float | None = None) -> np.ndarray:
        """Jacobian Fourier symbol around a spatially uniform field."""
        c = self.config
        mean = c.phi_bar if phi_bar is None else float(phi_bar)
        linear = self.Kehat if c.reciprocal_linear else self.Khat
        symbol = c.a + 3*c.b*mean**2 + c.kappa*self.Lhat + c.g1*linear
        if c.model_id in {"M01", "M02", "M04"}:
            symbol = symbol + 2*c.g2*mean*self.Khat + 2*c.eta*c.g2*mean*(1+self.Khat.conj())
        elif c.model_id == "M05":
            symbol = symbol + 2*c.g2*mean*self.Khat + 2*c.eta*c.g2*mean*self.Khat.conj()*(1+self.Khat)
        return symbol

    def noise(self, rng: np.random.Generator, dt: float) -> np.ndarray:
        """Independent bond noise with covariance 2 D dt L and conserved sum."""
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite.")
        bonds = rng.standard_normal((2, *self.shape))
        divergence = (bonds[0]-np.roll(bonds[0], 1, axis=-1)
                      + bonds[1]-np.roll(bonds[1], 1, axis=-2))
        return np.sqrt(2*self.config.D*dt)*divergence

    def initial_field(self, rng: np.random.Generator, scale: float = 0.05, *,
                      mode: str = "random", circle_radius_fraction: float = 0.25,
                      circle_width: float = 1.0, circle_amplitude: float = 1.0) -> np.ndarray:
        """Initialize a fixed-mean random field or a smooth periodic circle.

        The circle is centered at (n/2,n/2), with radius fraction*n and a tanh
        interface of width circle_width lattice sites. circle_amplitude is the
        tanh half-contrast before the one-time mean adjustment; it does not
        depend on the model's a or b. scale optionally adds Gaussian perturbations.
        A scale=0 circle consumes no RNG draws. The historical random mode keeps
        its exact draw sequence, including its scale=0 behavior. Initializing a
        circle changes no model coefficients and does not imply a stable droplet.
        """
        if not np.isfinite(scale) or scale < 0:
            raise ValueError("scale must be finite and nonnegative.")
        if mode not in {"random", "circle"}:
            raise ValueError("mode must be 'random' or 'circle'.")
        if mode == "random":
            p = scale*rng.standard_normal(self.shape)
        else:
            parameters = (circle_radius_fraction, circle_width, circle_amplitude)
            if not np.all(np.isfinite(parameters)):
                raise ValueError("Circle parameters must be finite.")
            if not 0 < circle_radius_fraction <= 0.5:
                raise ValueError("circle_radius_fraction must lie in (0, 0.5].")
            if circle_width <= 0 or circle_amplitude <= 0:
                raise ValueError("circle_width and circle_amplitude must be positive.")
            n = self.config.n
            center = n/2
            offsets = (np.arange(n, dtype=np.float64)-center+n/2) % n-n/2
            distance = np.hypot(offsets[:, None], offsets[None, :])
            p = circle_amplitude*np.tanh((circle_radius_fraction*n-distance)/circle_width)
            if scale:
                p = p+scale*rng.standard_normal(self.shape)
        return p-p.mean()+self.config.phi_bar

    def gaussian_reference(self, max_distance: int | None = None) -> dict[str, Any]:
        """Continuous-time fixed-mass Gaussian covariance and physical shell EPR.

        covariance_spectrum is the eigenvalue D/H of C (not E[abs(FFT phi)^2],
        which equals N times that value). The zero mass mode is exactly zero.
        """
        c = self.config
        if c.b != 0 or c.g2 != 0 or c.lambda0 != 0:
            raise ValueError("Gaussian reference requires b=g2=lambda0=0.")
        symbol = self.linear_symbol()
        h = symbol.real
        mask = self.Lhat > 0
        if np.min(h[mask]) <= 0:
            raise ValueError("Gaussian covariance requires H>0 on every nonzero mass mode.")
        covariance = np.zeros(self.shape)
        covariance[mask] = c.D/h[mask]
        a_hat = np.zeros(self.shape, complex) if c.reciprocal_linear else c.g1*self.Kohat
        integrand = np.zeros(self.shape)
        integrand[mask] = (np.abs(a_hat[mask])**2*self.Lhat[mask]/h[mask])
        channels: dict[str, float] = {}
        if c.model_id in {"M03", "M04"}:
            ar = (0 if c.reciprocal_linear else c.g1)*self._shell_hats["Ko"]
            values = np.sum((ar.conj()*a_hat).real* self.Lhat * covariance/c.D, axis=(-2,-1))
            channels = {key: float(value) for key, value in self._group("directional", values, max_distance).items()}
        return {"sigma_total": float(integrand.sum()), "sigma_density": float(integrand.sum()/self.N),
                "sigma_channels": channels, "min_H": float(np.min(h[mask])),
                "covariance_spectrum": covariance, "covariance_kernel": np.fft.ifft2(covariance).real,
                "Hhat": h, "Ahat": a_hat, "mass_mode_variance": 0.0,
                "convention": "continuous_time_fixed_mass"}

    def gaussian_field(self, rng: np.random.Generator) -> np.ndarray:
        """Draw exactly from the stationary continuous-time Gaussian reference."""
        covariance = self.gaussian_reference()["covariance_spectrum"]
        p = np.fft.ifft2(np.fft.fft2(rng.standard_normal(self.shape))*np.sqrt(covariance)).real
        return p+self.config.phi_bar

    def step(self, phi: np.ndarray, dt: float, rng: np.random.Generator, *,
             atol: float = 1e-10, rtol: float = 1e-8, maxiter: int = 20) -> tuple[np.ndarray, dict[str, Any]]:
        """One fixed-noise implicit midpoint step using Newton/GMRES.

        Noise is sampled once, including when Newton needs a line search. Failed
        solves raise, without clipping, redrawing noise, or changing dt. There is
        no field-mean reset; mass_correction is always zero.
        """
        current = self._field(phi)
        if atol <= 0 or rtol < 0 or maxiter < 1 or not np.isfinite(atol+rtol):
            raise ValueError("Require finite atol>0, rtol>=0 and maxiter>=1.")
        noise = self.noise(rng, dt)
        c = self.config
        if c.b == 0 and c.g2 == 0 and c.lambda0 == 0:
            factor = 0.5*dt*self.Lhat*self.linear_symbol()
            next_phi = np.fft.ifft2(((1-factor)*np.fft.fft2(current)+np.fft.fft2(noise))/(1+factor)).real
            residual = next_phi-current+dt*laplacian(self.mu((current+next_phi)/2))-noise
            residual_norm = float(np.linalg.norm(residual))
            iterations = 0
        else:
            next_phi = current+noise
            iterations = 0
            for iterations in range(maxiter+1):
                midpoint = (current+next_phi)/2
                residual = next_phi-current+dt*laplacian(self.mu(midpoint))-noise
                residual_norm = float(np.linalg.norm(residual))
                tolerance = atol+rtol*np.linalg.norm(next_phi)
                if residual_norm <= tolerance:
                    break
                if iterations == maxiter or not np.isfinite(residual_norm):
                    raise MidpointConvergenceError(f"Midpoint failed after {iterations} iterations; residual={residual_norm:.3g}. Restart at smaller dt.")
                def matvec(vector: np.ndarray) -> np.ndarray:
                    d = vector.reshape(self.shape)
                    return (d+0.5*dt*laplacian(self.mu_jvp(midpoint, d))).ravel()
                matrix = LinearOperator((self.N, self.N), matvec=matvec, dtype=np.float64)
                precond_symbol = (1+0.5*dt*self.Lhat*(c.a+3*c.b*np.mean(midpoint**2)
                                  +c.kappa*self.Lhat+c.g1*self.Kehat.real))
                if np.min(np.abs(precond_symbol)) < 1e-10:
                    precond_symbol = np.ones(self.shape)
                precond = LinearOperator((self.N, self.N),
                    matvec=lambda vector: np.fft.ifft2(np.fft.fft2(vector.reshape(self.shape))/precond_symbol).real.ravel(),
                    dtype=np.float64)
                update, status = gmres(matrix, -residual.ravel(), M=precond,
                                       atol=max(atol*0.01, 1e-14), rtol=1e-7,
                                       restart=min(self.N, 30), maxiter=10)
                if status != 0 or not np.all(np.isfinite(update)):
                    raise MidpointConvergenceError(f"Midpoint GMRES failed (status={status}); restart at smaller dt.")
                update = update.reshape(self.shape)
                fraction = 1.0
                for _ in range(12):
                    trial = next_phi+fraction*update
                    with np.errstate(over="ignore", invalid="ignore"):
                        trial_midpoint = (current+trial)/2
                        if np.all(np.isfinite(trial_midpoint)):
                            trial_residual = trial-current+dt*laplacian(self.mu(trial_midpoint))-noise
                            trial_norm = float(np.linalg.norm(trial_residual))
                        else:
                            trial_norm = np.inf
                    if np.isfinite(trial_norm) and trial_norm < residual_norm:
                        next_phi = trial
                        break
                    fraction *= 0.5
                else:
                    raise MidpointConvergenceError("Midpoint line search failed; restart the run at smaller dt.")
        tolerance = float(atol+rtol*np.linalg.norm(next_phi))
        if not np.isfinite(residual_norm) or residual_norm > tolerance:
            raise MidpointConvergenceError(f"Midpoint residual {residual_norm:.3g} exceeds tolerance {tolerance:.3g}.")
        return next_phi, {"residual_norm": residual_norm, "tolerance": tolerance,
                          "iterations": iterations, "mass_error": float(next_phi.sum()-current.sum()),
                          "mass_correction": 0.0, "solver": "implicit_midpoint_newton_gmres"}
