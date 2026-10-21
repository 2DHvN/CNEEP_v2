"""Batched float64 Torch backend for the same fixed-cutoff AMB lattice models.

Spatial FFTs, nonlinear chemical potentials and implicit midpoint solves remain
on the selected device. The batch axis contains independent replicas; no spatial
or stochastic coupling is introduced between them. ``device='cpu'`` is provided
for backend parity tests, while dataset acceleration explicitly selects CUDA.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from .model import AMBKernelModel, ModelConfig, MidpointConvergenceError


def laplacian(field: torch.Tensor) -> torch.Tensor:
    """Positive lattice Laplacian L=-Delta on the final two axes."""
    return (4*field-torch.roll(field,1,-1)-torch.roll(field,-1,-1)
            -torch.roll(field,1,-2)-torch.roll(field,-1,-2))


def gradients(field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return ((torch.roll(field,-1,-1)-torch.roll(field,1,-1))/2,
            (torch.roll(field,-1,-2)-torch.roll(field,1,-2))/2)


class TorchAMBKernelModel:
    """Full physical kernels and per-replica implicit solves on a Torch device.

    All field APIs require float64 tensors [batch, Ny, Nx] on ``self.device``.
    Kernel coefficients are copied exactly from the reference NumPy model;
    custom periodic kernels are neither regenerated nor truncated.
    """
    laplacian = staticmethod(laplacian)
    spec_version = AMBKernelModel.spec_version
    solver_name = "torch_implicit_midpoint_preconditioned_residual"

    def __init__(self, reference_or_config: AMBKernelModel | ModelConfig, device="cuda"):
        self.reference = (reference_or_config if isinstance(reference_or_config,AMBKernelModel)
                          else AMBKernelModel(reference_or_config))
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA simulation was requested, but this PyTorch installation has no available CUDA device.")
        self.device = torch.empty((),device=requested).device
        self.dtype = torch.float64
        self.config = self.reference.config
        self.shape, self.N = self.reference.shape, self.reference.N
        def tensor(value, dtype=torch.float64):
            return torch.as_tensor(np.array(value,copy=True),dtype=dtype,device=self.device)
        for name in ("K","Ke","Ko","K_shells","Ke_shells","Ko_shells","Lhat"):
            setattr(self,name,tensor(getattr(self.reference,name)))
        for name in ("Khat","Kehat","Kohat"):
            setattr(self,name,tensor(getattr(self.reference,name),torch.complex128))
        self.shell_labels = tensor(self.reference.shell_labels,torch.int64)
        self.radii = tensor(self.reference.radii,torch.int64)
        self._shell_hats = {key:tensor(value,torch.complex128)
                            for key,value in self.reference._shell_hats.items()}
        self._lk_shell_origin = tensor(self.reference._lk_shell_origin)
        self._lk_origin = self.reference._lk_origin
        self._linear_hat = self.Kehat if self.config.reciprocal_linear else self.Khat
        self._linear_symbol = (self.config.a+self.config.kappa*self.Lhat
                               +self.config.g1*self._linear_hat)
        x,y = np.abs(self.reference.offset_x),np.abs(self.reference.offset_y)
        horizontal = (x>y).astype(float)+0.5*(x==y)
        self._angular_hats = {
            "source_H":tensor(np.fft.fft2(self.reference.K_shells*horizontal),torch.complex128),
            "source_V":tensor(np.fft.fft2(self.reference.K_shells*(1-horizontal)),torch.complex128),
        }

    def _field(self, phi: torch.Tensor) -> torch.Tensor:
        if not isinstance(phi,torch.Tensor) or phi.ndim != 3 or tuple(phi.shape[-2:]) != self.shape or phi.shape[0]<1:
            raise ValueError(f"Field must be a tensor [batch,{self.shape[0]},{self.shape[1]}] with batch>=1.")
        if phi.dtype != torch.float64 or phi.device != self.device:
            raise ValueError(f"Field must use float64 on {self.device}; conversion is never implicit.")
        return phi

    def convolve(self, phi: torch.Tensor, kernel_hat: torch.Tensor | None = None) -> torch.Tensor:
        """FFT-origin convolution; stacked kernels produce [batch,shell,Ny,Nx]."""
        kernel = self.Khat if kernel_hat is None else kernel_hat
        transformed = torch.fft.fft2(phi)
        if kernel.ndim == 3:
            transformed = transformed[:,None]
            kernel = kernel[None]
        return torch.fft.ifft2(transformed*kernel).real

    def mu(self, phi: torch.Tensor) -> torch.Tensor:
        p,c = self._field(phi),self.config
        out = c.a*p+c.b*p**3+c.kappa*laplacian(p)+c.g1*self.convolve(p,self._linear_hat)
        if c.model_id == "M00":
            gx,gy = gradients(p)
            out = out+c.lambda0*(gx*gx+gy*gy)
        elif c.model_id in {"M01","M02","M04"}:
            out = out+c.g2*self.convolve(p*p)
            if c.eta:
                out = out+2*c.eta*c.g2*p*self.convolve(p,self.Khat.conj())
        elif c.model_id == "M05":
            u = self.convolve(p)
            out = out+c.g2*u*u
            if c.eta:
                out = out+2*c.eta*c.g2*self.convolve(p*u,self.Khat.conj())
        return out

    chemical_potential = mu

    def drift(self, phi: torch.Tensor) -> torch.Tensor:
        return -laplacian(self.mu(phi))

    def mu_jvp(self, phi: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        p,d,c = self._field(phi),self._field(direction),self.config
        if p.shape != d.shape:
            raise ValueError("phi and direction must have the same batch shape.")
        out = (c.a+3*c.b*p*p)*d+c.kappa*laplacian(d)+c.g1*self.convolve(d,self._linear_hat)
        if c.model_id == "M00":
            gx,gy = gradients(p)
            dx,dy = gradients(d)
            out = out+2*c.lambda0*(gx*dx+gy*dy)
        elif c.model_id in {"M01","M02","M04"}:
            out = out+2*c.g2*self.convolve(p*d)
            if c.eta:
                out = out+2*c.eta*c.g2*(d*self.convolve(p,self.Khat.conj())+p*self.convolve(d,self.Khat.conj()))
        elif c.model_id == "M05":
            u,du = self.convolve(p),self.convolve(d)
            out = out+2*c.g2*u*du
            if c.eta:
                out = out+2*c.eta*c.g2*self.convolve(d*u+p*du,self.Khat.conj())
        return out

    @staticmethod
    def _group(mechanism: str, values: torch.Tensor, max_distance: int | None):
        last = values.shape[1]-1
        if max_distance is not None:
            if isinstance(max_distance,bool) or int(max_distance)!=max_distance or max_distance<0:
                raise ValueError("max_distance must be a nonnegative integer.")
            last = min(last,int(max_distance))
        result = {f"{mechanism}:r{r}":values[:,r] for r in range(last+1)}
        if last<values.shape[1]-1:
            result[f"{mechanism}:tail"] = values[:,last+1:].sum(dim=1)
        return result

    def active_channels(self, phi: torch.Tensor, max_distance: int | None = None):
        p,c = self._field(phi),self.config
        out = {}
        if c.model_id == "M00":
            gx,gy = gradients(p)
            key = "local:tail" if max_distance is not None and max_distance<1 else "local:r1"
            return {key:c.lambda0*(gx*gx+gy*gy)}
        if c.model_id in {"M03","M04"}:
            coefficient = 0.0 if c.reciprocal_linear else c.g1
            out.update(self._group("directional",coefficient*self.convolve(p,self._shell_hats["Ko"]),max_distance))
        if c.model_id in {"M01","M02","M04"}:
            out.update(self._group("source",(1-c.eta)*c.g2*self.convolve(p*p,self._shell_hats["K"]),max_distance))
        if c.model_id == "M05":
            values = (1-c.eta)*c.g2*self.convolve(p)[:,None]*self.convolve(p,self._shell_hats["K"])
            out.update(self._group("receiver",values,max_distance))
        return out

    def active_mu(self, phi: torch.Tensor) -> torch.Tensor:
        return sum(self.active_channels(phi).values(),torch.zeros_like(phi))

    def passive_channels(self, phi: torch.Tensor, max_distance: int | None = None):
        p = self._field(phi)
        return self._group("passive",self.config.g1*self.convolve(p,self._shell_hats["Ke"]),max_distance)

    def angular_channels(self, phi: torch.Tensor, max_distance: int | None = None):
        p,c = self._field(phi),self.config
        if c.model_id not in {"M01","M02","M04"}:
            raise ValueError("Angular source sectors apply to M01, M02 and M04.")
        out = {}
        for mechanism,kernel in self._angular_hats.items():
            out.update(self._group(mechanism,(1-c.eta)*c.g2*self.convolve(p*p,kernel),max_distance))
        return out

    def active_channel_traces(self, phi: torch.Tensor, max_distance: int | None = None):
        p,c = self._field(phi),self.config
        out = {}
        if c.model_id == "M00":
            return {key:torch.zeros(p.shape[0],dtype=self.dtype,device=self.device)
                    for key in self.active_channels(p,max_distance)}
        if c.model_id in {"M03","M04"}:
            values = torch.zeros((p.shape[0],len(self.radii)),dtype=self.dtype,device=self.device)
            out.update(self._group("directional",values,max_distance))
        coefficient = (1-c.eta)*c.g2
        if c.model_id in {"M01","M02","M04"}:
            values = 2*coefficient*p.sum(dim=(-2,-1))[:,None]*self._lk_shell_origin[None]
            out.update(self._group("source",values,max_distance))
        if c.model_id == "M05":
            u,ur = self.convolve(p),self.convolve(p,self._shell_hats["K"])
            values = coefficient*(self._lk_origin*ur.sum(dim=(-2,-1))
                                  +self._lk_shell_origin[None]*u.sum(dim=(-2,-1))[:,None])
            out.update(self._group("receiver",values,max_distance))
        return out

    def noise_from_bonds(self, bonds: torch.Tensor, dt: float) -> torch.Tensor:
        """Conserved noise from [batch,2,Ny,Nx] independent standard normal bonds."""
        if not math.isfinite(dt) or dt<=0:
            raise ValueError("dt must be positive and finite.")
        if not isinstance(bonds,torch.Tensor) or bonds.ndim!=4 or bonds.shape[1]!=2:
            raise ValueError("bonds must have shape [batch,2,Ny,Nx].")
        self._field(bonds[:,0])
        bx,by = bonds[:,0],bonds[:,1]
        return math.sqrt(2*self.config.D*dt)*(bx-torch.roll(bx,1,-1)+by-torch.roll(by,1,-2))

    def noise(self, batch_size: int, dt: float, generator: torch.Generator | None = None):
        if isinstance(batch_size,bool) or not isinstance(batch_size,int) or batch_size<1:
            raise ValueError("batch_size must be a positive integer.")
        bonds = torch.randn((batch_size,2,*self.shape),dtype=self.dtype,device=self.device,generator=generator)
        return self.noise_from_bonds(bonds,dt)

    @torch.no_grad()
    def step_with_noise(self, phi: torch.Tensor, dt: float, noise: torch.Tensor, *,
                        atol: float = 1e-10, rtol: float = 1e-8, maxiter: int = 20):
        """Solve the exact fixed-noise implicit midpoint equation for every replica.

        A Fourier preconditioner inverts the complete stiff linear midpoint
        operator and the spatially averaged cubic derivative. The nonlinear
        iteration y <- y - P^-1 R(y) uses separate backtracking per replica and
        stops only at ||R||_2 <= atol + rtol*||y||_2 for EACH batch member. It is
        an implicit solve, not a semi-implicit/explicit substitute. Difficult
        parameters may need a smaller dt; failed replicas raise without noise
        redraw, field clipping, mean resetting or a hidden integrator fallback.
        """
        current,noise = self._field(phi),self._field(noise)
        if current.shape != noise.shape:
            raise ValueError("phi and noise must have the same batch shape.")
        if not math.isfinite(dt) or dt<=0:
            raise ValueError("dt must be positive and finite.")
        if (not math.isfinite(atol+rtol) or atol<=0 or rtol<0 or
                isinstance(maxiter,bool) or not isinstance(maxiter,int) or maxiter<1):
            raise ValueError("Require finite atol>0, rtol>=0, and integer maxiter>=1.")
        if not bool(torch.isfinite(current).all() & torch.isfinite(noise).all()):
            raise ValueError("phi and noise must be finite.")
        batch = current.shape[0]
        iterations = torch.zeros(batch,dtype=torch.int64,device=self.device)
        def norm(field):
            return torch.linalg.vector_norm(field.flatten(1),dim=1)
        def residual(next_phi):
            return next_phi-current+dt*laplacian(self.mu((current+next_phi)/2))-noise
        def fail(message,mask):
            indices = torch.nonzero(mask,as_tuple=False).flatten().cpu().tolist()
            raise MidpointConvergenceError(f"{message}; failed batch indices {indices}. Restart at smaller dt.")
        c = self.config
        if c.b==0 and c.g2==0 and c.lambda0==0:
            factor = .5*dt*self.Lhat*self._linear_symbol
            next_phi = torch.fft.ifft2(((1-factor)*torch.fft.fft2(current)+torch.fft.fft2(noise))/(1+factor)).real
        else:
            next_phi = current+noise
            for iteration in range(maxiter+1):
                midpoint = (current+next_phi)/2
                res = residual(next_phi)
                residual_norm = norm(res)
                tolerance = atol+rtol*norm(next_phi)
                nonfinite = ~torch.isfinite(residual_norm)
                if bool(nonfinite.any()):
                    fail("Nonfinite midpoint residual",nonfinite)
                pending = residual_norm>tolerance
                if not bool(pending.any()):
                    break
                if iteration == maxiter:
                    fail("Batched implicit midpoint did not converge",pending)
                local_derivative = 3*c.b*(midpoint*midpoint).mean(dim=(-2,-1))
                symbol = 1+.5*dt*self.Lhat[None]*(self._linear_symbol[None]+local_derivative[:,None,None])
                singular = symbol.abs().amin(dim=(-2,-1))<1e-10
                symbol = torch.where(singular[:,None,None],torch.ones_like(symbol),symbol)
                update = -torch.fft.ifft2(torch.fft.fft2(res)/symbol).real
                update = torch.where(pending[:,None,None],update,torch.zeros_like(update))
                base = next_phi
                fraction = torch.ones((batch,1,1),dtype=self.dtype,device=self.device)
                remaining = pending.clone()
                chosen = next_phi
                for _ in range(12):
                    trial = base+fraction*update
                    trial_norm = norm(residual(trial))
                    accepted = remaining & torch.isfinite(trial_norm) & (trial_norm<residual_norm)
                    chosen = torch.where(accepted[:,None,None],trial,chosen)
                    remaining = remaining & ~accepted
                    if not bool(remaining.any()):
                        break
                    fraction = torch.where(remaining[:,None,None],fraction*.5,fraction)
                if bool(remaining.any()):
                    fail("Batched midpoint line search failed",remaining)
                next_phi = chosen
                iterations = iterations+pending.to(torch.int64)
        final_residual = norm(residual(next_phi))
        tolerance = atol+rtol*norm(next_phi)
        failed = ~torch.isfinite(final_residual) | (final_residual>tolerance)
        if bool(failed.any()):
            fail("Batched midpoint residual exceeds tolerance",failed)
        return next_phi,{"residual_norm":final_residual,"tolerance":tolerance,
                         "iterations":iterations,"mass_error":(next_phi-current).sum(dim=(-2,-1)),
                         "mass_correction":torch.zeros(batch,dtype=self.dtype,device=self.device),
                         "solver":self.solver_name}
