"""Finite-grid algebra and Gaussian OU checks; no trajectory or NN training.
Run: python validation/verify_spec_math.py
Requires Python 3, numpy and scipy. Writes results.json beside this script.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.linalg import null_space, solve_continuous_lyapunov


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a-b) / max(np.linalg.norm(a), np.linalg.norm(b), 1e-14))


def fd_jac(fun, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    eye = np.eye(x.size)
    return np.column_stack([(fun(x+eps*e)-fun(x-eps*e))/(2*eps) for e in eye])


def fd_grad(fun, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    eye = np.eye(x.size)
    return np.array([(fun(x+eps*e)-fun(x-eps*e))/(2*eps) for e in eye])


def main() -> None:
    n = 7
    N = n*n
    I = np.eye(N)
    tx = np.zeros((N,N))
    ty = np.zeros((N,N))
    for y in range(n):
        for x in range(n):
            i=y*n+x
            tx[i,y*n+(x-1)%n]=1
            ty[i,((y-1)%n)*n+x]=1
    Lx=2*I-tx-tx.T
    Ly=2*I-ty-ty.T
    L=Lx+Ly
    B=np.concatenate([I-tx, I-ty], axis=1)
    Q=null_space(np.ones((1,N)))
    Pi=Q@Q.T

    def kernel(ex: float, ey: float, vx: float=0.0, vy: float=0.0) -> np.ndarray:
        op=I+ex*ex*Lx+ey*ey*Ly
        op+=max(vx,0)*(I-tx)+max(-vx,0)*(I-tx.T)
        op+=max(vy,0)*(I-ty)+max(-vy,0)*(I-ty.T)
        return np.linalg.inv(op)

    Ke=kernel(1.2,1.2)
    Kan=kernel(1.2*np.sqrt(2),1.2/np.sqrt(2))
    Kp=kernel(1.2,1.2,1.2,0)
    Kpm=kernel(1.2,1.2,-1.2,0)
    rng=np.random.default_rng(73031)
    phi=rng.normal(scale=0.4,size=N)
    phi=phi-phi.mean()+0.23
    a,b,kappa,g1,g2=1.0,1.0,2.0,0.25,0.20
    local=lambda x: a*x+b*x**3+kappa*L@x
    Jlocal=np.diag(a+3*b*phi**2)+kappa*L

    errors={"noise_factorization":rel(B@B.T,L),
            "mass_conservation_operator":float(np.linalg.norm(L@np.ones(N))),
            "polar_kernel_reversal":rel(Kpm,Kp.T),
            "even_kernel_symmetry":rel(Ke,Ke.T),
            "anisotropic_kernel_symmetry":rel(Kan,Kan.T),
            "kernel_row_normalization":float(np.max(np.abs(Kp.sum(axis=1)-1)))}
    observations={"polar_kernel_min_entry":float(Kp.min())}

    for name,K,eta in [("M01",Ke,0.0),("M02",Kan,0.0),("M04",Kp,0.0),
                       ("source_eta_intermediate",Kp,0.37)]:
        def fun(x,K=K,eta=eta):
            return local(x)+g1*K@x+g2*(K@(x*x)+2*eta*x*(K.T@x))
        J=Jlocal+g1*K+2*g2*K@np.diag(phi)
        J+=2*eta*g2*(np.diag(K.T@phi)+np.diag(phi)@K.T)
        expected=g1*(K-K.T)+2*g2*(1-eta)*(K@np.diag(phi)-np.diag(phi)@K.T)
        errors[name+"_jacobian_fd"]=rel(J,fd_jac(fun,phi))
        errors[name+"_curl_identity"]=rel(J-J.T,expected)
        observations[name+"_projected_curl_norm"]=float(np.linalg.norm(Pi@expected@Pi))

    # The asymmetric cubic functional must yield both source and receiver terms.
    Fsource=lambda x: g2*float(x@(Kp@(x*x)))
    musource=g2*(Kp@(phi*phi)+2*phi*(Kp.T@phi))
    errors["source_cubic_variation"]=rel(fd_grad(Fsource,phi),musource)
    Jcomp=2*g2*(Kp@np.diag(phi)+np.diag(Kp.T@phi)+np.diag(phi)@Kp.T)
    errors["source_completion_curl"]=float(np.linalg.norm(Pi@(Jcomp-Jcomp.T)@Pi))

    for eta in [0.0,0.37,1.0]:
        u=Ke@phi
        def recv(x,eta=eta):
            ux=Ke@x
            return local(x)+g1*ux+g2*(ux*ux+2*eta*Ke.T@(x*ux))
        J=Jlocal+g1*Ke+2*g2*np.diag(u)@Ke
        J+=2*eta*g2*(Ke.T@np.diag(u)+Ke.T@np.diag(phi)@Ke)
        expected=2*g2*(1-eta)*(np.diag(u)@Ke-Ke.T@np.diag(u))
        errors[f"M05_eta_{eta}_jacobian_fd"]=rel(J,fd_jac(recv,phi))
        errors[f"M05_eta_{eta}_curl_identity"]=float(np.linalg.norm((J-J.T)-expected))
    Frecv=lambda x: g2*float(x@((Ke@x)**2))
    murecv=g2*((Ke@phi)**2+2*Ke.T@(phi*(Ke@phi)))
    errors["receiver_cubic_variation"]=rel(fd_grad(Frecv,phi),murecv)

    # M00 active term and exact quadratic passive midpoint chain rule.
    Gx=(tx.T-tx)/2
    Gy=(ty.T-ty)/2
    activity=lambda x: (Gx@x)**2+(Gy@x)**2
    JA=2*np.diag(Gx@phi)@Gx+2*np.diag(Gy@phi)@Gy
    errors["M00_active_jacobian_fd"]=rel(JA,fd_jac(activity,phi))
    phinext=phi+rng.normal(scale=0.02,size=N)
    phinext-=phinext.mean()-phi.mean()
    midpoint=(phi+phinext)/2
    lhs=float((Ke@midpoint)@(phinext-phi))
    rhs=0.5*float(phinext@Ke@phinext-phi@Ke@phi)
    errors["passive_midpoint_chain_rule"]=abs(lhs-rhs)

    # Distance partition, nonzero-mean Ito correction, receiver cross-shell rule.
    coords=np.array([(y,x) for y in range(n) for x in range(n)])
    d=np.abs(coords[:,None,:]-coords[None,:,:])
    d=np.minimum(d,n-d)
    shells=d.max(axis=2)
    Krs=[Kp*(shells==r) for r in range(n//2+1)]
    Kers=[Ke*(shells==r) for r in range(n//2+1)]
    errors["kernel_partition"]=rel(sum(Krs),Kp)
    correction_errors=[]
    for Kr in Krs:
        Jr=2*g2*Kr@np.diag(phi)
        direct=float(np.trace(L@Jr))
        shortcut=2*g2*(L@Kr)[0,0]*float(phi.sum())
        correction_errors.append(abs(direct-shortcut))
    errors["source_Ito_trace_nonzero_mean"]=max(correction_errors)
    u=Ke@phi
    nus=[g2*u*(Kr@phi) for Kr in Kers]
    jrs=[g2*(np.diag(Kr@phi)@Ke+np.diag(u)@Kr) for Kr in Kers]
    errors["receiver_attribution_sum"]=rel(sum(nus),g2*u*u)
    errors["receiver_attribution_J_sum"]=rel(sum(jrs),2*g2*np.diag(u)@Ke)
    uniform=np.ones((N,N))/N
    Cunif=2*g2*(uniform@np.diag(phi)-np.diag(phi)@uniform)
    errors["uniform_source_mass_projected_curl"]=float(np.linalg.norm(Pi@Cunif@Pi))

    # M03 Gaussian, reduced to the fixed-mass subspace.
    D=0.3
    Ksym=(Kp+Kp.T)/2
    Kodd=(Kp-Kp.T)/2
    H=Q.T@(a*I+kappa*L+g1*Ksym)@Q
    A=Q.T@(g1*Kodd)@Q
    Lr=Q.T@L@Q
    Bd=Lr@(H+A)
    C=solve_continuous_lyapunov(Bd,2*D*Lr)
    Cref=D*np.linalg.inv(H)
    V=-Bd+D*Lr@np.linalg.inv(C)
    rate_current=float(np.trace(V.T@np.linalg.inv(D*Lr)@V@C))
    Hinv=np.linalg.inv(H)
    rate_trace=float(np.trace(A.T@Lr@A@Hinv))
    Ar=[Q.T@(g1*(Kr-Kr.T)/2)@Q for Kr in Krs]
    shellrates=[float(np.trace(Ar_i.T@Lr@A@Hinv)) for Ar_i in Ar]
    errors["M03_gaussian_covariance"]=rel(C,Cref)
    errors["M03_thermodynamic_force"]=rel(np.linalg.solve(D*Lr,V),-A/D)
    errors["M03_current_vs_trace_rate"]=abs(rate_current-rate_trace)
    errors["M03_shell_sum_rate"]=abs(sum(shellrates)-rate_trace)
    errors["M03_symmetrized_kernel_rate"]=float(np.linalg.norm((Ksym-Ksym.T)/2))
    observations.update({"M03_min_H":float(np.linalg.eigvalsh(H).min()),
                         "M03_rate_current":rate_current,"M03_rate_trace":rate_trace,
                         "M03_shell_rates":shellrates})

    tolerance=1e-7
    failed={k:v for k,v in errors.items() if not np.isfinite(v) or v>tolerance}
    if observations["polar_kernel_min_entry"] < -1e-12:
        failed["kernel_positivity"]=observations["polar_kernel_min_entry"]
    source=Path(__file__).resolve()
    out={"scope":"finite-grid algebra and Gaussian OU matrix checks only; no trajectories or NN training",
         "grid":[n,n],"seed":73031,"tolerance":tolerance,
         "script_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
         "errors":errors,"observations":observations,"passed":not failed,"failed":failed}
    target=source.with_name("results.json")
    target.write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(out,indent=2))
    if failed:
        raise SystemExit("Spec algebra validation failed")

if __name__ == "__main__":
    main()
