# M03 — Polar linear kernel from a driven mediator

Version 1.0 | 역할: 물리적 directional nonreciprocity + 해석 가능한 EPR calibration
우선순위: M00 다음 M01과 병행. fixed architecture의 정확도 검증에 가장 명확한 선형 극한을 제공한다.
공통 규약: COMMON_PROTOCOL.md. NN/trajectory 실험을 완료한 결과가 아니라 구현 명세이다.

## 1. 물리적 출발점

기판 위의 입자는 직접 advect되지 않고, 입자가 방출한 신호만 상대적으로 흐르는 매질에 의해 운반된다.

    c_t + u dot grad c = D_c Delta c - k*c + q0*rho

이 differential-flow chemotaxis의 실제 선행연구는 Muñoz-García와 Neufeld [R5]이다. 빠른 c를 제거하면

    c = (q0/k) K_u rho
    K_u = k*(k-D_c Delta+u dot grad)^(-1)

가 된다. receiver의 선형 response를 g1에 흡수한다. g1의 부호는 attraction/repulsion response에 따라 달라질 수 있다.

최종 simulator/NN 변수는 phi 하나이다. 외부 u는 동적 vector order parameter가 아니라 고정 protocol이다. fast-mediator scalar approximation 및 constant-mobility truncation은 [R2]의 일반적 방법에 근거한 선택이다. 원 논문 [R5]가 아래 stochastic scalar AMB와 kernel EPR까지 직접 유도한 것은 아니다.

## 2. 정확한 scalar 방정식

    dphi = -L mu dt + sqrt(2D) B dW,    B B^T=L
    mu = a phi + b phi^3 + kappa L phi + g1 K phi

primary calibration: b=0, a>0, kappa>0, D>0, fixed phi_bar=0.
nonlinear extension: b>0. 이 문서의 main model은 g2=0, lambda0=0이다. quadratic source를 추가하려면 M04로 구분한다.

    K_e=(K+K^T)/2,  K_o=(K-K^T)/2
    F_E=F0+g1*phi^T K_e phi/2
    mu_A=g1*K_o phi

등방적 AMB와 같은 symmetry class라고 주장하지 않는다. 고정된 polar transport direction이 inversion/rotation symmetry를 깬다.

## 3. Kernel의 수치 정의: positivity-preserving transport

h=1에서 ell_x^2=D_c,x/k, ell_y^2=D_c,y/k, v=u/k이다. continuum operator를 naive central-advection stencil로 만들면 강한 advection에서 음의 Green weights가 생길 수 있으므로, 이 명세는 upwind stencil을 기본으로 고정한다.

    A_K = I + ell_x^2 L_x + ell_y^2 L_y
          + vx_plus*(I-T_xminus) + vx_minus*(I-T_xplus)
          + vy_plus*(I-T_yminus) + vy_minus*(I-T_yplus)
    vx_plus=max(vx,0),  vx_minus=max(-vx,0)
    (T_xminus f)_i=f_(i-x)
    K=A_K^(-1)

Fourier 구현:

    Khat(q) = 1 / {1 + ell_x^2 Lxhat + ell_y^2 Lyhat
                     + |vx|*(1-cos q_x) + |vy|*(1-cos q_y)
                     + i*[vx*sin q_x + vy*sin q_y]}

    K=real(ifft2(Khat))

kernel impulse가 기대하는 downstream 방향으로 이동하는지 직접 검사한다. upwind의 |v|(1-cos q) 항은 수치 diffusion이며 실제 diffusion parameter에 조용히 흡수하지 않는다. 이 stencil 자체를 finite-grid model의 일부로 기록한다.

정규화 sum K=1, numerical tolerance 내 K_s>=0, K_(-v)(s)=K_v(-s), K_e^T=K_e, K_o^T=-K_o를 검사한다. v=0이면 M01/M00의 even Green kernel과 같아야 한다.

ell과 v는 각각 diffusive range와 advective displacement scale이다. 등방적 continuum에서는 Pe=|u|/sqrt(D_c*k)=|v|/ell이다. Pe가 같아도 ell, grid cutoff, periodic box size가 다르면 동일한 이산 kernel이 아니다.

## 4. 시간반전과 물리적 범위

이 명세의 EPR은 동일한 v에서 P_v[phi]/P_v[time-reversed phi]를 비교한다. denominator에서 v를 반전하지 않는다. v->-v는 별도의 physical control experiment이다. pump/flow의 전체 thermodynamic dissipation을 scalar EPR과 동일시하지 않는다.

흐름이 입자와 신호를 똑같이 advect하는 경우에는 이 상대 수송 모델이 아니다. 공통 translation을 제거할 수 있는지 먼저 따져야 한다. 여기에서는 신호와 substrate-bound particles 사이의 differential transport를 가정한다.

small-q에서 K_o의 leading term은 -(u/k) dot grad이며, conserved dynamics에는 (u dot grad)*Delta phi 형태의 q^3 contribution을 준다. 이것은 단순 uniform q-linear advection과 다르다. 모든 mode를 하나의 이동 좌표계로 제거할 수 있다고 가정하지 않는다.

## 5. Kernel EPR의 정확한 정의

    nu_r = g1*K_o,r phi
    sigma_r = -(g1/D)*E[(K_o,r phi)^T o dotphi]
    sigma_total=sum_(all r) sigma_r

K_o,r는 inversion-symmetric shell mask로 분할한다. odd part의 r=0은 0이다. even shell g1*K_e,r phi의 stationary work는 0이다.

    J_r=g1*K_o,r
    Tr(L J_r)=0
    sigma_r=E[(g1*K_o,r phi)^T L mu]/D

이는 L이 symmetric convolution이고 K_o,r가 antisymmetric이기 때문이다. 이 조건을 바꾸면 trace를 재검사한다.

pair-time-asymmetry 형태:

    A_ij = E[phi_j o dotphi_i - phi_i o dotphi_j]
    sigma_r = -g1/(2D) * sum_ij (K_o,r)_ij*A_ij

이 부호는 mu_NL=+g1*K phi라는 본 문서 convention에 따른다. 다른 문서의 -g*K convention을 혼합하지 않는다.

functional curl:

    C_mu=2*g1*K_o
    Pi*C_dstar*Pi=-(2*g1/D)*Pi*K_o*Pi

score correction의 Hessian은 대칭이므로 비선형 extension에서도 위 curl 기준은 유지된다. local polynomial과 reciprocal K_e는 curl에 기여하지 않는다.

## 6. Gaussian 극한: ground truth를 해석적으로 계산

b=0, fixed mass subspace, translationally invariant periodic operators에서

    H=a*I+kappa*L+g1*K_e
    A=g1*K_o

를 둔다. 모든 nonzero mass mode에서 H>0이어야 한다. H,A,L이 commute하는 기준 discretization에서는

    dphi=-L(H+A)phi dt+sqrt(2D L) dW
    P_ss proportional exp[-phi^T H phi/(2D)]
    C_ss=D*H^(-1)
    d_star=-A phi/D

이다. inverse와 trace는 고정 질량 subspace에서 취한다.

    sigma_total=Tr(A^T L A H^(-1))
    sigma_r=Tr(A_r^T L A H^(-1)),    A_r=g1*K_o,r

HSR이나 frequency-resolved EP가 아니라 실공간 operator trace와 kernel channel 정의이다. FFT를 수치 계산에 사용해도 decomposition 기준은 physical kernel이다.

독립 검증: B_drift=L(H+A)를 만들고 Lyapunov equation

    B_drift*C + C*B_drift^T = 2D L

를 reduced mass subspace에서 풀어 C_ss와 비교한다. probability-current velocity V=-B_drift + D L C^(-1)에서

    sigma=Tr[V^T (D L)^(-1) V C]

를 계산하여 kernel trace 합과 비교한다.

Gaussian sigma는 D와 무관하지만 sampling/NN 난이도는 D에 의존할 수 있다. 각 sigma_r는 negative일 수 있다. h나 grid cutoff가 바뀌면 trace의 UV contributions도 바뀔 수 있으므로 fixed-cutoff 비교를 우선한다.

finite observation lag에서의 최적 endpoint score가 이 continuous-time d_star와 정확히 같다고 가정하지 않는다. dt/delta_t convergence를 해야 한다.

## 7. Pilot configuration 및 controls

```yaml
model_id: M03
Nx: 32
Ny: 32
h: 1.0
phi_bar: 0.0
a: 1.0
b: 0.0
kappa: 2.0
g1: 0.25
D: 0.30
dt: 0.001
kernel:
  family: green_polar_upwind
  ell_x: 1.2
  ell_y: 1.2
  vx: 1.2
  vy: 0.0
```

필수 controls:

| 조건 | 예측 |
|---|---|
| v=0 | reciprocal kernel, EPR=0 |
| g1=0 | no nonlocal drive, EPR=0 |
| K를 K_e로 치환 | 동일 even kernel만 남고 EPR=0 |
| v->-v 및 공간 반사 | ensemble total EPR 동일; odd kernel 부호 반전 |
| b=0, H>0 | covariance/force/trace 정답과 일치 |

v 반전 control은 비교계의 예측이지 time-reversal denominator의 parameter를 바꾸라는 뜻이 아니다.

    vx: [0, 0.4, 1.2, 2.4, -1.2]
    vy: [0]  # x/y swap test는 별도
    ell_x=ell_y: [1.2,2.0,4.0]
    g1: [0,0.25,0.50,-0.50]
    D: [0.10,0.30]

각 Gaussian run 전에 min eigenvalue(H)>0인지 검사한다. 통과 후 b=1로 바꾸어 nonlinear score가 추가된 extension을 수행한다. a<0 phase-separating extension에는 Gaussian 정답을 적용하지 않는다.

## 8. 현재 architecture와의 대응

nu_r는 같은 shell 입력의 한 linear feature이다. high-rank quadratic source와 cross-shell product 문제가 없으므로 구조적으로 가장 단순한 test다. hidden nonlinear layers를 유지한 실제 NN의 학습 성공까지 보장하는 것은 아니다.

Gaussian case에서는 d_star 자체가 physical odd-kernel force이며, R 밖 tail이 작다면 branch/curl/total 세 수준에서 비교할 명확한 기준이 있다. R 밖 tail이 유의하면 정확한 global force의 표현을 주장하지 않는다. 전체 convolution weight의 raw sign은 Conv2d correlation convention을 반영해 비교한다.

b>0에서는 d_star에 nonlinear score correction이 생긴다. 그때도 raw branch sigma가 physical sigma_r에 일치하는지는 별도 검증한다. 전체 learned-force curl은 K_o를 식별하는 gauge-invariant 진단으로 남는다.

## 9. Go / no-go

필수 통과: Gaussian covariance, Lyapunov EPR, kernel-shell trace 합, reciprocal null, reflection control, finite-dt/lag convergence.

불일치 진단 순서: mass zero-mode -> convolution orientation -> noise covariance -> kernel parity/partition -> normalization N,D -> solver/lag -> tail -> NN optimization/representation.

보고값:

    v, ell_x, ell_y, min_H, Pe,
    sigma_trace, sigma_lyapunov, sigma_midpoint, sigma_drift,
    sigma_kernel_r, sigma_tail, NN_raw_r, NN_projected_r,
    covariance_error, projected_curl_error, confidence_intervals

이 model_id의 결과로 가능한 주장: '물리적 differential transport에서 유래한 polar interaction의 scalar kernel EPR을 검증했다.' 불가능한 주장: '등방적 scalar AMB의 같은 symmetry class를 그대로 유지했다' 또는 'flow의 총 연료 소비를 측정했다'.

## 10. 참고문헌

[R1] Nardini et al., PRX 7, 021007 (2017), DOI 10.1103/PhysRevX.7.021007.
[R2] Dinelli, O'Byrne and Tailleur, J. Phys. A 57, 395002 (2024), DOI 10.1088/1751-8121/ad72bc.
[R5] Muñoz-García and Neufeld, PRE 80, 061902 (2009), DOI 10.1103/PhysRevE.80.061902.
Gaussian trace, upwind kernel, NN 검증 protocol은 여기서 명세한 이론/수치 설계이며, R5의 직접적인 EPR 결과가 아니다.