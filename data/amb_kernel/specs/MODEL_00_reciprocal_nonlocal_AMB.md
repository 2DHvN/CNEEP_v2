# M00 — Reciprocal nonlocal interaction + local AMB

Version 1.0 | 역할: 필수 대조군 / Zwicker-type nonlocal sector의 영향 분리
우선순위: 먼저 실행. 새로운 nonlocal EPR source를 만드는 주 모델은 아니다.
공통 규약: COMMON_PROTOCOL.md. 아래 parameter는 pilot 제안이며 trajectory/NN 검증값이 아니다.

## 1. 목적과 가설

대칭적인 nonlocal interaction은 scalar EPR을 직접 생산하지 않지만, 기존 local AMB activity가 만들어내는 EPR과 morphology를 바꿀 수 있다. 이 둘을 분리하여 이후 M01–M04의 비상호적 kernel 효과와 비교한다.

필수 질문은 'kernel을 넣어서 EPR이 바뀌었는가'가 아니라 '새로운 비가역성의 원천인가, passive statistics의 변화인가'이다.

문헌 기반: Thewes 등의 자유에너지 기반 nonlocal phase separation [R3], Nardini 등의 scalar AMB EPR [R1]. 여기의 screened Green kernel과 polynomial local free energy 조합은 공통 비교를 위한 선택이며 [R3]의 모든 kernel/phase diagram을 그대로 재현하는 모델은 아니다.

## 2. 상태와 정확한 방정식

2D periodic, h=M=1, scalar phi, fixed phi_bar. L=-Delta_h는 공통 5-point stencil이다.

    dphi = -L mu dt + sqrt(2D) B dW,     B B^T=L
    mu = a phi + b phi^3 + kappa L phi + g1 K phi + lambda0 Q(phi)
    Q_i = [(phi_(i+x)-phi_(i-x))/2]^2
        + [(phi_(i+y)-phi_(i-y))/2]^2

b>0, kappa>0, D>0를 기본으로 한다. K=K^T인 kernel만 이 model_id에서 허용한다.

    F = sum_i [a phi_i^2/2 + b phi_i^4/4]
        + kappa*phi^T L phi/2 + g1*phi^T K phi/2
    mu_E = delta F/delta phi
    mu_A = lambda0 Q

전달 매질이나 pair potential에 대응하는 reciprocal contribution을 나타낸다. g1>0은 이 quadratic density functional에서 repulsive penalty에 해당한다. microscopic mediator의 elimination으로 g1의 부호까지 자동으로 정해지는 것은 아니다.

## 3. Kernel 정의

기본 kernel은 정규화된 screened Poisson Green function이다.

    Khat(q) = 1/[1 + ell^2*(Lxhat(q)+Lyhat(q))]
    Lxhat = 4 sin(q_x/2)^2
    Lyhat = 4 sin(q_y/2)^2
    K = real(ifft2(Khat)),    sum_s K_s=1

real-space kernel을 생성한 뒤 inversion symmetry와 sum=1을 검사한다. 임의의 shell cutoff로 자르지 않는다. full kernel의 source shell work와 NN R 밖 tail을 모두 저장한다.

ell -> 0이면 K -> I이며 g1은 local quadratic coefficient의 재정의가 된다. 이 극한에서 새로운 nonlocal physics를 주장하지 않는다. 유한 ell의 kernel은 screened interaction이지 asymptotic power-law interaction이 아니다.

## 4. EPR 정답과 대조군

    sigma_local = -lambda0/D * E[Q^T o dotphi]
    sigma_total = sigma_local

각 passive shell K_r=K_r^T에 대해

    W_passive,r = -g1/D * E[(K_r phi)^T o dotphi] = 0

이다. 이 null은 lambda0가 0이 아니어도 성립한다. finite midpoint step에서는 quadratic F_r의 정확한 endpoint 차이이므로 chain-rule unit test로도 사용한다.

기록해야 할 값:

- 전체 local active EPR 및 midpoint/equal-time 두 estimator.
- passive shell별 work와 해당 quadratic free-energy endpoint 차이.
- S(q), gradient energy, 계면량을 사용할 경우 그 추출 기준.
- NN raw branch, 전체 learned force의 EPR, functional curl residual.

필수 controls:

| 조건 | 예측 |
|---|---|
| lambda0=0, g1 임의 | stationary scalar EPR=0 |
| g1=0 | 표준 이산 AMB |
| ell=0 | local a -> a+g1인 AMB |
| lambda0!=0, g1 또는 ell 변화 | EPR 변화 가능; 부호/단조성 보장 없음 |

## 5. 이론 검증

K의 Jacobian은 symmetric이므로 C_mu,NL=0이다. local AMB의 curl만 남는다. Q의 Jacobian은 central gradient operators G_alpha를 이용해

    J_Q = 2 sum_alpha diag(G_alpha phi) G_alpha

로 계산한다. mass projector Pi를 적용하여 실제 learned-force curl과 비교한다.

균일한 phi_bar 주변의 linear relaxation은

    omega(q) = -Lhat(q)*[a+3b phi_bar^2+kappa Lhat(q)+g1 Khat(q)]

이다. local activity Q는 균일한 배경의 linearization에 기여하지 않는다. q=0은 보존 mode이다. mode stability를 확인하지 않고 'homogeneous pilot'로 run을 분류하지 않는다.

finite-q 패턴이 생길 수 있는 parameter도 있지만 자동으로 생기는 것은 아니다. full lattice dispersion을 먼저 계산한다. phase-separated coarsening이 계속되는 구간은 stationary EPR 측정에서 제외한다.

## 6. Pilot configuration

```yaml
model_id: M00
Nx: 32
Ny: 32
h: 1.0
phi_bar: 0.0
a: 1.0
b: 1.0
kappa: 2.0
g1: 0.25
lambda0: 0.10
D: 0.30
dt: 0.001
kernel:
  family: green_even
  ell_x: 1.2
  ell_y: 1.2
```

순서: lambda0=0 reference -> lambda0=0.1 -> g1/ell 변화. 한 번에 full Cartesian sweep을 돌리지 않는다.

권장 one-at-a-time sweep:

    lambda0: [0, 0.05, 0.10, 0.20]
    g1: [0, 0.25, 0.50]
    ell: [0, 1.2, 2.0, 4.0]
    D: [0.10, 0.30]
    Nx=Ny: [32,64]

ell=4와 R=4는 tail stress test로 취급하며 정확한 NN coverage를 가정하지 않는다. morphology 단계에서는 a를 -0.5로 변경한 별도 run을 시도할 수 있으나 stationarity와 계면 해상도를 다시 검증한다.

## 7. Shell-force와의 연결

Q는 r=1의 축 방향 이웃 네 점만 사용하므로 기존 shell mask와 정보 접근이 맞는다. 두 gradient feature의 quadratic response이며 raw force가 low-rank라는 이점이 있다. 그러나 thermodynamic force에는 P_ss의 score가 들어가므로 learned r=1 score가 sigma_local과 반드시 같다고 가정하지 않는다.

특히 NN의 r>1 branch가 nonzero라고 해서 passive K_r가 entropy를 생산했다고 해석하면 안 된다. 그것은 score correction, approximation, finite-lag effect, branch attribution의 문제일 수 있다. 물리적 passive shell work의 정답은 여전히 0이다.

## 8. 통과/실패 기준

통과: equilibrium reference와 passive shell work가 0에 수렴하고, 두 EPR estimator가 시간간격 감소에 따라 일치한다. active run의 유의한 EPR을 확보한 뒤만 morphology/NN 비교를 수행한다.

실패 처리: lambda0=0에서 지속되는 nonzero EPR은 새로운 물리 결과가 아니라 먼저 FDT, convolution symmetry, 시간간격, stationary sampling을 점검할 신호이다. passive shell의 양의 값만 선택하여 총 EPR에 합하지 않는다.

## 9. 제출 결과 표

    run_id, g1, ell, lambda0, D, Nx, dt, delta_t,
    sigma_total, sigma_local, passive_work_r0...rmax,
    NN_total, NN_raw_branch_r0...rR, tail_metrics,
    curl_error, block_SE, stationarity_status

논문에서 가능한 주장: 'reciprocal nonlocal organization이 local active irreversibility를 어떻게 바꾸는가'. 불가능한 주장: '이 대칭 kernel 자체가 새로운 EPR source이다'.

## 10. 참고문헌

[R1] Nardini et al., Physical Review X 7, 021007 (2017), DOI 10.1103/PhysRevX.7.021007.
[R3] Thewes, Qiang, Paulin and Zwicker, Physical Review Research 8, L022023 (2026), DOI 10.1103/gdwt-v9yj.
상세한 문헌 역할은 REFERENCES.md 참조. 위의 이산 모델·parameter·검증 규칙은 본 명세의 설계이다.