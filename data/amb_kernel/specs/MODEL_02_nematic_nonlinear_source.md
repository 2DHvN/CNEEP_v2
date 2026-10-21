# M02 — Even anisotropic kernel + nonlinear source

Version 1.0 | 역할: nematic anisotropy와 nonreciprocity의 분리
우선순위: M01 이후. 외부 polar direction 없이 각도 효과를 시험하는 후보.
공통 규약: COMMON_PROTOCOL.md. 아래 값은 pilot 설계이다.

## 1. 연구 질문과 물리적 의미

방향에 따라 신호 diffusivity가 다르지만 앞뒤는 구별하지 않는 매질을 생각한다. 고정된 substrate director가 있으므로 scalar phi만 유지할 수 있다. 이 director를 자체적으로 움직이는 polarization field로 해석하지 않는다.

    c_t = D_parallel*partial_parallel^2 c
          + D_perp*partial_perp^2 c - k*c + S(rho)
    S(rho) approximately s0+s1*phi+s2*phi^2

fast-field limit는 anisotropic even Green kernel을 준다. anisotropy 자체는 reciprocity를 깨지 않는다. 비가역성은 M01과 같은 nonlinear source–response mismatch에서 나온다. [R2]의 scalar taxis/QS coarse-graining을 anisotropic transport로 확장한 제안이며, 이 특정 모델의 실험적 유도가 완료된 것은 아니다.

## 2. 방정식과 reciprocal control

    dphi = -L mu dt + sqrt(2D) B dW,     B B^T=L
    mu = a phi+b phi^3+kappa L phi
         +g1 K phi
         +g2 [K(phi^2)+2*eta*phi*(K^T phi)]

K=K^T이지만 rotationally invariant일 필요는 없다. eta=0이 primary, eta=1은 reciprocal reference. b>0, kappa>0, D>0.

    F_E = F0 + g1*phi^T K phi/2
          + eta*g2*sum_i phi_i*[K(phi^2)]_i
    mu_A = (1-eta)*g2*K(phi^2)

F0는 local phi^4 및 kappa gradient energy이다. local AMB lambda0는 primary run에서 0으로 둔다.

## 3. 정확한 kernel 구현

기준 discretization에서는 director를 x 또는 y 축에 정렬한다. arbitrary-angle mixed-derivative stencil을 임의로 추가하지 않는다.

    Khat(q) = 1/[1+ell_x^2 Lxhat(q)+ell_y^2 Lyhat(q)]
    Lxhat=4 sin(q_x/2)^2,   Lyhat=4 sin(q_y/2)^2
    ell_x=ell_geo*sqrt(chi)
    ell_y=ell_geo/sqrt(chi)
    chi=ell_x/ell_y > 0

sum K=1, K_s=K_(-s). chi=1이면 M01의 isotropic lattice kernel로 돌아간다. ell_geo^2=ell_x*ell_y를 고정하는 것은 interaction footprint의 한 척도를 고정하는 convention이며, 모든 spectral moment를 동시에 고정하는 것은 아니다. coupling normalization과 실제 directional range도 기록한다.

x/y director reversal과 front/back reversal은 다르다. 이 모델은 K(s)=K(-s)인 even anisotropy를 유지한다. 'cos(2 theta)' 같은 even harmonic이 생겨도 linear term은 equilibrium contribution이다.

90도 회전은 Nx=Ny square lattice에서 x/y swap으로 정확히 구현된다. 45도 또는 임의각 회전을 추가하려면 anisotropic diffusion operator의 positive discretization과 lattice anisotropy를 별도로 명세해야 한다. 첫 실험군에는 넣지 않는다.

## 4. Functional curl과 local limit

    C_mu,ij = 2*g2*(1-eta)*K_ij*(phi_j-phi_i)

이다. chi 변화는 이 kernel의 방향 분포를 바꾸지만 eta=1에서 curl은 항상 0이다. phi가 균일한 background이면 nonlinear source의 linearized dynamics는 reciprocal하다.

range보다 매끄러운 field에서

    K*f = f + M_ab*partial_a partial_b f + ...

이고 nonvariational local contribution은

    mu_A,local = (1-eta)*g2*M_ab*partial_a phi*partial_b phi

이다. 축 정렬 Green kernel의 continuum limit에서는 M_xx=ell_x^2, M_yy=ell_y^2, M_xy=0이다. 이는 anisotropic AMB coefficient tensor에 대응한다. 동시에 생기는 conservative gradient correction을 포함해야 원 nonlocal model과 비교할 수 있다.

angular anisotropy를 free-energy의 quadratic term에만 넣는다면 이 active tensor가 생기지 않는다. source nonlinearity가 핵심이다.

## 5. Kernel EPR 및 angular 진단

    nu_r = (1-eta)*g2*K_r(phi^2)
    sigma_r = -(1/D)*E[nu_r^T o dotphi]
    sigma_total = sum_(all r) sigma_r

midpoint 및 drift-minus-trace estimator는 M01과 같다.

    J_r=2*(1-eta)*g2*K_r*diag(phi)
    sigma_r=E[nu_r^T L mu]/D-E[Tr(L J_r)]

공간 shell score 하나가 angular harmonic 분해까지 뜻하지는 않는다. NN architecture는 유지하고 simulator-only angular diagnostic을 추가할 수 있다. 예를 들어 offset를 서로 겹치지 않는 horizontal sector와 vertical sector로 나눠 K_r=K_r,H+K_r,V로 분할한다. 대각선 offset는 양쪽에 1/2씩 배분하는 등 tie rule을 고정하여 합을 정확히 유지한다.

    horizontal: |s_x|>|s_y|
    vertical:   |s_y|>|s_x|
    diagonal:   half to each

그 결과 sigma_r,H+sigma_r,V=sigma_r여야 한다. sector는 inversion-symmetric하게 나눈다. 각도 bin을 먼저 평균하거나 |K|로 바꾸지 않는다. 이 angular 분해는 kernel에 대해 정의한 진단이며, raw NN branch가 자동으로 출력하는 값은 아니다.

## 6. Controls와 예측

| 조작 | 검사할 결과 |
|---|---|
| chi=1 | 동일 parameter의 M01 |
| g2=0, chi 임의 | even anisotropic linear interaction만으로는 EPR=0 |
| eta=1, chi 임의 | reciprocal nonlinear completion, EPR=0 |
| chi -> 1/chi 및 x/y 전체 swap | stationary total EPR 동일; H/V sector 역할 교환 |
| ell_x=ell_y=0 | local source, EPR=0 |

축 swap 대칭은 fixed-mass ensemble, 초기조건 분포, boundary condition도 함께 swap했을 때의 통계적 예측이다. 같은 random seed의 단일 realization이 동일한 EPR을 가져야 한다는 요구가 아니다.

chi에 따른 total EPR 증가/감소는 미리 정하지 않는다. kernel geometry와 stationary correlations가 동시에 변한다.

## 7. Pilot configuration과 최소 sweep

```yaml
model_id: M02
Nx: 32
Ny: 32
h: 1.0
phi_bar: 0.0
a: 1.0
b: 1.0
kappa: 2.0
g1: 0.25
g2: 0.10
eta: 0.0
D: 0.30
dt: 0.001
kernel:
  family: green_even_anisotropic
  ell_geo: 1.2
  chi: 2.0
  director_axis: x
```

    chi: [1, 2, 4, 0.5]
    ell_geo: [1.2, 2.0]
    g2: [0, 0.10, 0.20]
    eta: [0, 1]
    D: [0.10, 0.30]

첫 비교는 M01의 ell=1.2와 M02의 chi=1이 수치적으로 같은 모델인지 확인하는 것이다. 다음 chi=2와 chi=0.5를 비교한다. chi=4에서 좁은 방향의 kernel이 grid scale보다 작아지면 continuum anisotropic interpretation과 별개로 lattice model임을 명시한다.

긴 축이 NN R를 넘어가는 tail이 커질 수 있으므로 shell-tail과 H/V-sector tail을 따로 기록한다. anisotropy로 EPR이 변한 것과 receptive-field 밖으로 정보가 이동한 것을 구분한다.

## 8. Shell-force 적합성과 위험

shell 내부 방향별 weight가 독립이므로 angular dependence 자체는 현재 구조와 양립한다. 그러나 K_r(phi^2)의 nonlinear full-rank 문제는 M01과 동일하다. chi를 키웠다고 자동으로 rank bottleneck이 없어지는 것은 아니다. 실질적으로 지배적인 몇 방향이 생기더라도 이는 trajectory/weight distribution의 근사적 특성이다.

다음 네 결과를 구분한다: 물리 sigma_r; angular sector work; NN raw radial branch; 전체 learned force로 projection한 physical sigma_r. 첫 convolution weight의 모양만 보고 'nematic kernel을 복원했다'고 결론내리지 않는다. functional curl의 방향 구조까지 확인한다.

## 9. 통과 기준과 보고

통과: M01 limit, reciprocal null, axis-swap covariance, shell/sector 합산 항등식, EPR estimator convergence. NN 실패 시 tail, rank/score 문제, lattice anisotropy 순으로 분리 진단한다.

추가 저장 column:

    chi, ell_x, ell_y, director_axis,
    sigma_r_H, sigma_r_V, sector_tail,
    structure_factor_x, structure_factor_y,
    projected_curl_directional_error

가능한 물리 주장: 'even anisotropy가 nonlinear scalar irreversibility의 공간적 분포를 조절한다'. 잘못된 주장: 'even angular kernel 자체가 linear nonreciprocity를 만든다'.

## 10. 참고문헌

[R1] Nardini et al., PRX 7, 021007 (2017), DOI 10.1103/PhysRevX.7.021007.
[R2] Dinelli, O'Byrne and Tailleur, J. Phys. A 57, 395002 (2024), DOI 10.1088/1751-8121/ad72bc.
[R4] Dinelli et al., Nature Communications (2023), DOI 10.1038/s41467-023-42713-5.
본 모델의 anisotropic source extension, tensor limit, parameter sweep은 여기서 제안한 설계이다.