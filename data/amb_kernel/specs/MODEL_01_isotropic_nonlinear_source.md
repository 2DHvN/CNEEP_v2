# M01 — Isotropic nonlinear source / density-dependent nonreciprocity

Version 1.0 | 역할: 등방성을 유지하는 핵심 물리 후보
우선순위: M00의 수치 검증 다음. 현재 architecture에서의 nonlinear 표현오차를 함께 평가한다.
공통 규약: COMMON_PROTOCOL.md. pilot parameter는 미검증 출발값이다.

## 1. 연구 질문

등방적인 전달 kernel을 유지하면서 source의 밀도 의존성과 receiver response를 독립적으로 정하면, scalar phi만으로 kernel-resolved EPR이 생기는가? 이를 K*(phi^2)에서 검증한다.

이 모델은 'phi_x K_xy phi_y^2를 자유에너지에 넣었다'는 정의가 아니다. 그 functional을 완전히 변분하면 reciprocal completion이 추가된다. 그 completion의 계수를 따로 제어하는 dynamics가 여기의 모델이다.

## 2. 물리적 출발점과 가정

설계한 microscopic response rule의 한 예:

    rho = rho_ref + phi_scale*phi
    S(rho) = rho*q_particle(rho_short)
    q_particle = q0 + q1*rho_short
    rho_short approximately rho
    c_t = D_c Delta c - k c + S(rho)
    receiver response is linear in c

빠른 mediator를 제거하면 c=(k-D_c Delta)^(-1) S(rho)가 된다. source를 rho_ref 주변에서 전개하면 constant + s1*phi+s2*phi^2가 생긴다. 공간 상수는 conserved drift에서 사라지며, 나머지는 g1*K phi + g2*K(phi^2)가 된다.

문헌 [R2]는 taxis/QS의 scalar hydrodynamics와 fast-signal elimination을 뒷받침한다. 정확한 quadratic source, constant mobility, Landau polynomial은 본 실험의 모델링 선택이다. phi^2를 그대로 실제 농도 또는 양의 방출량이라고 부르지 않는다. 순수 S(rho)=q*rho^2를 주장하려면 g1,g2,rho_ref의 관계도 지켜야 하며, 두 계수를 독립 sweep한 모든 점을 그 동일한 microscopic law로 설명할 수는 없다.

mediator가 passive energy를 최소화할 뿐이라면 receiver에 S'(phi)가 생겨 reciprocity가 복구된다. 이 모델은 독립적인 active source/response 규칙을 가정한다. mediator 자체를 simulate하거나 NN에 입력하지 않는다. 빠른 relaxation·작은 독립 mediator noise·빠른 orientation relaxation의 근거가 없는 실제 계에는 이 scalar Markov approximation을 그대로 적용하지 않는다.

## 3. 정확한 scalar dynamics

    dphi = -L mu dt + sqrt(2D) B dW,    B B^T=L
    mu = a phi + b phi^3 + kappa L phi
         + g1 K phi
         + g2 [K(phi^2) + 2*eta*phi*(K^T phi)]

곱 phi*(K^T phi)는 pointwise 곱이다. 기본 K=K^T이다. b>0, kappa>0, D>0. 기존 local AMB 항 lambda0*|grad phi|^2는 이 모델의 첫 단계에서는 넣지 않는다. 새로운 interaction 자체의 EPR을 분리하기 위해서다.

핵심 run: eta=0. reciprocal reference: eta=1.

    F0 = sum_i [a phi_i^2/2 + b phi_i^4/4] + kappa*phi^T L phi/2
    F3 = g2 * sum_i phi_i * [K(phi^2)]_i
    delta F3/delta phi = g2 [K(phi^2)+2 phi*(K^T phi)]
    mu_E = delta[F0+g1*phi^T K phi/2+eta*F3]/delta phi
    mu_A = (1-eta)*g2*K(phi^2)

eta는 reciprocity를 수학적으로복구하는 control parameter이지, 모든 실험계에서 독립적으로 조절 가능한 물성이라고 가정하지 않는다. eta 변화에 따른 morphology 변화도 함께 기록한다.

## 4. Kernel

    Khat(q) = 1/[1+ell^2*Lhat(q)]
    Lhat=4 sin(q_x/2)^2+4 sin(q_y/2)^2
    K=real(ifft2(Khat)),  sum K=1,  K_s=K_(-s)

ell=sqrt(D_c/k)/h의 차원 없는 screening length이다. h=1 고정. kernel normalization을 고정하여 ell 변화와 integrated coupling 변화가 섞이지 않게 한다. ell->0은 delta, ell이 격자 크기보다 충분히 커지는 normalized periodic limit는 uniform kernel이다. 둘 다 중요한 null control이다.

kernel을 shell에 맞춰 자르지 않는다. K_r와 r>R tail을 별도 저장한다.

## 5. Detailed-balance 검증과 AMB 연결

J_ij=partial mu_i/partial phi_j라고 하면 nonlocal functional curl은

    C_ij = 2*g2*(1-eta)*K_ij*(phi_j-phi_i)

이다. eta=1에서 0이고, eta=0의 일반적인 비균일 field에서는 nonzero이다. conserved system의 엄밀한 test는 Pi*C*Pi이다. random fixed-mass field에서 analytic Jacobian과 finite difference를 비교한다.

continuum kernel의 finite second moment가 존재하고 field가 range보다 천천히 변할 때,

    K*f = f + m2*Delta f + ...
    g2*K(phi^2)
      = delta/delta phi [g2*integral(phi^3)/3
                         - g2*m2*integral(phi*|grad phi|^2)]
        + g2*m2*|grad phi|^2 + ...

이다. eta=0에서 local AMB coefficient는 g2*m2이다. eta!=0에서는 active residual의 계수에 (1-eta)가 붙는다. 동시에 생성되는 conservative gradient correction을 버리고 '정확히 local AMB와 동일하다'고 주장하지 않는다. finite second moment가 없거나 q*ell이 작지 않으면 이 expansion 대신 원 convolution을 사용한다.

균일한 phi_bar 주변, eta=0의 linear stiffness:

    H(q)=a+3b phi_bar^2+kappa Lhat+(g1+2g2 phi_bar)*Khat
    omega(q)=-Lhat*H(q)

일반 eta, even K에는 H(q)에 2*eta*g2*phi_bar*(1+Khat)를 추가한다. 균일한 배경에서는 linear operator가 reciprocal하므로 Gaussian approximation만으로는 이 모델의 비가역성을 보지 못한다. nonlinear fluctuations 또는 비균일 background가 필요하다.

## 6. 정확한 kernel EPR

    nu_r = (1-eta)*g2*K_r(phi^2)
    sigma_r = -(1/D)*E[nu_r^T o dotphi]
    sigma_total = sum_(all r) sigma_r

각 sigma_r는 signed이다. linear g1*K_r phi의 stationary work는 0이다. K_0(phi^2)=K(0)*phi^2도 local functional derivative이므로 r=0 active-work average는 0이다.

independent drift estimator:

    J_r = 2*(1-eta)*g2*K_r*diag(phi)
    sigma_r = E[nu_r^T L mu]/D - E[Tr(L J_r)]
    Tr(L J_r)=2*(1-eta)*g2*(L K_r)_00*N*phi_bar

mean zero일 때 trace는 0이지만, nonzero mean sweep에서는 생략하지 않는다. midpoint와 equal-time estimator가 dt->0에서 일치하는지 확인한다.

비상호성의 signed pair-correlation observable:

    sigma_r = -(1-eta)*g2/D
              *sum_(i,j in shell r) K_ij * E[phi_j^2 o dotphi_i]

HSR을 쓰지 않고 실제 nonlinear source kernel로 정의한 work이다. kernel coefficients가 ground truth라고 해서 NN branch weight나 score까지 동일하다고 가정하지 않는다.

## 7. 필수 controls

| 조건 | continuum-time stationary 예측 |
|---|---|
| g2=0 | equilibrium, EPR=0 |
| eta=1 | reciprocal completion, EPR=0 |
| K=I | 모든 항이 local chemical potential, EPR=0 |
| K_ij=1/N | K(phi^2)는 공간 상수; completion도 local linear term이 되어 EPR=0 |
| phi_bar=0, eta=0, g2!=0 | Gaussian linearization은 reciprocal; full nonlinear EPR은 일반적으로 nonzero |

eta=1에서 raw K(phi^2) work 하나만 평가하여 'EPR이 남았다'고 해석하지 않는다. EPR의 active residual은 (1-eta)g2*K(phi^2)이며, 또는 full chemical potential의 stationary work를 계산해야 한다.

## 8. Pilot configuration과 sweep

```yaml
model_id: M01
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
  family: green_even
  ell_x: 1.2
  ell_y: 1.2
```

실행 순서: g2=0 -> eta=1 -> 본 run eta=0. EPR 신호가 약하면 stationarity와 amplitude를 확인하며 |g2| 또는 D를 높인다.

    g2: [0, 0.05, 0.10, 0.20, -0.10]
    eta: [0, 0.5, 1]
    ell: [0, 1.2, 2.0, 4.0]
    phi_bar: [0, 0.3, -0.3]
    D: [0.10, 0.30, 0.60]
    Nx=Ny: [32,64]

먼저 one-at-a-time sweep; 가장 큰 값들을 동시에 조합하지 않는다. phase-separated extension은 a=-0.5 등의 별도 parameter군으로 구분하고, coarsening이 멈추었는지 확인한다. ell 증가에 따른 EPR의 단조 증가나 계면 패턴을 사전에 가정하지 않는다.

## 9. 현재 architecture에서의 성공 범위

eta=0의 raw force K_r(phi^2)는 해당 shell만 읽으므로 shell-additivity에는 맞는다. 그러나 shell 내 모든 점을 제곱한 뒤 더해야 한다. full input Hessian rank는 일반적으로 8r이고, 첫 linear projection width H가 작은 경우 정확한 raw force 표현은 제한된다. H/R을 바꾸지 말고 실제 trajectory distribution에서의 근사오차를 결과로 기록한다.

eta!=0의 reciprocal completion은 중심 phi와 외부 shell의 곱을 포함한다. 이것은 바깥 branch의 입력 구조에 맞지 않지만, equilibrium 최적 total force는 0이므로 그 사실만으로 reciprocal null test가 실패해야 하는 것은 아니다.

검증 순서:

1. simulator-only physical sigma_r와 full EPR을 확보한다.
2. NN total EPR과 전체 learned force의 model-channel projection을 비교한다.
3. raw branch와 physical sigma_r를 비교한다.
4. learned-force curl에서 위의 density-dependent K_ij*(phi_j-phi_i)가 복원되는지 검사한다.

## 10. Go / no-go 및 해석

물리 모델 통과: projected curl test, 세 equilibrium nulls, time-step convergence, stationary positive full EPR.
NN attribution 통과: total rate 일치만이 아니라 channel mean-work 및 curl residual까지 확인.

NN branch가 실패해도 model-defined kernel decomposition은 유효할 수 있다. 이를 물리 모델 실패와 구분한다. 반대로 branching 결과가 깔끔해도 scalar coarse-graining의 microscopic 조건을 검증한 것은 아니다.

가능한 주장: 'isotropic nonlinear source–response mismatch가 finite-range scalar irreversibility를 만들며, kernel-distance work로 분해된다.'
과도한 주장: 'quadratic source law가 특정 실제 시스템에서 이미 실증되었다', '각 shell은 독립적인 양의 dissipation reservoir이다'.

## 11. 참고문헌

[R1] Nardini et al., PRX 7, 021007 (2017), DOI 10.1103/PhysRevX.7.021007.
[R2] Dinelli, O'Byrne and Tailleur, J. Phys. A 57, 395002 (2024), DOI 10.1088/1751-8121/ad72bc; arXiv:2402.05072v2.
[R4] Dinelli et al., Nature Communications (2023), DOI 10.1038/s41467-023-42713-5.
R2/R4는 coarse-graining과 reciprocity 조건의 근거이며 위 단일-scalar quadratic source 모델 전체의 직접적인 선행 실험 결과는 아니다.