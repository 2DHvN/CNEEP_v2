# M04 — Mixed directional and density-dependent nonreciprocity

Version 1.0 | 역할: odd angular kernel과 nonlinear source의 결합 및 간섭
우선순위: M01과 M03을 각각 검증한 뒤. 처음부터 이 모델만 학습하지 않는다.
공통 규약: COMMON_PROTOCOL.md. parameter와 예상 비교는 실험 설계이며 검증된 phase diagram이 아니다.

## 1. 물리적 모델과 연구 질문

density-regulated source가 흐르는 매질을 통해 전달되고, receiver가 그 신호에 선형으로 반응한다.

    c_t + u dot grad c = D_c Delta c - k c + S(rho)
    S(rho_ref+phi_scale*phi) = s0+s1*phi+s2*phi^2+...

fast-field elimination 후 spatially constant source를 제외하면 g1*K phi+g2*K(phi^2)가 된다. source regulation은 M01, differential transport는 M03과 동일한 가정이다. scalar coarse-graining의 일반적 근거는 [R2], differential-flow 물리 설정은 [R5]이다. 이 둘을 결합한 polynomial scalar dynamics와 아래 EPR 분해는 본 명세의 제안이다.

질문: spatial odd coupling과 density-dependent mismatch는 EPR에 어떻게 함께 기여하는가? 각 mechanism의 거리별 기여가 강화·상쇄되는가? 전체 EPR만으로 mechanism을 구별할 수 있는가?

## 2. 정확한 primary dynamics

    dphi=-L mu dt+sqrt(2D) B dW,     B B^T=L
    mu=a phi+b phi^3+kappa L phi+g1 K phi+g2 K(phi^2)

b>0, kappa>0, D>0, lambda0=0. eta=0인 source response가 primary이다. source와 receiver를 바꾼 (K phi)^2는 M05이며 여기와 다르다.

K는 M03의 positivity-preserving upwind Green kernel로 고정한다.

    Khat=1/{1+ell_x^2 Lxhat+ell_y^2 Lyhat
             +|vx|*(1-cos q_x)+|vy|*(1-cos q_y)
             +i*(vx*sin q_x+vy*sin q_y)}

K=real(ifft2(Khat)), sum K=1. K_e=(K+K^T)/2, K_o=(K-K^T)/2. K sign, FFT origin, offset transpose는 COMMON_PROTOCOL.md의 convention이다.

    mu_E=a phi+b phi^3+kappa L phi+g1 K_e phi
    mu_A=nu_dir+nu_src
    nu_dir=g1 K_o phi
    nu_src=g2 K(phi^2)

nonlinear source의 even/odd 부분을 더 나눌 수 있지만, 첫 결과표에서는 directional-linear와 source-nonlinear 두 mechanism만 기본 분해로 사용한다.

## 3. 비가역성의 해석적 분리

    C_mu,ij
      =2*g2*K_e,ij*(phi_j-phi_i)
       +2*K_o,ij*[g1+g2*(phi_i+phi_j)]

첫 줄은 even kernel에서도 남는 source mismatch, 둘째는 spatial front/back asymmetry이다. projected curl Pi*C*Pi를 analytic/finite-difference/learned-force 수준에서 비교한다.

균일한 phi_bar 주변에서

    c_eff=g1+2*g2*phi_bar
    delta_mu=[a+3b phi_bar^2+kappa L]delta_phi
             +c_eff*K delta_phi+O(delta_phi^2)

이다. 따라서 linearized H/A는 각각 local stiffness+c_eff*K_e, c_eff*K_o이다.

중요한 cancellation control:

    phi_bar=-g1/(2*g2),    g2!=0

에서는 nonlocal linear term 전체가 상쇄된다. 그러나 K(delta_phi^2)는 남으므로 full nonlinear EPR이 0이어야 하는 것은 아니다. 이는 Gaussian 수준의 irreversibility와 nonlinear irreversibility를 분리하는 시험이다.

mean을 바꾸면 local stiffness a+3b phi_bar^2도 바뀐다. cancellation point의 EPR 차이를 오직 c_eff 상쇄의 효과라고 주장하려면 별도 stiffness-matched comparison을 수행하거나 그 변화를 명시한다. phi가 signed field라는 점과 실제 density의 admissible range도 구분한다.

## 4. 정확한 mechanism × kernel EPR

    nu_dir,r=g1 K_o,r phi
    nu_src,r=g2 K_r(phi^2)
    sigma_dir,r=-(1/D)E[nu_dir,r^T o dotphi]
    sigma_src,r=-(1/D)E[nu_src,r^T o dotphi]
    sigma_total=sum_r (sigma_dir,r+sigma_src,r)

두 mechanism은 같은 trajectory ensemble에서 평가한다. 별도 g2=0 run의 EPR을 full run에서 빼는 것은 위의 sigma_src와 일반적으로 같지 않다. interaction을 끄면 P_ss 자체가 바뀌기 때문이다.

source 항을 더 세분화할 때만

    nu_src,e,r=g2 K_e,r(phi^2)
    nu_src,o,r=g2 K_o,r(phi^2)

를 추가하고 정확한 합산을 확인한다. signed contribution의 음수는 허용한다. mechanism별 양의 독립 dissipation reservoir로 해석하지 않는다.

equal-time check:

    J_dir,r=g1 K_o,r
    J_src,r=2*g2 K_r diag(phi)
    sigma_c,r=E[nu_c,r^T L mu]/D-E[Tr(L J_c,r)]
    Tr(L J_dir,r)=0
    Tr(L J_src,r)=2*g2*(L K_r)_00*N*phi_bar

특히 mean-shift/cancellation 실험에서 source trace를 빠뜨리지 않는다.

## 5. Controls: 한 mechanism을 끄는 것과 reciprocity 복구

| 조작 | 정확한 해석 |
|---|---|
| g2=0 | M03 linear-kernel model |
| v=0 | M01의 eta=0 even-source model |
| g1=0, v!=0, g2!=0 | odd nonlinear source는 남음; equilibrium control 아님 |
| v->-v 및 spatial reflection | stationary total EPR의 반사 대칭성 검사 |
| c_eff=0 | linear nonlocal irreversibility 제거; nonlinear residual은 남음 |

모든 nonlinear interaction의 reciprocity를 복구하는 reference는 다음 별도 control이다.

    mu_ref=mu_loc+g1 K_e phi
           +g2 [K(phi^2)+2*phi*(K^T phi)]

이는

    F_ref=F0+g1*phi^T K_e phi/2
          +g2*sum_i phi_i*[K(phi^2)]_i

의 functional derivative이므로 EPR=0이다. K가 asymmetric이어도 cubic functional의 두 변분 항을 모두 포함하면 reciprocal하다. primary dynamics와 reference의 모든 통계가 같아야 하는 것은 아니다.

부분 completion으로 eta를 사용한다면

    mu_eta=mu_loc+g1 K phi
           +g2[K(phi^2)+2*eta*phi*(K^T phi)]
    mu_A,eta=g1 K_o phi+(1-eta)*g2 K(phi^2)

이다. eta=1이어도 g1*K_o phi가 남아 전체 EPR은 일반적으로 0이 아니다. full equilibrium reference와 혼동하지 않는다.

## 6. Pilot configuration과 최소 실험 행렬

```yaml
model_id: M04
Nx: 32
Ny: 32
h: 1.0
phi_bar: 0.0
a: 1.0
b: 1.0
kappa: 2.0
g1: 0.25
g2: 0.10
D: 0.30
dt: 0.001
kernel:
  family: green_polar_upwind
  ell_x: 1.2
  ell_y: 1.2
  vx: 1.2
  vy: 0.0
```

최소 비교군은 다음 다섯 가지다.

    A: g2=0                         # linear directional only
    B: vx=vy=0                      # nonlinear source only
    C: primary configuration        # both mechanisms
    D: full mu_ref                  # reciprocal control
    E: vx=-1.2, vy=0                # reflected transport

A와 M03을 비교할 때 b를 동일하게 맞춘다. b=0 Gaussian M03과 b=1 mixed model을 동일한 baseline이라고 부르지 않는다. B는 동일한 even kernel parameter의 M01과 비교한다.

이후 one-at-a-time sweep:

    g2: [0,0.05,0.10,0.20,-0.10]
    vx: [0,0.4,1.2,-1.2]
    phi_bar: [0,0.3,-0.3]
    ell_x=ell_y: [1.2,2.0]

cancellation 전용 예시는 g1=0.10, g2=0.20, phi_bar=-0.25이다. 이 값은 대수적 cancellation을 위한 선택이지 큰 EPR이 검증된 조건은 아니다.

## 7. Shell-force 적합성과 해석 제한

한 branch에 linear directional feature와 nonlinear source가 함께 들어간다. linear part는 쉽게 접근 가능한 function class지만 source K_r(phi^2)의 rank 문제와 전체 stationary-score correction은 남는다.

NN가 전체 EPR을 맞춘 뒤에도 mechanism separation은 model-based observable 또는 projected curl로 확인해야 한다. 기존 network는 shell당 하나의 출력만 주므로 source/linear 두 mechanism을 raw branch 하나만으로 고유하게 분리할 수 없다. architecture를 바꾸지 않고 known nu_dir,r, nu_src,r를 전체 learned force에 projection하여 비교한다.

    sigma_c,r_est = -E[nu_c,r^T L d_theta,total]

이는 known-model diagnostic이다. network가 두 independent latent mechanisms를 스스로 식별했다고 주장하지 않는다.

r>R tail은 source와 directional mechanism별로 계산한다. 두 tail의 합이 cancellation으로 작아졌다고 각 force가 모두 잘 관측되었다고 판단하지 않는다.

## 8. 성공 조건과 저장 항목

통과: M01/M03 reduction, full reciprocal reference, projected curl의 두 항, mechanism/shell 합산, drift-minus-trace check, stationary positive total EPR, dt/lag convergence.

추가 column:

    c_eff, sigma_dir_r, sigma_src_r, sigma_src_even_r, sigma_src_odd_r,
    sigma_tail_dir, sigma_tail_src,
    NN_total, NN_raw_shell, NN_projection_dir, NN_projection_src,
    curl_even_density_error, curl_odd_density_error

검증 전 기대하지 말 것: 두 mechanism의 EPR이 단순히 따로 측정한 값의 합과 같음; g2/v 증가에 따른 단조성; cancellation point에서 full EPR=0; shell별 기여가 모두 양수.

## 9. 참고문헌

[R1] Nardini et al., PRX 7, 021007 (2017), DOI 10.1103/PhysRevX.7.021007.
[R2] Dinelli, O'Byrne and Tailleur, J. Phys. A 57, 395002 (2024), DOI 10.1088/1751-8121/ad72bc.
[R4] Dinelli et al., Nature Communications (2023), DOI 10.1038/s41467-023-42713-5.
[R5] Muñoz-García and Neufeld, PRE 80, 061902 (2009), DOI 10.1103/PhysRevE.80.061902.
모델 결합과 cancellation/attribution protocol은 본 명세에서 유도한 설계이다.