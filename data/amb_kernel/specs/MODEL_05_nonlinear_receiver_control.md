# M05 — Nonlinear receiver / conventional quorum-sensing control

Version 1.0 | 역할: 물리적으로 자연스러운 대안 및 shell-additivity의 한계 실험
우선순위: 선택적. 정확한 raw shell attribution의 주 성공 모델로 추천하지 않는다.
공통 규약: COMMON_PROTOCOL.md. 모델 자체의 타당성과 NN 표현 실패를 구분하기 위한 명세이다.

## 1. 왜 이 모델을 포함하는가

입자가 여러 위치에서 온 신호를 먼저 합한 뒤 그 총신호에 반응하는 경우,

    u=K phi
    mu_QS=h(u)

가 자연스럽다. source에서 제곱한 뒤 전달하는 K(phi^2)와는 다른 물리다. density-dependent motility의 log v가 sensed density에 의존하는 scalar hydrodynamics는 [R2]의 일반적 coarse-graining과 연결된다.

이 문서는 h의 quadratic Taylor truncation을 선택한다. 실제 rho의 log rho, density-dependent mobility 등을 모두 그대로 유지한 exact microscopic model은 아니다. constant-mobility AMB-type truncation이라는 점을 밝힌다. phi는 signed order parameter이므로 h(phi)와 실제 v(rho)를 연결하려면 rho_ref와 expansion interval을 기록한다.

이 모델을 제외하면 architecture에 유리한 source mechanism만 고른다는 비판을 받을 수 있다. 반대로 이 모델에서 NN가 실패했다고 모든 nonlocal scalar theory를 부정해서도 안 된다.

## 2. 정확한 dynamics

    dphi=-L mu dt+sqrt(2D) B dW,    B B^T=L
    u=K phi
    mu=mu_loc+g1*u+g2*u^2
    mu_loc=a phi+b phi^3+kappa L phi

K=K^T는 normalized even Green kernel이다.

    Khat(q)=1/[1+ell^2*Lhat(q)]

primary b>0, kappa>0, D>0, lambda0=0. pointwise square는 convolution 뒤에 계산한다. 아래 두 연산이 같다는 가정은 금지한다.

    receiver: (K phi)^2
    source:    K(phi^2)

## 3. Reciprocal completion reference

primary의 h(u)=g2*u^2에 대응하는 cubic functional을

    F3_recv=g2*sum_i phi_i*u_i^2

로 두면

    delta F3_recv/delta phi
      =g2*[u^2+2*K^T(phi*u)]

이다. 따라서 optional control family는

    mu_eta=mu_loc+g1*u+g2*[u^2+2*eta*K^T(phi*u)]
    mu_E=delta[F0+g1*phi^T K phi/2+eta*F3_recv]/delta phi
    mu_A=(1-eta)*g2*u^2

이다. primary eta=0, reciprocal eta=1. 이 completion은 M01의 2*phi*(K^T phi)와 다르다. source/receiver 모델의 derivative를 혼동하지 않는다.

## 4. 비가역성 및 local AMB limit

primary의 Jacobian/curl은

    J_receiver = 2*g2*diag(u)*K
    C_mu,ij=2*g2*K_ij*(u_i-u_j)

이다. eta family에서는 curl에 (1-eta)가 곱해진다. notation에서 이 항을 source라고 부르지 않기 위해 구현 함수명은 mu_receiver를 사용한다.

K*f=f+m2*Delta f+...의 장파장 근사에서

    g2*(K phi)^2 = g2*phi^2 + 2*g2*m2*phi*Delta phi+...
      =delta/delta phi [g2*integral(phi^3)/3
                       -g2*m2*integral(phi*|grad phi|^2)]
       -g2*m2*|grad phi|^2+...

이다. 동일한 g2와 even kernel에서 M01의 leading active coefficient와 부호가 반대다. 그렇다고 두 full models가 g2 부호 반전만으로 같아지는 것은 아니다. conservative sector와 higher-gradient terms가 다르다.

균일한 phi_bar에서 linearized stiffness는 eta=0일 때

    H(q)=a+3b phi_bar^2+kappa Lhat+(g1+2g2*phi_bar)*Khat

이다. M01 primary와 같은 linearization을 가질 수 있으므로 구조인자나 선형 안정성만으로 두 mechanism을 식별할 수 없다. nonlinear time-asymmetry나 curl의 field dependence를 비교한다.

## 5. EPR 총량과 거리 attribution convention

    sigma_total=-(1-eta)*g2/D*E[(u^2)^T o dotphi]

이고 이는 명확한 scalar EPR이다. 그러나 h(sum_r u_r)는 nonlinear이므로 독립 거리 channel의 합이 자연적으로 주어지지 않는다. 비교를 위해 다음 attribution convention을 명시적으로 고정한다.

    u_r=K_r phi
    u=sum_(all r) u_r
    nu_r=(1-eta)*g2*u*u_r
    sum_r nu_r=(1-eta)*g2*u^2
    sigma_r=-(1/D)*E[nu_r^T o dotphi]

이 convention은 cross-shell term의 contribution을 두 관련 shell에 대칭적으로 나눈다. 다른 attribution rule도 가능하므로 이 sigma_r를 고유한 independent interaction-channel EPR이라고 부르지 않는다. primary source model M01의 'source distance'와 용어를 구분한다.

중요: r=0의 nu_0=g2*u*K(0)*phi는 다른 shell 정보 u도 포함한다. 따라서 물리적 attribution sigma_0는 일반적으로 0일 필요가 없다. 반면 현재 NN의 순수 on-site branch는 stationary continuous-time mean이 0이다. 이것은 이 모델에서 raw branch와 위 attribution이 일치하지 않을 수 있는 구체적인 이유다.

independent equal-time check:

    c=(1-eta)*g2
    J_r=c*[diag(u_r)*K+diag(u)*K_r]
    sigma_r=E[nu_r^T L mu]/D-E[Tr(L J_r)]

이 trace를 M01의 source 공식으로 대체하지 않는다. total J_A=2c*diag(u)*K와 shell Jacobian 합도 비교한다.

## 6. Shell-force의 구조적 mismatch

각 branch는 자신의 shell만 읽지만 nu_r=c*u*u_r는 모든 shell의 u에 의존한다. 더 직접적으로 r!=s이면

    partial^2 h(sum_t u_t)/(partial u_r partial u_s)=2*g2

이고 이는 일반적인 shell-additive raw force의 mixed derivative 0과 다르다. K 범위가 여러 shell을 포함하는 한 원래 physical force를 같은 function class로 정확히 표현한다고 보장할 수 없다.

다만 field-space gradient gauge, 실제 trajectory support, finite approximation 때문에 total EPR 추정이 유용할 가능성까지 배제하는 정리는 아니다. 정확한 kernel attribution을 사전에 보장할 수 없다는 뜻이다.

금지: u=K phi를 입력 채널로 추가; branch끼리 곱하기; h(sum shell feature)를 위한 새 head 추가. 이 모델은 현재 architecture를 고정한 상태의 stress test이다.

## 7. Controls와 pilot

```yaml
model_id: M05
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

controls:

| 조건 | 예측 |
|---|---|
| g2=0 | even linear interaction만 남고 EPR=0 |
| eta=1 | full functional derivative, EPR=0 |
| K=I | local chemical potential, EPR=0 |
| K_ij=1/N | u=phi_bar constant, primary nonlinear drift=0 |
| 같은 g1,g2,phi_bar의 M01 | 동일 linearization 가능, nonlinear EPR은 다를 수 있음 |

우선 g2=0/eta=1 null을 통과한 뒤 M01과 동일한 Nx,D,ell에서 비교한다.

    ell: [0,1.2,2.0,4.0]
    g2: [0,0.10,0.20,-0.10]
    phi_bar: [0,0.3]
    eta: [0,1]

receiver cross-shell mismatch가 강해지는 parameter를 '물리 모델이 잘못되었다'고 자동 분류하지 않는다. 반대로 NN error가 증가했다고 그 원인이 반드시 cross-shell 구조라고 단정하지 말고 tail·optimization·lag를 먼저 분리한다.

## 8. 결과 판정과 필수 보고

먼저 simulator-only의 stationary EPR, shell attribution 합, reciprocal null, curl을 검증한다. 그 다음 다음 세 결과를 분리한다.

    total_NN_accuracy
    known_model_projection_accuracy
    raw_branch_attribution_accuracy

NN total이 맞더라도 raw branch가 본 attribution에 맞지 않을 수 있다. 특히 sigma_0와 learned local branch의 차이는 오류가 아니라 attribution/function-class mismatch일 수 있다.

추가 저장:

    receiver_u_moments, mixed_shell_covariances,
    sigma_total, sigma_attribution_r, attribution_rule='u_times_ur',
    raw_NN_r, force_projection_r, receiver_curl_error,
    model_M01_matched_linearization_run_id

권장 결론: '현재 shell-additive architecture의 물리적 적용 범위가 source-additive와 receiver-nonlinear mechanism에서 어떻게 다른가'. 추천하지 않는 결론: '한 번 정의한 attribution이 유일한 microscopic kernel EPR이다'.

## 9. 참고문헌

[R1] Nardini et al., PRX 7, 021007 (2017), DOI 10.1103/PhysRevX.7.021007.
[R2] Dinelli, O'Byrne and Tailleur, J. Phys. A 57, 395002 (2024), DOI 10.1088/1751-8121/ad72bc.
[R4] Dinelli et al., Nature Communications (2023), DOI 10.1038/s41467-023-42713-5.
quadratic receiver truncation, reciprocal control, u*u_r attribution, architecture stress test는 본 명세의 제안이다.