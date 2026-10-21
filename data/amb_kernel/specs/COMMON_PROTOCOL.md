# 공통 실험 프로토콜 — scalar nonlocal AMB / shell-force EPR

Version: 1.0 | 작성일: 2026-10-20 | 상태: 실험 설계 명세, NN/trajectory 실험 결과 아님

이 문서는 각 MODEL_*.md에 공통으로 적용되는 이산화, EPR 정의, 저장 규약, 판정 기준이다. 모델 문서의 방정식이 모델을 정의하며, 여기의 수치 조건은 그 방정식을 구현·검증하는 공통 규약이다. parameter 제안은 pilot 출발값이며, 안정한 패턴이나 유의한 EPR을 보장하는 검증값이 아니다.

## 1. 범위와 출처

관측하고 학습하는 변수는 보존된 실수 scalar phi 하나이다. phi는 signed order parameter이며 실제 밀도 rho와 동일하다고 가정하지 않는다. 물리적 해석이 필요하면 rho = rho_ref + phi_scale * phi를 별도로 명시한다. fuel, polarization, mediator field를 NN 입력이나 추가 동적 변수로 넣지 않는다.

EPR은 Nardini 등의 scalar trajectory 시간반전 비가역성 [R1]이다. 전체 물리계의 연료 소비나 열 방출량으로 해석하지 않는다. 모든 외부 parameter, 특히 polar kernel의 흐름 방향은 forward/reverse trajectory 비교에서 고정한다.

문헌의 일반적 coarse-graining과 여기서 제안하는 polynomial truncation·kernel 선택·수치 검증을 구별한다. REFERENCES.md에는 각 논문이 실제로 뒷받침하는 범위를 적었다. 이 패키지는 제안 모델의 신규성이나 완전한 microscopic derivation을 주장하지 않는다.

## 2. 고정된 NN 계약

기준 소스: 사용자 제공 shell_force.py.
SHA-256: baea2ddf87ed10ac5c4150617e8a24556f6d6490c54009b671d4254d1a79cf79
이 파일은 수정하지 않았다. 소스에서 확인한 사항:

- ShellForceKNEEP2D의 input은 [batch, 2, components, Ny, Nx]이다.
- 이 연구에서는 기존 생성자 옵션 n_components=1, ep_component_indices=(0,)를 사용한다. 기본값 2를 그대로 사용해 가짜 두 번째 변수를 만들지 않는다.
- hidden_channels, hidden_layers, max_distance, activation은 실험군 전체에서 동일하게 고정한다. 파일 기본값은 각각 8, 2, 4, 'elu'이지만, 사용 중인 실제 설정이 있으면 그 값을 보존하고 manifest에 기록한다.
- radius r>0은 Chebyshev shell S_r={s:max(|s_x|,|s_y|)=r}만 읽는다. 중심점과 내부 shell을 읽지 않는다. 이후 층은 1x1 convolution이다.
- radius 0은 중심 scalar만 읽는다. 모든 branch의 합은 shell-additive function class이다.
- 입력 pair로 midpoint와 increment를 만들고, branch output은 force * increment의 공간 평균이다.
- return_forces=True로 입력 좌표계의 force를 반환할 수 있다.

금지: phi^2, K*phi, gradient, mediator를 추가 채널로 넣기; shell mask 변경; cross-shell head 추가; 성능을 맞추기 위해 H/R을 모델별로 변경하기. simulator의 진단량 계산은 가능하지만 해당 진단량을 NN 입력으로 보내지 않는다.

학습 loss/optimizer/checkpoint는 제공된 파일에 없다. 이 명세는 기존 학습 절차를 임의로 대체하지 않는다. 실제 loss와 출력의 합산·정규화 방식을 manifest에 기록한다. 독립 branch별 bound를 최적화한 뒤 합한 값을 additive EPR이라고 간주하지 않는다.

## 3. 기준 이산 모델: h=1, M=1, k_B=1

2D periodic square lattice, Nx=Ny, N=Nx*Ny. 우선 h=1인 고정 coarse-graining cutoff에서 연구한다. 수식의 inner product는 벡터 내적 sum_i이다.

    L_x phi_i = 2 phi_i - phi_(i+x) - phi_(i-x)
    L_y phi_i = 2 phi_i - phi_(i+y) - phi_(i-y)
    L = L_x + L_y = -Delta_h
    mu_loc = a*phi + b*phi^3 + kappa*L phi
    dphi = -L mu dt + sqrt(2D) B dW
    B B^T = L

B는 periodic nearest-neighbor bond incidence matrix로 구현할 수 있다. noise는 각 bond의 독립 표준 Gaussian을 divergence하여 만든다. sitewise 독립 additive noise는 보존 noise가 아니므로 사용하지 않는다. 동등한 L^(1/2) noise도 가능하나 그 covariance를 검사한다.

    E[dphi_noise dphi_noise^T] = 2D L dt
    sum_i phi_i = N*phi_bar

chemical potential에서 공간 평균을 빼도 drift와 EPR contraction은 변하지 않는다. 단, step마다 field를 다른 평균으로 재설정하지 않는다. roundoff 수준의 mass correction이 필요하면 크기를 기록한다.

L의 Fourier symbol은 4 sin(q_x/2)^2 + 4 sin(q_y/2)^2이다. q=2*pi*fftfreq(N)와 일치하는 배열 순서를 사용한다. 아래의 kernel들도 같은 grid와 periodic convention을 따른다.

h를 바꾸는 별도 연구에서는 mu_i=(1/h^d)*partial F_h/partial phi_i, noise covariance=2D L dt/h^d, EPR=-h^d/D * E[mu_A dot dotphi]를 함께 바꿔야 한다. h만 바꾸고 기존 noise를 유지한 결과는 동일한 continuum model의 convergence test가 아니다.

## 4. Convolution, transpose, shell 분할

모든 kernel array는 FFT-origin 배열로 저장한다. 원점은 index (0,0)이다.

    (K f)_i = sum_s K_s f_(i-s)
    K^T_s = K_(-s mod lattice)
    K_e = (K + K^T)/2
    K_o = (K - K^T)/2

기본 Green kernel은 sum_s K_s=1이다. 이미 정규화된 이산 가중치이므로 다시 cell volume을 곱하지 않는다. delta kernel의 K_0=1과 uniform kernel의 K_s=1/N을 단위 test로 사용한다.

주의: torch.nn.Conv2d는 cross-correlation convention이다. 물리적 K_s를 NN weight와 비교할 때 offset s와 -s의 차이를 반영해야 한다. odd kernel의 전체 부호를 잘못 해석하기 쉬우므로 impulse-response test를 의무화한다.

signed offset은 주기 격자의 최소 이미지 convention을 사용한다. 짝수 격자의 Nyquist offset은 자기 자신과 반대 offset이 같으므로 odd part가 0이어야 한다. shell은 다음처럼 완전 분할한다.

    K_r(s) = K(s) * 1[max(|s_x|,|s_y|)=r]
    r=0,...,floor(Nx/2)  [정사각형]
    K = sum_r K_r

NN의 R=max_distance까지만 분할하고 나머지를 버리지 않는다. 물리적 ground truth에는 항상 r>R tail bucket을 별도로 포함한다. shell은 Euclidean radial annulus가 아니다. Euclidean 거리별 결과가 필요하면 별도 진단량으로 이름을 달리한다.

odd kernel은 방향 평균을 먼저 하면 사라질 수 있다. 각 offset의 kernel과 trajectory observable을 먼저 contraction한 뒤 shell 안에서 합한다.

## 5. 기본 수치 적분과 시간간격 검증

권장 기준 적분기는 additive noise를 가진 implicit midpoint이다.

    noise_n = sqrt(2D dt) * B * xi_n
    phi_next = phi_now - dt*L*mu((phi_now+phi_next)/2) + noise_n
    xi_n ~ N(0,I)

각 step에서 noise를 한 번만 생성하고 nonlinear solve 동안 고정한다. Newton/Krylov solve는 작은 dt에서 이전 step에 연속인 해를 찾는다. 권장 residual 기준은 ||res||_2 <= 1e-10 + 1e-8*||phi_next||_2이다. 실패 step을 clipping하거나 새 noise로 다시 시도해 숨기지 않는다. 실패하면 dt를 줄여 해당 run을 처음부터 재실행한다.

초기 dt 후보는 1e-3; 동일한 물리 parameter에서 dt/2, dt/4로 검증한다. 이 값이 모든 parameter에서 안전하다는 보장은 없다. 다른 integrator를 사용해도 되지만 noise covariance, stationary reference, EPR convergence test를 통과해야 한다.

implicit midpoint도 nonlinear passive 계의 equilibrium distribution/EPR을 유한 dt에서 일반적으로 정확히 재현하는 것은 아니다. 선형 OU covariance를 맞췄다고 finite-dt EPR까지 정확한 것도 아니다. physical continuous-time EPR은 dt와 observation lag를 줄여 비교한다.

EPR은 매 integration step에서 streaming accumulation한다. NN용 observation lag는 delta_t=dt, 4dt, 16dt 등을 별도 데이터로 저장한다. 이 lag에서의 endpoint estimator는 연속 trajectory EPR과 다른 finite-lag quantity일 수 있다. 서로 다른 lag 결과를 직접 같은 ground truth로 간주하지 않는다.

## 6. EPR: model-defined 정답

각 모델은 mu=mu_E+sum_c nu_c를 지정한다. mu_E=delta F/delta phi이고 nu_c는 active chemical-potential channel이다. c는 shell 또는 mechanism label이다.

    sigma_c = -(1/D) E[nu_c^T o dotphi]
    sigma_total = sum_c sigma_c
    sigma_density = sigma_total/N

'o'는 Stratonovich이다. discrete midpoint estimator는

    phi_m = (phi_next+phi_now)/2
    dphi = phi_next-phi_now
    dS_c = -dot(nu_c(phi_m), dphi)/D
    sigma_c_hat = sum_n dS_c / (n_steps*dt)

이다. scalar EPR 전체는 stationary continuum-time limit에서 비음수가 되지만 개별 channel은 signed contribution이다. negative shell을 clipping하거나 absolute value로 바꾸지 않는다.

독립적인 equal-time estimator도 기록한다. J_c,ij=partial nu_c,i/partial phi_j라 하면

    sigma_c = E[nu_c^T L mu]/D - E[Tr(L J_c)]

이다. nonlinear channel에서 두 번째 항을 임의로 생략하지 않는다. source channel nu=c*K(phi^2)에는

    J_c = 2c K diag(phi)
    Tr(L J_c) = 2c (L K)_00 * N*phi_bar

가 성립한다. 마지막 식은 translationally invariant periodic convolution이라는 가정에 의존한다. mean zero이면 전체 trace가 0이지만, 원리를 확인하지 않고 모든 nonlinear model에 적용하지 않는다. receiver nonlinearity나 local AMB에는 일반식을 사용한다.

quadratic conservative shell nu=K_e,r phi는 midpoint에서도

    nu(phi_m)^T dphi = F_r(phi_next)-F_r(phi_now)
    F_r(phi)=0.5*phi^T K_e,r phi

를 정확히 만족한다. 그 stationary mean work는 active dynamics가 함께 있어도 0이다. cubic 이상 F의 midpoint chain rule은 유한 step에서 일반적으로 정확하지 않다.

finite-run boundary term, sampling error, time-discretization error를 분리한다. nonstationary coarsening 구간의 active work를 stationary EPR이라고 보고하지 않는다.

## 7. Kernel ground truth와 NN output은 다른 대상이다

기준 thermodynamic force는 mass-conserving subspace에서

    d_star = -mu/D - grad_phi log P_ss

이다. 따라서 d_star와 -mu_A/D는 gradient functional만큼 다를 수 있다. mu_A의 shell 분해가 알려져도 학습된 각 branch의 평균이 그 분해와 같다고 자동으로 결론내릴 수 없다. 이는 전체 EPR을 맞춰도 남는 attribution 문제이다.

Pi=I-11^T/N이라 하고 C_mu=J_mu-J_mu^T라 하면

    Pi*(J_dstar-J_dstar^T)*Pi = -(1/D)*Pi*C_mu*Pi

이다. 소형 격자 Jacobian 또는 무작위 zero-mean 방향의 bilinear test로 확인한다. 큰 격자의 full Jacobian 저장은 필요하지 않다.

특정 branch r을 물리 channel nu_r에 대응시키는 충분조건은

    e_r = d_theta,r + nu_r/D
    Pi*(J_er-J_er^T)*Pi = 0

이다. smooth field-space와 fixed-mass affine subspace에서 이 residual은 gradient가 되어 stationary mean work가 0이다. 수치적으로 작은 curl만으로 global exactness를 증명한 것은 아니므로, 별도의 mean-work 비교도 한다.

branch attribution을 가정하지 않는 post-processing은

    sigma_c = -E[nu_c^T L d_star]
    sigma_c_est = -E[nu_c^T L sum_r d_theta,r]

이다. 전체 learned force가 정확해야 하며 알려진 nu_c를 사용하는 model-based 진단이다. model-free kernel recovery라고 부르지 않는다.

중요한 null: scalar on-site branch d_0(phi_i)는 어떤 U의 도함수이므로 stationary continuous-time mean E[d_0^T o dotphi]=0이다. 단, 물리적으로 'r=0'이라는 이름이 붙었어도 그 channel이 다른 shell의 입력에 의존하면 이 null은 적용되지 않는다.

## 8. Normalization과 총량/밀도

branch raw score는 공간 평균이다. phi 입력에서 dimensionless thermodynamic force를 학습했다면

    predicted_dS_total = N * sum_r branch_score_r
    predicted_sigma_total = E[predicted_dS_total]/delta_t

이다. extensive EP에 대한 exponential loss를 쓴다면 N factor가 학습 objective에도 어떻게 들어가는지 확인한다. 기존 trainer가 이미 합산했다면 다시 곱하지 않는다. 물리 EP와 raw score 비교 이전에 이 단위를 문서화한다.

정규화는 training split에서 구한 고정 affine map만 허용한다.

    z=(phi-m_train)/s_train
    d_z dz = d_phi dphi
    d_phi = d_z/s_train

Jacobian/curl도 같은 좌표 변환을 적용한다. frame별 mean/std normalization, spatial whitening, kernel convolution 전처리는 관측 dynamics와 shell 의미를 바꾸므로 금지한다. standardization은 positivity clipping이나 phi amplitude saturation과 다르다.

## 9. Tail, cutoff, 표현오차

Green kernel은 대개 infinite-support periodic kernel이다. NN 반경 밖을 잘라 renormalize하면 다른 물리 모델이다. 꼭 truncated model을 시험할 때는 별도 model_id와 cutoff, normalization을 기록하고 원래 Green model의 결과와 섞지 않는다.

각 run에서 최소한 다음을 저장한다.

    kernel_tail_L1 = sum_(r>R) |K_r| / sum_r |K_r|
    active_tail_ratio = E[nu_tail^T L nu_tail] / E[mu_A^T L mu_A]
    sigma_tail = sum_(r>R) sigma_r

active_tail_ratio는 작은 tail이라는 충분조건이나 EPR error bound가 아니다. 여러 mechanism의 cancellation으로 denominator가 작아질 수 있으므로 mechanism별 값도 저장한다. sigma_tail은 0 근처 전체 EPR로 나누지 말고 절대량/신뢰구간도 기록한다.

quadratic source K_r(phi^2)는 shell 입력에 대한 Hessian rank가 일반적으로 8r이다. branch 첫 linear projection의 width H가 8r보다 작으면 full input space에서 raw force의 정확한 표현을 보장할 수 없다. 이 사실만으로 trajectory distribution에서의 근사나 gradient-gauge를 통한 EP 추정까지 불가능하다고 단정하지 않는다. nonlinear receiver h(sum_r K_r phi)는 cross-shell coupling까지 가지므로 더 강한 구조적 mismatch가 있다.

h=1, fixed kernel, fixed R에서 Nx=32와 64 등의 비교는 finite-size test이다. grid refinement와 coarse-graining cutoff 변경은 다른 test이다. R=4 안에 대부분 들어오는 ell~1 정도의 실험은 architecture calibration이며 엄밀한 asymptotic long-range regime의 증거가 아니다.

## 10. 단계적 실험과 통계

1. Algebra/unit tests: mass conservation, noise covariance, convolution sign, K parity, kernel partition, functional derivative/curl, EPR channel sum.
2. Simulator-only pilot: Nx=32, 같은 parameter의 independent seeds 최소 5개. stationary homogeneous regime부터 시작한다.
3. Stationarity: <phi^2>, gradient energy, selected low-q structure factor, EPR block mean의 drift를 확인한다. patterned state는 domain size/coarsening도 확인한다.
4. 적어도 관심 slow observable의 integrated autocorrelation time tau_int를 추정한 뒤 burn-in과 block 길이를 정한다. 권장 출발은 burn-in>=20*tau_int, block>=5*tau_int, 전체 independent block>=30이다. tau 추정 자체가 안정하지 않으면 stationary rate를 확정하지 않는다.
5. EPR 양성 판정은 sigma_total>3*block_SE와 dt/lag convergence를 동시에 요구한다. 0-reference는 95% CI에 0이 포함되고 dt 감소에 따른 systematic bias가 active signal보다 작아야 한다.
6. 통과한 군만 fixed-architecture NN에 전달한다. split은 independent seed 또는 연속 시간 block 기준이며 인접·겹친 frame이 train/test에 동시에 들어가지 않게 한다.
7. total EPR, model-channel EPR, NN raw branch, NN-force projection EPR, projected curl을 각각 별도 column으로 기록한다.

초기 EPR이 약하면 nonlinear 모델에서 |g2| 또는 D를 단계적으로 키우고 amplitude/solver/stationarity를 재검사한다. phase separation을 먼저 강제로 만들지 않는다. finite-size stationary phase-separated ensemble을 확보하지 못했다면 transient morphology 결과만 보고한다.

본 패키지는 임의의 NN accuracy threshold를 물리적 성공 조건으로 제시하지 않는다. 상대오차는 truth가 0 근처일 때 의미가 없으므로 CI와 절대오차를 같이 사용한다.

## 11. 저장 규격

각 run directory에 아래 항목을 둔다.

    manifest.json
    kernel.npz          # K, Ke, Ko, Khat, shell labels; physical parameters
    fields.*            # [time, 1, Ny, Nx], float32 or float64
    times.*             # exact physical times
    ep_blocks.csv       # method, mechanism, shell, mean, SE, sample count
    diagnostics.json    # stationarity, mass, dt, covariance, tail tests
    model_checkpoint.*  # existing NN pipeline format, optional

simulator 및 EPR accumulation은 float64를 기준으로 한다. training frames를 float32로 저장한다면 작은 increment 손실을 측정한다. fine-lag dphi가 float32 ULP에 가까우면 float64를 보존한다.

manifest 필수: model_id, spec_version, seed, Nx, Ny, h, phi_bar, D, dt, observation_lags, burn_in_rule, measurement_length, 모든 mu/kernel parameter, solver 및 tolerance, kernel convention, normalization, source hash, 실제 NN 생성자 인자, loss 정의, split 방법, 알려진 tail/표현 한계.

## 12. 해석의 경계

- additive kernel work는 decomposition convention에 의존할 수 있다. independent positive dissipation channels를 의미하지 않는다.
- mediator를 제거한 Markovian scalar model의 물리적 사용에는 fast-relaxation, 약한 mediator noise, 빠른 orientation relaxation 등의 근거가 필요하다 [R2].
- b>0인 finite-grid 모델의 quartic local confinement는 quadratic nonlocal drift에 비해 큰 field에서 지배적이다. 이것이 continuum cutoff 제거의 well-posedness 정리까지 의미하지는 않는다.
- 문서의 수치 허용오차·pilot 값·실험 순서는 제안이다. validation/의 대수 검증 결과 외에 실제 NN 또는 trajectory 결과는 포함하지 않는다.