# 명세의 비학습 이론 검증

실행일: 2026-10-20
실행 파일: `verify_spec_math.py`
상세 결과: `results.json`
상태: **모든 구현된 check가 설정 tolerance를 통과했다.**

## 범위

7x7 periodic lattice, seed 73031, float64 선형대수로 검증했다. NN 학습, SDE trajectory 생성, stationarity sampling, phase diagram 탐색은 수행하지 않았다. 따라서 아래의 통과는 대수·이산 operator 일관성의 확인이며, 실제 실험의 EPR 크기나 학습 성능에 대한 증거가 아니다.

코드는 이 패키지의 mu/kernel convention을 사용한다. 이전 대화에서 생성된 별도 검증 스크립트의 수치를 재사용한 것이 아니다.

## 검증한 항목

| 항목 | 결과의 의미 |
|---|---|
| B B^T=L, L*1=0 | 보존 noise factorization과 질량 mode |
| even/anisotropic kernel symmetry | even angular anisotropy 자체가 antisymmetric coupling이 아님 |
| K(-v)=K(v)^T, sum K=1, positive entries | upwind Green kernel convention |
| M00 gradient-square Jacobian | local AMB 이산 active term의 미분 |
| M01/M02/M04 analytic Jacobian와 finite difference | nonlinear source 및 mixed kernel 미분 |
| source cubic functional variation | K(phi^2)+2 phi*(K^T phi) completion |
| M05 receiver Jacobian 및 cubic completion | (K phi)^2+2 K^T(phi*(K phi)) 미분 |
| projected reciprocal curl=0 | fixed-mass subspace의 integrability controls |
| passive quadratic midpoint chain rule | passive shell work가 endpoint 차이라는 identity |
| full shell partition | source/receiver shell 항과 원 operator의 합 일치 |
| nonzero-mean source Ito trace | drift estimator에서 trace correction의 필요성과 정확한 식 |
| receiver u*u_r attribution과 Jacobian 합 | M05에서 명시한 convention의 algebraic consistency |
| M03 covariance / Lyapunov / probability-current EPR | Gaussian benchmark의 독립적인 행렬 계산 |

설정 tolerance는 각 기록된 error에 대해 1e-7이다. error가 relative norm인지 absolute residual인지는 스크립트에서 각각 정의했다. 서로 단위가 다른 error들을 하나의 물리적 정확도 지표로 해석하지 않는다.

## 주요 수치

source cubic functional의 finite-difference derivative 상대오차: 약 4.90e-10.
receiver cubic functional의 derivative 상대오차: 약 4.59e-10.
M03 stationary covariance의 해석식/Lyapunov solution 상대오차: 약 7.89e-15.
M03 thermodynamic-force matrix 비교 상대오차: 약 4.21e-12.

Gaussian benchmark의 총 EPR:

    probability-current 계산: 0.0012454226986791225
    real-space trace 계산:    0.0012454226986791362

shell별 trace 기여 (r=0,1,2,3):

    0.0
    0.0008182508462867479
    0.0003855885818450181
    0.000041583270547369973

이 예에서는 모두 비음수지만, 일반적으로 individual shell contribution의 비음수성을 보장하지 않는다. 여기의 값은 테스트 전용 7x7 lattice의 값이며 Nx=32/64 pilot의 예상 EPR로 사용하지 않는다. 작은 periodic box에서의 검증이 long-range tail의 finite-size physics까지 검증한 것도 아니다.

## 재현

패키지 root에서 다음을 실행한다.

```bash
python validation/verify_spec_math.py
```

NumPy와 SciPy가 필요하다. 스크립트는 실행 시 `results.json`을 재작성한다. 플랫폼과 선형대수 라이브러리에 따라 마지막 자리의 차이가 생길 수 있다. failure가 발생하면 tolerance를 먼저 완화하지 말고 operator, dtype, 라이브러리 환경을 확인한다.

## 아직 검증하지 않은 것

실제 parameter에서의 stationary distribution 수렴, EPR confidence interval, finite-dt/observation-lag bias, large-domain morphology, NN의 optimization 및 expression error, learned branch의 고유성, 실제 물리계의 fast-mediator/constant-mobility 가정은 별도 실험 대상이다. 그 검증 절차는 COMMON_PROTOCOL.md와 각 model spec에 적었다.