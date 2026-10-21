# Scalar nonlocal AMB — 모델별 실험 명세

Version 1.0 | 2026-10-20

사용자 제공 `shell_force.py`의 architecture를 바꾸지 않고, scalar phi trajectory에서 Nardini-type EPR과 physical kernel decomposition을 비교하기 위한 설계 문서이다. **모델별 spec 6개**, 공통 프로토콜, 참고문헌, 비학습 대수 검증을 포함한다.

목표는 HSR의 wavevector/frequency decomposition이 아니라 dynamics에 정의된 kernel operator의 work contribution이다. 단, physical kernel decomposition과 learned branch attribution이 자동으로 같다고 가정하지 않는다.

## 빠른 시작

먼저 [공통 프로토콜](COMMON_PROTOCOL.md)의 이산화, noise, EPR normalization, NN input 계약을 적용한다. 그다음 아래에서 목적에 맞는 model spec 하나를 선택한다. 각 spec의 방정식과 pilot YAML이 model definition이며 공통 프로토콜이 runtime/measurement contract이다.

| ID / 개별 문서 | 핵심 chemical-potential 항 | 물리적 역할 | 실험 우선순위 |
|---|---|---|---|
| [M00 — Reciprocal nonlocal AMB](MODEL_00_reciprocal_nonlocal_AMB.md) | g1 K_e phi + lambda0 Q(phi) | Zwicker-type passive nonlocal sector의 대조군 | 필수 수치 reference |
| [M01 — Isotropic nonlinear source](MODEL_01_isotropic_nonlinear_source.md) | g1 K_e phi + g2 K_e(phi^2) | 등방적 source–response mismatch | 핵심 물리 후보 |
| [M02 — Nematic nonlinear source](MODEL_02_nematic_nonlinear_source.md) | even anisotropic K에 M01 적용 | anisotropy와 nonreciprocity 분리 | M01 이후 |
| [M03 — Polar linear kernel](MODEL_03_polar_linear_kernel.md) | g1 K phi, K_o != 0 | 상대 신호 수송에서 유래한 directional coupling | 핵심 calibration 후보 |
| [M04 — Mixed polar source](MODEL_04_mixed_polar_nonlinear_source.md) | g1 K phi + g2 K(phi^2), K_o != 0 | directional/nonlinear 두 mechanism의 간섭 | M01/M03 이후 |
| [M05 — Nonlinear receiver](MODEL_05_nonlinear_receiver_control.md) | g1 K_e phi + g2 (K_e phi)^2 | 총신호에 반응하는 QS / 표현 한계 control | 선택적 stress test |

M00 표의 local activity는 실제 문서에서 central-gradient stencil로 정확히 정의한다. M01/M02의 eta reciprocal-completion family와 M03의 Gaussian/nonlinear variants는 각 동일 model spec 안의 control configuration이다.

## 권장 실행 순서

**M00 equilibrium controls -> M03 Gaussian calibration 및 M01 homogeneous nonlinear pilot -> M02 anisotropy -> M04 mixed mechanisms.** M05는 physical receiver mechanism을 비교하거나 현재 shell-additive network의 적용 한계를 조사할 때 추가한다.

실험 자원이 제한되어 있으면 M00, M01, M03 세 문서부터 시작하는 것이 좋다. M01은 등방성을 유지하는 물리 후보이고, M03은 kernel/force/EPR의 정답을 가장 명확히 대조할 수 있는 후보다. 두 모델의 장점은 서로 다르다.

## 각 model spec에 들어 있는 내용

모든 문서에는 물리적 가정과 문헌의 지원 범위, 정확한 scalar dynamics, kernel 생성법과 sign/normalization, active/passive 분리, functional curl, model-defined EPR, reciprocal/null controls, pilot parameter와 단계적 sweep, shell-force 표현 한계, 통과 기준 및 저장 observable을 명시했다.

parameter는 **실험으로 최적화된 값이 아니다.** 안정한 homogeneous pilot부터 시작해 stationarity, 시간간격, 유의한 EPR을 확인한 뒤 parameter를 확장한다. 원하는 microphase나 NN recovery가 자동으로 나올 것으로 가정하지 않는다.

## Architecture를 유지한다는 의미

제공된 코드의 Chebyshev shell mask, branch/head, 입력 계약을 보존한다. 단일 scalar 연구이므로 기존 생성자 옵션 `n_components=1`, `ep_component_indices=(0,)`를 사용한다. 추가적인 phi^2/kernel/gradient 채널이나 cross-shell layer는 넣지 않는다. 기존 H/R/activation 설정이 있으면 모든 모델에서 고정한다.

source hash: `baea2ddf87ed10ac5c4150617e8a24556f6d6490c54009b671d4254d1a79cf79`.

원본 NN 파일은 수정하지 않았으며 이 패키지는 simulator/trainer 구현을 포함하지 않는다. 실제 loss와 normalization은 제공된 architecture 파일만으로 확인되지 않아, 공통 프로토콜에 확인·기록할 계약을 별도로 적었다.

## 반드시 분리해서 보고할 세 가지 결과

1. **Physics truth**: simulator에서 알려진 kernel의 Stratonovich work 또는 drift-minus-trace로 측정한 EPR.
2. **Total neural inference**: 전체 learned force/score의 scalar EPR 정확도.
3. **Kernel attribution**: raw branch 또는 known-force projection이 physical kernel contribution과 맞는지.

하나를 통과했다고 다른 둘을 통과한 것은 아니다. reciprocal kernel의 work는 0이지만 learned force의 score correction 때문에 먼 shell branch가 nonzero일 수 있다. source nonlinearity는 같은 shell 내 quadratic-rank 문제가 있고, receiver nonlinearity는 cross-shell dependency까지 있다.

NN R 밖 tail은 physical model에서 삭제하지 않는다. shell coverage 밖의 force와 EPR을 별도 bucket으로 기록한다. R=4 안에 interaction을 억지로 자르고 long-range model 전체를 복원했다고 하지 않는다.

## 이 패키지에서 실제로 확인한 것

[대수 검증 보고서](validation/README.md)와 [검증 코드](validation/verify_spec_math.py)를 포함한다. 검증은 7x7 periodic lattice의 finite differences, functional curl, kernel partition, source/receiver reciprocal completion, nonzero-mean Ito correction, Gaussian OU covariance/EPR trace의 행렬 계산이다.

**NN 학습과 stochastic trajectory simulation은 실행하지 않았다.** 해당 단계의 stationarity, EPR signal strength, tail 영향, branch recovery는 사용자의 실제 실험에서 검증할 대상이다. 대수 검증 통과를 최종 물리적 validation으로 확대 해석하지 않는다.

검증 스크립트 재실행:

```bash
python validation/verify_spec_math.py
```

Python 3, NumPy, SciPy가 필요하다. package version을 pin한 simulator 환경을 제공하는 것은 아니며, 실행 환경 버전은 본인의 run manifest에 기록한다.

## 파일 구성

```text
amb_kernel_specs/
  README.md
  COMMON_PROTOCOL.md
  REFERENCES.md
  MODEL_00_reciprocal_nonlocal_AMB.md
  MODEL_01_isotropic_nonlinear_source.md
  MODEL_02_nematic_nonlinear_source.md
  MODEL_03_polar_linear_kernel.md
  MODEL_04_mixed_polar_nonlinear_source.md
  MODEL_05_nonlinear_receiver_control.md
  validation/
    README.md
    verify_spec_math.py
    results.json
```

외부 근거의 역할은 [REFERENCES.md](REFERENCES.md)에 구분했다. 문헌의 일반적 체계, 이 명세의 직접 유도, 아직 실험하지 않은 가설을 혼동하지 않는 것이 이 실험군의 핵심이다.