# Scalar nonlocal AMB M00–M05

명세의 여섯 scalar 모델을 기존 CNEEP_v2의 시뮬레이션 → ShellForce 학습 → checkpoint → EPR 분석 흐름으로 실행합니다. 시작점은 [Corr_AMB_Kernel_ShellForce.ipynb](../../notebooks/Corr_AMB_Kernel_ShellForce.ipynb)입니다. 기본 실행은 짧은 CPU smoke이며 물리 검증 결과가 아닙니다.

## 실행

CNEEP_v2의 기존 Python 환경에서 notebook을 열고 위에서부터 실행합니다. 시뮬레이터는 NumPy/SciPy, 학습은 PyTorch, notebook 그림과 표는 Matplotlib/Pandas를 사용합니다. 네트워크 접근이나 자동 패키지 설치는 하지 않습니다.

- 기본: `n=12`, burn-in 64 steps, measurement 128 steps, 여섯 모델을 모두 저장·시각화합니다.
- 물리 실험: `RUN_EXTENDED=True`; 모델/control, 5 independent seeds, `dt`, 물리 측정 시간, block 길이 및 observation lag를 설정합니다. 결과 경로와 요약만 보관하고 trajectory를 run마다 메모리에서 해제합니다. 큰 run은 시작 전에 표시되는 저장 용량과 peak RAM을 확인합니다.
- 학습: `DO_TRAIN=True`; `MODELS_TO_TRAIN`으로 한 모델 또는 여섯 모델을 선택합니다. 기본 300 iterations는 연결 예시이며 수렴 보장이 아닙니다. 검증한 긴 run을 `load_experiment`로 읽어 `TRAINING_RUNS`에 지정합니다.
- split: 기본 `contiguous`는 frame을 먼저 분할한 뒤 pair를 만듭니다. `independent_seeds`는 train/validation/test seed 집합을 분리합니다.
- 기존 network의 `H=64`, `n_hidden=3`, `R=3`, ELU를 모든 군에서 동일하게 사용합니다. `ARCHITECTURE`를 바꾸려면 전체 실험군을 함께 바꾸고 별도 결과로 기록합니다.

Notebook 생성기는 `python scripts/make_amb_kernel_notebook.py`입니다. 생성 notebook은 코드/설정만 포함하고 저장된 실행 output은 포함하지 않습니다.

```python
from data.amb_kernel.model import pilot_config, control_configs
from data.amb_kernel.experiment import RunConfig, run_experiment, load_experiment

result = run_experiment(
    pilot_config('M01', n=12),
    RunConfig(seed=1, burn_in_steps=64, measurement_steps=128,
              block_steps=32, max_distance=3),
    output_dir='results/amb_kernel/example_M01',
)
# Same model definition, alternative reciprocal/null/reflection controls:
cases = control_configs('M01', n=12)
restored = load_experiment('results/amb_kernel/example_M01')
```

출력 폴더에 이미 run/checkpoint가 있으면 덮어쓰지 않습니다. 새 폴더를 지정하세요. 실패한 nonlinear solve는 clipping이나 noise 재추출로 숨기지 않으며 실패 자료가 저장됩니다. 원인을 확인하고 더 작은 dt로 새 run을 시작합니다.

## 모델과 controls

| ID | Chemical potential의 비국소/active 항 | 주요 controls |
|---|---|---|
| M00 | `g1 K_e phi + lambda0 Q(phi)` | passive `lambda0=0`, local AMB, delta |
| M01 | `g1 K_e phi + g2[K_e(phi²)+2 eta phi(K_e phi)]` | no source, reciprocal, delta, uniform |
| M02 | anisotropic even kernel의 M01 | 위 controls, isotropic, axis swap |
| M03 | `g1 K phi`, polar upwind Green kernel | Gaussian `b=0`, no flow, no interaction, even kernel, reflection |
| M04 | `g1 K phi + g2 K(phi²)` | directional/source only, full reciprocal, reflection, linear cancellation |
| M05 | `g1 K_e phi + g2(K_e phi)²` | no receiver, reciprocal, delta, uniform, matched M01 |

M04의 `eta=1` 부분 completion은 odd linear activity를 제거하지 않습니다. `full_reciprocal` control은 even linear sector와 complete cubic variation을 함께 사용합니다. M05는 source의 제곱과 receiver의 제곱을 구별하며, shell attribution은 `nu_r=(1-eta)g2*u*u_r` convention입니다.

[원본 명세](specs/README.md)와 [공통 프로토콜](specs/COMMON_PROTOCOL.md)을 함께 보세요. 원본 문서는 첨부 ZIP에서 가져온 모델 정의·설계 자료입니다. 참고문헌의 일반론을 구현 모델의 검증 결과로 대체하지 않습니다.

## 수치·물리 계약

상태는 `[time,1,Ny,Nx]`의 scalar `phi`입니다. `h=M=1`, periodic lattice, `L=-Delta`이며 covariance `2D L dt`인 보존 bond noise를 사용합니다. 적분기는 한 step의 noise를 고정하는 implicit midpoint입니다. 시뮬레이션/EPR 누적은 float64이고 optional float32 저장 시 increment 손실을 진단합니다.

Kernel은 FFT-origin 배열이며 `(K f)_i=sum_s K_s f_(i-s)`입니다. full periodic support를 버리거나 재정규화하지 않습니다. 실제 convolution과 NN `Conv2d`의 correlation convention 차이는 impulse test로 검증합니다. physical shells 전체와 `r>R` tail을 저장합니다. tail을 명시적 outer shells와 동시에 합해 두 번 세면 안 됩니다.

EPR은 매 integration step에서 두 방식으로 측정합니다.

```text
midpoint:          dS_c = -nu_c(phi_mid) dot dphi / D
                       sigma_c = sum(dS_c) / measurement_time
drift-minus-trace: sigma_c = <nu_c dot L(mu)>/D - <Tr(L J_c)>
```

Local AMB, nonlinear source와 receiver 각각의 trace를 사용합니다. 특히 nonzero mean source에서 trace를 생략하지 않습니다. raw work 및 개별 channel은 signed입니다. passive quadratic work는 endpoint free-energy 변화와 비교합니다.

저장 항목은 `manifest.json`, `kernel.npz`, `fields.npy`, `times.npy`, `ep_blocks.csv`, `ep_block_values.npz`, `observables.npz`, `diagnostics.json`입니다. manifest는 모델/solver/seed/time/lag 설정, source hash, 환경 버전, NN metadata를 기록합니다. 진단에는 mass, solver residual, stationarity/slow-observable tau, tail energy 및 sigma, 가능한 Gaussian reference가 포함됩니다.

## 기존 network 연결과 단위

`models/NEEP_ShellForce_2D.py`는 수정하지 않았습니다. 사용 클래스는 `MultiScaleShellForceCNEEP2D`이며 기존 scalar API가 `[B,2,Ny,Nx]`를 받으므로 singleton component 축만 제거합니다. `n_components=1`, `ep_component_indices=(0,)`, `learned_absolute`, exclusive Chebyshev shells, center mixing 없음, positional/external channels 없음으로 설정합니다. `phi²`, gradient, kernel signal을 NN 입력으로 추가하지 않습니다.

명세 기준 `shell_force.py` SHA256:

```text
baea2ddf87ed10ac5c4150617e8a24556f6d6490c54009b671d4254d1a79cf79
```

이 구현 시점의 기존 저장소 network SHA256:

```text
5468691638a875052ad13c28853514b03949986168a282505ec53d57172ba3e8
```

두 파일을 같은 소스라고 간주하지 않습니다. 실행 시 실제 hash를 다시 계산하고 차이, 클래스/입력 adapter, 실제 생성자 옵션을 manifest에 남깁니다.

`learning.py`는 기존 alpha-NEEP objective와 AdamW를 사용합니다. branch raw score가 공간 평균이므로 joint loss의 **S=N*sum(branch_score)** 에 N을 정확히 한 번 적용합니다. 독립 branch별 exponential bound를 더하지 않습니다. `alpha=-0.5`의 sample loss는 `4*expm1(-S/2)`이며 alpha=0 극한도 지원합니다.

정규화는 training split에만 fit한 고정 affine map입니다.

```text
z = (phi - m_train) / s_train
d_phi = d_z / s_train
sigma_NN_total = N * <sum(branch_score)> / observation_interval
sigma_NN_projection,c = -<nu_c^T L sum_r d_phi,r>
```

Projection은 equal-time rate이므로 observation interval로 다시 나누지 않습니다. 모델이 알려진 `nu_c`를 사용하는 후처리이므로 model-free mechanism recovery가 아닙니다. `evaluate_shell_force`는 `NN_total`, `NN_raw_branch`, `endpoint_midpoint`, `NN_force_projection`을 별도 행으로 저장합니다. endpoint quantity는 coarse observation pair에서 계산하며 integration-step streaming truth와 다릅니다. `projected_curl_check`는 zero-mean random bilinear probes로 total force curl과 analytic chemical-potential curl을 비교합니다.

Checkpoint는 기존 방식의 `settings`, `state_dict`, `optimizer`, `iteration`, `mean`, `std`와 추가 manifest/history를 저장합니다. 최저 validation loss checkpoint를 복원하며 test는 학습에 사용하지 않습니다. `load_checkpoint`는 tensor-safe loading을 사용합니다. `learning_manifest.json`, `training_history.json`, `nn_evaluation.csv`, `nn_evaluation_values.npz`도 저장됩니다.

## 검증 범위와 남은 실험

```powershell
python -m unittest discover -s tests -p 'test_amb_kernel*.py' -v
```

Tests는 operator/noise/convolution/transpose/functional derivative/trace/Gaussian reference와 저장·로드, scalar network reversal, affine force 단위, extensive joint loss, split 경계, CPU checkpoint/evaluation 및 독립 Gaussian-force projection oracle을 검사합니다. 작은 smoke training은 통계적 학습 성공을 판정하지 않습니다.

실제 물리 검증에는 independent seeds, 충분한 stationary sampling, 관심 slow observable의 안정적인 tau 추정, burn-in ≥20 tau, block ≥5 tau와 충분한 independent blocks, null controls 및 dt/lag 수렴이 필요합니다. 짧은 run에서 양의 점추정치가 나왔다는 이유로 EPR 양성이나 NN recovery를 주장하지 않습니다. M03 Gaussian covariance와 EPR는 analytic reference로 비교할 수 있습니다.

NN total 정확도, known-model projection, raw branch attribution은 서로 다른 결과입니다. Green kernel tail, source quadratic-rank 및 receiver cross-shell mismatch를 별도로 기록합니다. 작은 total curl residual은 모든 branch의 attribution이나 전역 exactness를 증명하지 않습니다.
