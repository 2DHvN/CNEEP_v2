# Scalar nonlocal AMB M00–M05

명세의 여섯 scalar 모델을 기존 CNEEP_v2의 시뮬레이션 → ShellForce 학습 → checkpoint → EPR 분석 흐름으로 실행합니다. 시작점은 [Corr_AMB_Kernel_ShellForce.ipynb](../../notebooks/Corr_AMB_Kernel_ShellForce.ipynb)입니다. 기본 실행은 짧은 CPU smoke이며 물리 검증 결과가 아닙니다.

## 실행

CNEEP_v2의 기존 Python 환경에서 notebook을 열고 위에서부터 실행합니다. 시뮬레이터는 NumPy/SciPy, 학습은 PyTorch, notebook 그림과 표는 Matplotlib/Pandas를 사용합니다. 네트워크 접근이나 자동 패키지 설치는 하지 않습니다.

- 기본: `n=12`, burn-in 64 steps, measurement 128 steps, 여섯 모델을 모두 저장·시각화합니다.
- 물리 실험: `RUN_EXTENDED=True`; 모델/control, 5 independent seeds, `dt`, 물리 측정 시간, block 길이 및 observation lag를 설정합니다. 결과 경로와 요약만 보관하고 trajectory를 run마다 메모리에서 해제합니다. 큰 run은 시작 전에 표시되는 저장 용량과 peak RAM을 확인합니다.
- 학습: `DO_TRAIN=True`; `MODELS_TO_TRAIN`으로 한 모델 또는 여섯 모델을 선택합니다. 짧은 `quick_runs`를 학습에 사용하지 않고 아래 규모의 독립 dataset을 별도로 생성합니다. 기본 2,000 iterations이며 수렴 보장은 아닙니다.
- split: train/validation/test 각각의 seed 집합에서 독립 trajectory를 생성합니다. 완료된 trajectory는 동일한 물리 parameter/kernel/seed/길이일 때 재사용할 수 있습니다. `DATASET_ROOT`에 기존 dataset 상위 폴더를 지정하세요.
- 기존 network의 `H=64`, `n_hidden=3`, `R=3`, ELU를 모든 군에서 동일하게 사용합니다. `ARCHITECTURE`를 바꾸려면 전체 실험군을 함께 바꾸고 별도 결과로 기록합니다.

Notebook 생성기는 `python scripts/make_amb_kernel_notebook.py`입니다. 생성 notebook은 코드/설정만 포함하고 저장된 실행 output은 포함하지 않습니다.

## 학습 데이터 규모와 진행 시간

기본 ensemble 크기는 현재 `Corr_labpm.ipynb`의 설정을 따릅니다. `Corr_labpm_ctmc.ipynb`는 train 2,048 / validation 256 / test 16을 사용하며, 기존 AMB2D ShellForce는 전체 128 중 train 102 / validation 26과 test 1을 사용합니다. 새 기본값은 LABPM 기준으로 고정했고 network H64/layers3/R3는 유지합니다.

| Split | 독립 trajectory | trajectory당 frame | lag=1 pair 수 |
|---|---:|---:|---:|
| Train | 1,024 | 1,025 | 1,048,576 |
| Validation | 256 | 1,025 | 262,144 |
| Test | 10 | 1,025 | 10,240 |

설정 위치는 notebook 학습 cell의 `TRAIN_M`, `VAL_M`, `TEST_M`, `N_INTERVALS`입니다. `TRAIN_N=32`, `BURN_IN_STEPS=10000`, `TRAIN_DT=0.001`, `TRAIN_OBSERVATION_STRIDE=1`이 기본입니다. Burn-in 10,000은 기존 AMB2D 실험을 참고한 출발값이며 각 모델의 stationarity와 correlation time에 맞춰 검증·조정해야 합니다.

32×32 float64 field 파일은 모델당 약 **10.09 GiB**(10.83 GB), 여섯 모델 약 60.5 GiB이며 진단/체크포인트 공간이 추가됩니다. 데이터 생성은 trajectory별로 저장하고 read-only memory map으로 재개합니다. 학습 정규화와 pair 접근도 부분적으로 읽으며, 모델 하나를 학습·평가한 뒤 map을 닫습니다. 그림에 필요한 test pair만 복사하므로 모든 모델의 전체 ensemble을 RAM에 올리지 않습니다.

학습 dataset 생성은 notebook의 `SIMULATION_DEVICE='auto'`로 CUDA 장치를 자동 선택하고, `SIMULATION_BATCH_SIZE=32`개의 독립 trajectory를 GPU batch로 함께 적분합니다. `'cpu'`는 기존 SciPy simulator를 선택하며 실제 병렬 batch 크기는 1입니다. `'cuda'`, `'cuda:0'`처럼 GPU를 명시하면 해당 장치를 사용할 수 없을 때 오류를 내고, `'auto'`일 때만 CPU를 선택할 수 있습니다. 선택한 backend/device와 batch 상한은 실행 전 표시되고 `training_plan.json`에 기록됩니다.

Simulation batch는 병렬 trajectory 수입니다. NN의 logical batch나 gradient 누적용 micro batch와 별도로 설정합니다. GPU 메모리가 부족하면 `SIMULATION_BATCH_SIZE`를 낮추세요. Notebook은 CUDA 장치명/총 VRAM과 한 개 float64 field batch tensor 크기를 표시하지만, 후자는 solver/kernel/noise/진단 buffer를 제외한 하한이며 peak VRAM 예측치가 아닙니다. Quick physics 진단 및 별도 control sweep은 CPU 기준 simulator를 사용하고, 대규모 학습 dataset 생성이 GPU 병렬 경로를 사용합니다.

GPU 경로는 FFT convolution, implicit midpoint 방정식 풀이, 매 step의 EPR·shell 진단을 float64 tensor batch로 계산합니다. CPU의 Newton/GMRES 대신 Fourier preconditioner를 이용한 반복 풀이를 사용하며, 같은 midpoint 방정식과 replica별 residual 기준 `atol + rtol * ||phi_next||`를 유지합니다. 미수렴 시 오류를 내고 noise를 다시 뽑거나 field를 보정하지 않습니다. 독립 seed와 batch 크기에 무관한 난수열을 유지하기 위해 각 trajectory의 PCG64 난수는 CPU에서 생성하여 GPU로 전달합니다. 관측 frame은 최대 32개씩 임시 보관한 뒤 디스크에 저장합니다.

CPU/GPU solver backend와 소스가 다른 dataset은 같은 경로에서 섞어 생성하지 않습니다. 같은 GPU backend에서 batch 크기를 바꾸는 것은 가능하며 완료된 replica는 재사용됩니다. 현재 개발 환경은 CPU 전용 PyTorch이므로 Torch CPU 경로의 수식·trajectory·저장 호환성과 notebook 실행을 검증했고, 실제 CUDA 실행 및 속도 측정은 CUDA 환경에서 확인해야 합니다.

학습은 logical train batch 2,048, validation/test batch 2,048, micro batch 32, 2,000 optimizer iterations입니다. Micro batch별 loss를 logical batch 크기로 가중하여 gradient를 누적하고 optimizer는 logical batch마다 한 번 갱신합니다. 기본 float64를 유지하며 CUDA가 있으면 사용합니다. 설정 cell의 `TrainingConfig(...)`에서 `device='cpu'`를 지정하면 CPU에서도 실행됩니다.

`DATASET_SMOKE=False`가 기본입니다. **연결 검사만** 하려면 `DO_TRAIN=True`, `DATASET_SMOKE=True`를 함께 설정하세요. 이때 grid 12×12, split당 2 trajectory, 32 interval, burn-in 16, 4 training iterations의 명시적인 작은 preset을 사용합니다. 실제 학습 dataset 기본값을 이 작은 크기로 바꾸지 않습니다.

`tqdm.auto` progress에는 완료 step/replica/iteration, elapsed, ETA가 표시됩니다. 긴 dataset bar는 burn-in과 measurement를 포함한 총 replica-step을 기준으로 진행합니다. CUDA에서는 현재 batch의 trajectory들이 함께 진행하므로 해당 batch 전체의 완료 step이 집계됩니다. 학습의 train/validation, test 평가, curl 검사, 모델 전체 loop와 controls/dt/seed sweep도 진행을 표시합니다. `SHOW_PROGRESS=False`로 notebook 전체 표시를 끌 수 있습니다.

Run manifest와 diagnostics의 `elapsed_seconds`, 학습/evaluation의 elapsed 기록 외에 notebook은 `stage_timings.csv`, `elapsed_summary.json`을 저장합니다. 최종 notebook wall time에는 cell 사이 대기 시간이 포함되고 개별 stage 시간은 실제 계산 구간입니다. 전체 pipeline 시간과 하위 stage 시간을 중복 합산하지 않습니다.

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

학습 dataset API는 다음과 같습니다. 아래 코드는 production 기본값을 구성하므로 실행 전 원하는 규모를 선택합니다.

```python
from data.amb_kernel.dataset import DatasetConfig, generate_dataset, load_dataset

config = DatasetConfig()  # 1024/256/10 replicas, each 1024 intervals
# config = DatasetConfig.smoke()  # explicit small execution check
with generate_dataset(pilot_config('M01', n=32), 'results/amb_kernel_dataset/M01',
                      dataset_config=config, device='auto', simulation_batch_size=32,
                      show_progress=True, resume=True) as dataset:
    splits = dataset.training_splits()
    # Pass splits to train_shell_force, then evaluate on splits.test while the maps are open.
# Reopen saved data later: dataset = load_dataset('results/amb_kernel_dataset/M01')
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

`learning.py`는 기존 alpha-NEEP objective와 AdamW를 사용합니다. branch raw score가 공간 평균이므로 joint loss의 **S=N*sum(branch_score)** 에 N을 정확히 한 번 적용합니다. 독립 branch별 exponential bound를 더하지 않습니다. `alpha=-0.5`의 sample loss는 `4*expm1(-S/2)`이며 alpha=0 극한도 지원합니다. Gradient 누적은 이 동일한 loss의 memory 분할이며 branch별 objective로 바꾸지 않습니다.

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

Tests는 operator/noise/convolution/transpose/functional derivative/trace/Gaussian reference와 저장·로드, scalar network reversal, affine force 단위, extensive joint loss, split 경계, CPU checkpoint/evaluation 및 독립 Gaussian-force projection oracle을 검사합니다. 독립 dataset의 규모·seed·resume/mmap 계약과 micro batch gradient 동등성, progress 표시가 수치 결과를 바꾸지 않는지도 검사합니다. 작은 smoke training은 통계적 학습 성공을 판정하지 않습니다.

현재 로컬 검증 환경의 PyTorch는 CPU-only입니다. 실제 CUDA 장치에서의 실행 시간, peak VRAM 또는 가속비를 측정했다고 주장하지 않습니다. GPU 실험에서는 장치별 속도/메모리와 수치 진단을 함께 확인하세요.

실제 물리 검증에는 independent seeds, 충분한 stationary sampling, 관심 slow observable의 안정적인 tau 추정, burn-in ≥20 tau, block ≥5 tau와 충분한 independent blocks, null controls 및 dt/lag 수렴이 필요합니다. 짧은 run에서 양의 점추정치가 나왔다는 이유로 EPR 양성이나 NN recovery를 주장하지 않습니다. M03 Gaussian covariance와 EPR는 analytic reference로 비교할 수 있습니다.

NN total 정확도, known-model projection, raw branch attribution은 서로 다른 결과입니다. Green kernel tail, source quadratic-rank 및 receiver cross-shell mismatch를 별도로 기록합니다. 작은 total curl residual은 모든 branch의 attribution이나 전역 exactness를 증명하지 않습니다.
