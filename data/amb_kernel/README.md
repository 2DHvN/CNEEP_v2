# Scalar nonlocal AMB M00–M05

물리 parameter·시스템 확인과 NN 학습을 두 notebook으로 분리했습니다.

| Notebook | 목적 | 기본 실행 |
|---|---|---|
| [Corr_AMB_Kernel_ShellForce.ipynb](../../notebooks/Corr_AMB_Kernel_ShellForce.ipynb) | 독립 dataset 생성, ShellForce 학습·평가, 영상·EP map·spectrum | 선택한 모델의 실제 학습 pipeline |
| [Sanity_AMB_Kernel.ipynb](../../notebooks/Sanity_AMB_Kernel.ipynb) | Parameter/control, stationarity, mass/solver, 물리 EPR 및 Gaussian calibration | M01 primary, seed 하나 |

학습 notebook은 별도의 짧은 실행이나 control sweep을 포함하지 않고, sanity notebook은 NN을 학습하지 않습니다. 기존 CNEEP_v2 환경의 NumPy/SciPy/PyTorch/Matplotlib/Pandas/tqdm을 사용하며 notebook에서 자동 설치나 네트워크 요청을 하지 않습니다.

## 학습 notebook

모델 목록과 물리 parameter를 설정한 뒤 순서대로 실행하면 dataset 생성 → training/validation → best checkpoint 복원 → 독립 test 평가 → 시각화가 이어집니다.

| Split | 독립 trajectory | trajectory당 interval / frame | lag=1 pair 수 |
|---|---:|---:|---:|
| Train | 1,024 | 1,000 / 1,001 | 1,024,000 |
| Validation | 256 | 1,000 / 1,001 | 256,000 |
| Test | 10 | 1,000 / 1,001 | 10,000 |

기본 grid는 32×32, TRAIN_DT=0.01, burn-in은 10,000 step입니다. Trajectory 수는 기존 LABPM 규모를 따릅니다. 학습은 2,000 optimizer iterations, train/eval logical batch 2,048, gradient 누적 micro batch 32이며 H64/layers3/R3/ELU architecture를 모든 모델에 고정합니다.

Float64 field 파일은 모델당 약 **9.85 GiB**, 여섯 모델 약 59.1 GiB이며 진단과 checkpoint 공간이 추가됩니다. Trajectory별 저장과 read-only memory map을 사용하고, 모델별 학습·평가를 마치면 map을 닫습니다. 기존 dataset은 물리 설정·kernel·seed·backend·source가 일치하는 완료 자료만 재사용합니다.

학습 결과에서 다음을 확인합니다.

- 실제 scalar phi profile 영상
- 알려진 active work와 NN 예측을 비교하는 EP-map 영상
- 전체 test 자료의 시간 평균 EP map
- Signed EP shell spectrum
- Training/validation loss curve, checkpoint 및 정규화·loss·split 기록

결과는 `results/amb_kernel/training/<timestamp>/` 아래에 저장됩니다. `visualizations/<model>/`의 `profile_video.html`, `ep_map_video.html`, `time_averaged_ep_map.png`, `ep_spectrum.png`에서 네 가지 시각화를 확인하고, 수치 자료는 `visualization_data.npz`, `ep_spectrum.csv`에 저장합니다. Parameter와 stationarity 검토에는 아래 sanity notebook을 사용합니다.

## Parameter / system sanity notebook

기본값은 **M01 / primary / seed 73031 한 run**, N=32, DT=0.01, burn-in 10,000, measurement 1,000 step입니다. SELECTED_MODELS, CASE_NAMES, MODEL_OVERRIDES, SEEDS에서 필요한 비교만 명시적으로 선택합니다. 선택하지 않은 모델이나 control을 추가 실행하지 않습니다.

DT_FACTORS=(1.0, 0.5, 0.25)로 dt 비교를 선택하면 burn-in/measurement/block 시간과 observation interval을 물리 시간 기준으로 맞춥니다. Integer step으로 표현할 수 없는 조합은 오류로 알립니다. Kernel의 full/odd component와 축 profile, field profile, 평균 phi², gradient energy, low-q structure factor, mass drift, solver residual을 확인합니다.

물리 EPR는 integration-step midpoint work와 drift-minus-trace를 비교하고 signed shell/tail 기여와 seed/dt 요약을 따로 표시합니다. Observation pair의 finite-lag work는 streaming EPR와 구분합니다. M03의 b=0 run을 선택하면 analytic covariance와 EPR의 Gaussian calibration을 추가로 표시합니다. 결과는 `results/amb_kernel/sanity/<timestamp>/` 아래에 저장됩니다.

## GPU와 진행 시간

SIMULATION_DEVICE='auto'는 CUDA가 있으면 GPU를 선택합니다. SIMULATION_BATCH_SIZE=32는 함께 적분할 독립 trajectory 수입니다. Sanity에서는 같은 model/control/dt의 seeds를 묶고, 학습 dataset에서는 동일한 물리 설정의 replicas를 묶습니다. CPU는 기준 SciPy simulator를 사용합니다.

'cuda' 또는 'cuda:0'을 명시했는데 해당 장치를 사용할 수 없으면 오류를 냅니다. 'auto'일 때만 CPU 선택이 가능합니다. Simulation batch는 NN의 logical/micro batch와 별도 설정이며 GPU 메모리가 부족하면 이를 줄일 수 있습니다. NN 학습 장치도 독립적으로 선택합니다.

진행률에는 elapsed/ETA와 완료 step·replica·iteration을 표시합니다. Dataset 전체 replica-step, training/validation/test와 모델 loop의 진행을 확인할 수 있고 결과에 시간 기록을 저장합니다. 현재 로컬 검증 환경은 CPU-only PyTorch이므로 실제 GPU 속도·peak VRAM·가속비 측정을 했다고 주장하지 않습니다.

## 모델과 수치 계약

| ID | 주요 interaction | 주요 controls |
|---|---|---|
| M00 | reciprocal nonlocal + local AMB | passive, local AMB, delta |
| M01 | isotropic nonlinear source | no source, reciprocal, delta, uniform |
| M02 | anisotropic even nonlinear source | M01 controls, isotropic, axis swap |
| M03 | polar linear kernel | Gaussian, no flow, no interaction, even kernel, reflection |
| M04 | directional + nonlinear source | directional/source only, full reciprocal, reflection, cancellation |
| M05 | nonlinear receiver | no receiver, reciprocal, delta, uniform, matched source |

정확한 식과 가정은 [원본 모델 명세](specs/README.md)와 [공통 프로토콜](specs/COMMON_PROTOCOL.md)에 있습니다. M04의 eta=1만으로 directional activity가 없어지지 않으며 M05의 shell attribution은 u*u_r convention입니다.

상태는 보존 scalar phi 하나, h=M=1이며 noise covariance는 2D L dt입니다. Float64 implicit midpoint의 한 step에서 noise를 고정합니다. Solver 실패는 clipping이나 noise 재추출로 숨기지 않고 partial artifacts를 보존합니다. Full periodic kernel과 NN 반경 밖 tail을 버리거나 다시 정규화하지 않습니다.

models/NEEP_ShellForce_2D.py는 수정하지 않았습니다. 기존 scalar 생성자 옵션 n_components=1, ep_component_indices=(0,), exclusive learned_absolute shells를 사용합니다. phi², gradient, kernel signal을 추가 입력으로 넣지 않습니다. 첨부 명세의 기준 source hash와 저장소의 실제 network hash는 다르며 실제 hash·클래스·scalar input adapter·생성자 설정을 manifest에 기록합니다.

    z = (phi - m_train) / s_train           # train-only fixed affine normalization
    d_phi = d_z / s_train
    S = N * sum(branch_score)              # joint extensive alpha-NEEP loss, AdamW
    sigma_NN_total = <S> / observation_interval
    sigma_projection,c = -<nu_c^T L sum_r d_phi,r>

Branch score는 공간 평균이므로 N을 한 번 적용합니다. 독립 branch별 bound를 더하지 않습니다. 위 projection 식은 learning.py의 선택적 평가 API이며 학습 notebook의 자동 시각화 출력에는 포함되지 않습니다. Projection rate를 lag로 다시 나누지 않고 NN total, raw branch, 알려진 active force의 projection, 물리 channel work를 서로 구분합니다. Scalar on-site null과 M05의 physical r=0 attribution도 구분합니다.

Run 파일은 manifest.json, kernel.npz, fields.npy, times.npy, ep_blocks.csv, ep_block_values.npz, observables.npz, diagnostics.json입니다. 학습 checkpoint에는 기존 settings, state_dict, optimizer, iteration, mean, std와 추가 manifest/history를 저장합니다.

## 재생성과 검증

CNEEP_v2에서 실행합니다.

    python scripts/make_amb_kernel_notebook.py
    python scripts/make_amb_kernel_sanity_notebook.py
    python -m unittest discover -s tests -p 'test_amb_kernel*.py' -v
    python tests/check_amb_kernel_notebooks.py --output-dir output/notebook_check

마지막 개발 검증 명령은 새 output directory를 사용합니다. 실행 비용을 줄이는 설정은 검증 코드가 메모리 안에서만 바꾸며 공개 notebook 기본값을 별도의 짧은 preset으로 바꾸지 않습니다.

실제 물리 판정에는 independent seeds, 안정적인 correlation-time 추정, 충분한 burn-in과 blocks, reciprocal/null controls, dt/lag 수렴이 필요합니다. sigma > 3*block_SE 한 조건이나 학습 loss 감소만으로 stationarity, 양의 EPR 또는 kernel attribution을 검증했다고 결론내리지 않습니다.
