# L40S 실행 및 성능 확인

## 설치

노트북과 같은 Python/Jupyter kernel 환경에 최소 패키지를 설치한다.

```bash
python -m pip install -r data/lattice_abp_tc/requirements-l40s.txt
```

환경이 Jupyter kernel 목록에 없다면 한 번 등록한다.

```bash
python -m ipykernel install --user --name cneep-l40s --display-name "CNEEP L40S"
```

CUDA용 PyTorch는 위 파일에서 의도적으로 제외했다. 먼저 `nvcc --version`으로
로컬 CUDA Toolkit을 확인한 뒤 [PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)에서
호환되는 CUDA wheel을 골라 같은 환경에 설치한다. 이 시뮬레이션에는
`torchvision`과 `torchaudio`가 필요 없다. Torch를 설치하거나 교체했다면
Jupyter kernel을 재시작한다.

pip 외에 다음 시스템 도구가 필요하다.

- NVIDIA driver와 L40S가 보이는 CUDA-enabled PyTorch
- CUDA Toolkit의 `nvcc` (`nvidia-smi`에 표시되는 CUDA 버전만으로는 부족함)
- Linux의 `g++`/`c++` 또는 Windows의 MSVC C++ Build Tools
- 배포판 Python에 헤더가 빠진 경우 `python3-dev`/동등 패키지
- 쓰기 가능한 PyTorch extension build cache

L40S GPU 실행에는 `numba`, `seaborn`, `pandas`, `ipywidgets`,
`torchvision`, `torchaudio`가 필요 없다. `tqdm`은 progress bar용이며
시뮬레이션 수식에는 관여하지 않는다.

`mips_demo.ipynb`는 첫 셀부터 순서대로 실행하면 pip 항목, CUDA-visible
PyTorch, L40S compute capability, `CUDA_HOME`, `nvcc`, Ninja, host compiler를
검사하고 본 계산 전에 fused extension과 2-step CUDA smoke test를 실행한다.

## 실행

```powershell
python data/lattice_abp_tc/run_mips_demo.py `
  --device cuda:0 --backend cuda_fused --steps 5000 --save-interval 100
```

기본 `backend=cuda_fused`는 빠른 경로를 구성할 수 없을 때 즉시 오류를 내므로
느린 fallback을 모르고 실행하는 일을 막는다. `backend=auto`를 명시하면 fused
확장을 우선 시도한 뒤 fixed-shape Torch CUDA로 fallback할 수 있다. 격자 크기,
`dt`, WCA cutoff, 입자 수를 성능 때문에 바꾸지 않으므로
random-sequential single-particle update와 hop별 medium EP 정의는 그대로다.
이는 알고리즘 수준의 exactness를 뜻한다. CPU와 CUDA의 난수 스트림 및
부동소수점 초월함수는 장치별 비트 일치를 보장하지 않는다.

`resolved_backend`가 `cuda_fused`인지 확인하면 된다. JIT CUDA 확장을 반드시
사용하도록 검증하려면 `--backend cuda_fused`를 명시한다. 이 경우 CUDA
지원 PyTorch, CUDA toolkit의 `nvcc`, 호환되는 host compiler와 Ninja가
필요하며, 준비되지 않은 환경에서는 조용히 다른 계산법으로 바꾸지 않고
오류를 낸다. 첫 실행은 확장을 JIT compile하므로 느릴 수 있지만, 같은
PyTorch/CUDA/소스 조합의 다음 실행부터는 build cache를 재사용한다.

`ThermodynamicLatticeABPParams`를 직접 생성할 때의 라이브러리 기본 backend는
기존 수치 경로와의 호환을 위해 `torch`다. L40S fused 실행에는 데모처럼
`backend="cuda_fused"`를 명시한다.

GPU 내부 sweep 처리량은 다음처럼 측정한다.

```powershell
python data/lattice_abp_tc/benchmark_l40s.py `
  --device cuda:0 --backend cuda_fused --batch-sizes 1 16 64 `
  --warmup 3 --steps 100
```

벤치마크의 warm-up에는 lazy CUDA 초기화와 백엔드 준비가 포함되며, 측정
구간에서는 초기화·파일 저장·그림 생성·CPU 전송을 제외한다. 측정 후에는
입자수, 배타적 점유, site/occupancy 일치, EP 분해 항등식을 검사한다.

정확한 동역학을 유지하면서 조절해도 되는 실행 옵션은 다음과 같다.

- `save_interval`을 키우면 저장과 CPU 전송량만 감소하며 sweep은 생략되지
  않는다. CUDA probability status도 이 경계에서 모아 확인하므로 정상 실행의
  동역학은 그대로인 채 host synchronization 횟수도 줄어든다.
- `B`를 키우면 독립 ensemble의 총 처리량을 높일 수 있다. 하나의 trajectory가
  더 빨리 진행되는 것은 아니며 각 ensemble의 동역학은 변하지 않는다.
- `float32`와 `float64`는 같은 식을 계산하지만 반올림 결과는 다르다. 기존
  결과와 비교할 때는 기존 dtype을 유지한다.

CUDA 검증 테스트는 다음과 같이 실행한다. CUDA가 없는 개발 머신에서는 GPU
항목만 `skipped`가 되고, GPU-friendly 수식의 CPU reference 검사는 계속
실행된다.

```powershell
python data/lattice_abp_tc/test_cuda_exactness.py
python data/lattice_abp_tc/test_cuda_backend.py
```
