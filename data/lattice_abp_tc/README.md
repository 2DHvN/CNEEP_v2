# Thermodynamically consistent lattice ABP

이 디렉터리는 주기 경계를 갖는 2차원 square lattice에서 active Brownian
particles(ABP)를 5-color sublattice 방식으로 적분한다. 병진 hop은 local detailed
balance를 만족하도록 구성하고, 수락된 각 hop의 microscopic medium entropy
production(EP)을 정확히 누적한다.

핵심 구현은 [`core.py`](core.py), CPU fused backend는
[`_numba_backend.py`](_numba_backend.py), CUDA fused backend는
[`csrc/cuda_sweep_kernel.cu`](csrc/cuda_sweep_kernel.cu)에 있다. L40S 설치와
실행 방법은 [`L40S_RUN.md`](L40S_RUN.md)를 참고한다.

## 상태와 공간

- 물리 영역은 길이 `L`의 정사각형이고 양쪽 방향 모두 periodic boundary를 쓴다.
- 격자는 `G = grid_size`, 격자 간격은 `dl = L/G`이다.
- 입자 위치는 정수 site `sites[B,N,2]`, 방향은 `theta[B,N]`로 저장한다.
- physical position은 site 중심 `(site + 1/2) * dl`이다.
- 한 site에는 입자를 하나만 허용한다. 이미 점유된 목적지는
  `Delta V = +inf`, hop probability `0`으로 처리한다.
- WCA pair potential은 `r < 2^(1/6) sigma`인 lattice offset에 대해 한 번
  계산한 뒤 stencil로 재사용한다.

WCA potential은

$$
V_{\mathrm{WCA}}(r)=
4\epsilon\left[\left(\frac{\sigma}{r}\right)^{12}
-\left(\frac{\sigma}{r}\right)^6\right]+\epsilon
$$

이고 cutoff 밖에서는 0이다.

## 병진 전이 확률

현재 방향 벡터를
`e(theta) = (cos(theta), sin(theta))`, 네 이웃 방향의 변위를
`Delta r_d`라 하자. 한 입자를 occupancy에서 잠시 제외한 상태로 old/new WCA
energy를 계산하고

$$
\Delta V_d=V(\mathbf r+\Delta\mathbf r_d)-V(\mathbf r),
\qquad
A_d=v_0\mathbf e(\theta)\cdot\Delta\mathbf r_d-\mu\Delta V_d,
\qquad
x_d=\frac{A_d}{2D_t}
$$

를 만든다. 한 sweep에서 방향 `d`로 이동할 절대 확률은

$$
p_d=\frac{dt\,D_t}{dl^2}
\begin{cases}
\exp(x_d), & \texttt{prefactor="c0"},\\[3pt]
x_d\exp(x_d)/\sinh(x_d), & \texttt{prefactor="cv"}.
\end{cases}
$$

`cv` 식은 `x -> 0`에서 1이 되도록 안정적인 limiting form으로 계산한다. 두
prefactor 모두

$$
\frac{p(A)}{p(-A)}=\exp\left(\frac{A}{D_t}\right)
$$

를 만족한다. 따라서 reservoir temperature를 `T = D_t/mu`, `k_B = 1`로
놓으면 accepted hop의 Clausius medium EP는

$$
\Delta S_{\mathrm{med}}
=\log\frac{p_{\mathrm{forward}}}{p_{\mathrm{reverse}}}
=\frac{v_0\mathbf e(\theta)\cdot\Delta\mathbf r_d-\mu\Delta V_d}{D_t}
$$

이다.

`strict_probabilities=True`가 과학 계산의 기본 설정이다. 이때 모든 확률은
finite이고 `sum_d p_d <= 1 + tolerance`여야 한다. 조건을 어기면 simulation은
즉시 실패한다. `strict_probabilities=False`는 확률을 sanitize/renormalize하는
debug 경로이므로 비교 가능한 thermodynamic production run에는 사용하지 않는다.

## inverse CDF가 필요한 이유

inverse CDF는 새로운 물리량이나 근사를 계산하는 단계가 아니다. 한 입자의
substep에는 다음 다섯 개의 **상호배타적인** 결과가 있다.

1. `+x` hop
2. `-x` hop
3. `+y` hop
4. `-y` hop
5. stay

네 hop probability를 `p_0,...,p_3`라 하면 stay probability는

$$
p_{\mathrm{stay}}=1-\sum_{d=0}^3p_d
$$

이다. `u ~ Uniform[0,1)` 하나를 뽑고

$$
C_d=\sum_{j=0}^{d}p_j,\qquad
d=\min\{d:u<C_d\}
$$

로 방향을 고른다. 그런 `d`가 없으면 stay다. 즉 명시적인
`Categorical([p_0,p_1,p_2,p_3,p_stay])`와 같은 분포를 더 적은 allocation과
kernel dispatch로 샘플링하는 구현이다. 네 개의 독립 Bernoulli를 쓰면 한
substep에 여러 방향이 동시에 수락될 수 있으므로 현재 transition law와 같지
않다. Gillespie sampling도 가능하지만, 그것은 현재의 fixed-`dt` lattice
integrator와 다른 알고리즘이다.

CDF bin은 반드시 half-open interval

$$
[0,C_0),\ [C_0,C_1),\ldots,[C_2,C_3),\ [C_3,1)
$$

로 해석해야 한다. 현재 구현은 다음과 같이 첫 `CDF > u`를 선택한다.

- Torch: `(u >= cumulative).sum()`
- Numba/CUDA: 처음으로 `u < cumulative`가 되는 방향

이 표현은 `p_0=0`이고 `u=0`이어도 비어 있는 첫 bin을 건너뛴다.

## 한 simulation step

한 step은 5개의 color substep과 한 번의 angular Brownian update로 구성된다.

```text
for colour c in 0,...,4:
    selected <- particles at (x + 3y) mod 5 == c
    selected 모두에 대해 동일한 pre-substep occupancy에서 Delta V, p_d 계산
    inverse CDF로 selected의 hop 또는 stay를 독립 선택
    selected 결과를 한 번에 commit
    다음 colour 전에 새 occupancy에서 WCA를 다시 계산

theta <- (theta + sqrt(2 Dr dt) * Normal(0,1)) mod 2 pi
```

같은 colour의 cardinal-neighbor site는 존재하지 않으므로 hard-core destination
충돌 없이 동시에 commit할 수 있다. `B` replicas는 서로 다른 초기 상태와 난수를
갖는다. finite-range WCA에서 동시에 발생한 두 hop의 joint energy change는
single-particle rate에 포함되지 않는 finite-`dt` 효과가 있으므로, 해석과 검증
범위는 [`technial_report.md`](technial_report.md)를 따른다.

## exact medium EP와 local map

accepted hop마다 다음 세 값을 누적한다.

$$
\Delta S_{\mathrm{active}}
=\frac{v_0\mathbf e(\theta)\cdot\Delta\mathbf r}{D_t},
\qquad
\Delta S_{\mathrm{WCA}}=-\frac{\mu\Delta V}{D_t},
\qquad
\Delta S_{\mathrm{med}}
=\Delta S_{\mathrm{active}}+\Delta S_{\mathrm{WCA}}.
$$

- stay/rejected event의 increment는 0이다.
- finite한 음의 single-hop 또는 finite-time interval EP는 정상적으로 가능하다.
- `nan`, `+inf`, `-inf`는 유효한 thermodynamic trajectory에서 허용되지 않는다.
- local EP map은 각 hop의 **departure site**에 기록한다. 이는 local gauge의 한
  선택이며 learned map과 pixelwise 비교할 때 같은 convention을 확인해야 한다.
- `save_interval` 동안 발생한 모든 sweep의 EP를 합산한다. sweep 자체를
  subsample하지 않는다.

`T = n_steps // save_interval + 1`일 때 주요 출력 shape은 다음과 같다.

| key | shape | 의미 |
|---|---|---|
| `sites` | `[T,B,N,2]` | 저장 시점의 lattice sites |
| `theta` | `[T,B,N]` | 저장 시점의 orientations |
| `occupancy` | `[T,B,G,G]` | integer occupancy |
| `exact_medium_ep` | `[B,T-1]` | 저장 구간별 total medium EP |
| `exact_active_medium_ep` | `[B,T-1]` | active contribution |
| `exact_wca_medium_ep` | `[B,T-1]` | WCA contribution |
| `exact_medium_ep_maps` | `[T-1,B,G,G]` | departure-site local EP |

`exact_medium_ep_rate`는
`exact_medium_ep / (save_interval * dt)`이다. 마지막 production frame까지
저장하려면 `n_steps`가 `save_interval`로 나누어떨어져야 한다. burn-in frame과
burn-in EP는 반환하지 않는다.

## backends와 재현성

| backend | 역할 |
|---|---|
| `torch` | library reference/default; CPU 또는 fixed-shape Torch CUDA |
| `numba` | legacy CPU random-sequential backend; 5-color에서는 사용하지 않음 |
| `cuda_fused` | legacy random-sequential extension; 5-color에서는 Torch로 해석됨 |
| `auto` | 5-color에서는 Torch로 해석됨 |

5-color 설정에서는 Torch CPU/CUDA tensor path가 transition probability와
5-colour commit 순서를 담당한다. legacy CUDA fused kernel은 random-sequential
규칙을 구현하므로 사용하지 않는다. CUDA에서 5-color를 실행하려면
`device="cuda:0", backend="torch"`를 사용한다.

## 현재 KNEEP/MIPS state point

[`../../notebooks/Corr_LatticeABP_TC.ipynb`](../../notebooks/Corr_LatticeABP_TC.ipynb)
의 현재 production 설정은 다음과 같다. dataclass와 CLI의 일반 기본값과는
구분한다.

| parameter | value |
|---|---:|
| `L`, `grid_size`, `dl` | `16`, `32`, `0.5` |
| `sigma`, target `phi`, `N` | `0.5`, `0.30`, `391` |
| `epsilon`, `mobility` | `1`, `1` |
| `v0`, `Dr`, `Dt` | `50`, `1.5`, `1` |
| `Pe_r = v0/(Dr*sigma)` | `66.667` |
| `dt`, `prefactor` | `1e-4`, `cv` |
| backend, dtype, seed | `torch`, `float32`, `7` |
| run | `B=101`, burn-in `100,000`, production `100,000` |
| saving | every `100` steps: 1,001 frames/1,000 intervals |
| split | train `80`, validation `20`, test `1` replicas |

## 2026-08-01 inverse-CDF 경계 수정과 기존 결과

수정 전 구현은 CDF 경계를 inclusive하게 비교했다. 따라서 `u=0`이고 첫 방향의
확률이 0이면, 확률 0인 첫 방향을 잘못 선택할 수 있었다. 첫 목적지가 이미
점유된 경우에는 `Delta V=+inf`, `p_0=0`인데도 hop이 commit되어 single-site
exclusion이 깨지고 `Delta S_med=-inf`가 되었다.

이 수정은 양의 transition probability나 thermodynamic model의 parameter를
바꾼 것이 아니다. 원래부터 금지된 probability-zero event를 실제로 금지하도록
categorical sampler를 고친 것이다. 따라서 **의도한 모델의 MIPS phase diagram은
바뀌지 않는다.** 그러나 수정 전 trajectory에 forbidden overlap이 한 번이라도
들어갔다면 그 이후의 seeded trajectory, observable, EP는 모두 달라질 수 있다.

`float32` uniform을 약 `2^24`개의 값으로 보는 근사에서 `u=0`의 확률은
약 `5.96e-8`이다. 한 proposal에서는 작지만 현재 계산은 매우 길다.

| run | uniform draws | 예상 `u=0` 횟수 |
|---|---:|---:|
| 기존 `mips_demo`, `B=1`, `N=391`, 200k sweeps | `7.82e7` | `4.7` |
| 현재 KNEEP, `B=101`, `N=391`, burn+prod 200k | `7.90e9` | `471` |
| 기존 phase pilot 전체 42 points | `3.07e10` | `1,828` |

실제 forbidden hop 횟수는 이 중 첫 방향이 blocked/zero-mass인 경우만 세므로 더
작고 상태 의존적이다. 그래도 기존 `mips_demo.ipynb`에는 실제
`mean_exact_medium_ep_rate_by_ensemble = [-Infinity]`가 기록되어 있어 오류가
발현됐음이 확인된다. 기존 phase pilot은 `save_exact_medium_ep=False`였고 binary
occupancy를 검사하지 않아 같은 오류가 조용히 포함됐을 수 있다.

따라서 다음처럼 결과를 취급한다.

- gross MIPS morphology는 비슷하게 나올 가능성이 높지만, 이는 재실행 전에는
  검증된 결론이 아니다.
- 수정 전 exact/local EP trajectory와 KNEEP cache는 사용하지 않는다.
- 수정 전 phase heatmap, susceptibility, ridge 위치는 새 kernel로 다시 계산해
  통계 오차 안에서 비교한다.
- `phase_diagram_demo.ipynb` checkpoint에는 현재 simulator source hash가 없으므로
  재실행할 때 기존 `OUTPUT_TAG`를 쓰지 말고 새 tag/output directory를 사용한다.
- 새 source를 pull/copy한 뒤 먼저 Jupyter kernel을 재시작한다.

현재 KNEEP notebook cache는 simulator source SHA-256을 cache key에 포함하므로
source 수정 전 trajectory를 자동으로 재사용하지 않는다.

## 최소 사용 예

Google Colab에서는 [mips_demo_colab.ipynb](mips_demo_colab.ipynb)를 사용한다.
Colab GPU runtime에서 Google Drive의 `MyDrive/CNEEP_v2`에 현재 repository를
복사한 뒤 실행하면 된다. 이 notebook은 legacy CUDA extension을 빌드하지 않고
5-color Torch CUDA path를 사용한다.

```python
from data.lattice_abp_tc import (
    ThermodynamicLatticeABP,
    ThermodynamicLatticeABPParams,
)

params = ThermodynamicLatticeABPParams(
    N=391,
    L=16.0,
    grid_size=32,
    sigma=0.5,
    epsilon=1.0,
    mobility=1.0,
    v0=50.0,
    Dr=1.5,
    Dt=1.0,
    dt=1.0e-4,
    prefactor="cv",
    strict_probabilities=True,
    seed=7,
    device="cuda",
    dtype="float32",
    backend="torch",
)
sim = ThermodynamicLatticeABP(params)
result = sim.simulate(
    B=101,
    burn_in=100_000,
    n_steps=100_000,
    save_interval=100,
    save_exact_medium_ep=True,
    save_ep_maps=True,
)
```

긴 결과를 사용하기 전에는 최소한 다음 invariant를 검사한다.

```python
assert result["occupancy"].min() >= 0
assert result["occupancy"].max() <= 1
assert (result["occupancy"].sum((-2, -1)) == params.N).all()
assert result["exact_medium_ep"].isfinite().all()
assert result["exact_medium_ep_maps"].isfinite().all()
assert (
    result["exact_active_medium_ep"] + result["exact_wca_medium_ep"]
).allclose(result["exact_medium_ep"])
assert result["exact_medium_ep_maps"].sum((-2, -1)).T.allclose(
    result["exact_medium_ep"]
)
```
