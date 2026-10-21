# 참고문헌과 이 명세에서의 역할

확인일: 2026-10-20. 이 문서는 단순 관련 논문 목록이 아니라 무엇이 문헌에 근거하고 무엇이 여기의 모델링 선택인지 구분하기 위한 목록이다. 본 패키지의 구체적인 parameter, upwind discretization, quadratic response family, kernel별 work 분해, NN 검증 순서는 별도 표시가 없는 한 이론적 설계 제안이다.

## R1 — Scalar field trajectory EPR

Cesare Nardini, Étienne Fodor, Elsen Tjhung, Frédéric van Wijland, Julien Tailleur, Michael E. Cates.
**Entropy Production in Field Theories without Time-Reversal Symmetry: Quantifying the Non-Equilibrium Character of Active Matter.**
Physical Review X **7**, 021007 (2017).
DOI: `10.1103/PhysRevX.7.021007`. arXiv: `1610.06112`.

역할: AMB의 path-probability 기반 scalar EPR, active chemical-potential work, conservative boundary term과 spatial/spectral decomposition의 기준. 본 명세는 그 scalar-EPR 정의를 채택하지만 HSR을 kernel attribution의 정의로 사용하지 않는다. 새 nonlinear/odd kernel 모델의 EPR 값이나 NN 성능이 이 논문에서 검증된 것은 아니다.

## R2 — Taxis/QS의 microscopic-to-scalar coarse-graining

Alberto Dinelli, Jérémy O'Byrne, Julien Tailleur.
**Fluctuating hydrodynamics of active particles interacting via taxis and quorum sensing: static and dynamics.**
Journal of Physics A: Mathematical and Theoretical **57**, 395002 (2024).
DOI: `10.1088/1751-8121/ad72bc`. arXiv: `2402.05072v2` (수정본 2025-01-13).

역할: conserved density가 유일한 hydrodynamic mode인 dry active particles의 coarse-graining, 빠른 신호의 Green-function elimination, taxis와 motility regulation의 연결. 실제 microscopic 모델에서는 density-dependent mobility/noise가 생길 수 있다. 본 패키지의 constant mobility와 local phi^4 truncation은 추가 근사이다. 특히 quadratic source S(rho)~rho^2나 특정 anisotropic source-response law가 이 논문에서 실증되었다고 주장하지 않는다.

## R3 — Reciprocal nonlocal phase separation / Zwicker 계열

Filipe C. Thewes, Yicheng Qiang, Oliver W. Paulin, David Zwicker.
**Phase separation with nonlocal interactions.**
Physical Review Research **8**, L022023 (2026); 출판일 2026-05-04.
DOI: `10.1103/gdwt-v9yj`. preprint arXiv: `2511.05214`.

역할: free-energy 기반 nonlocal scalar phase separation의 동기와 M00의 passive reference. M00의 screened kernel 선택은 비교를 위한 선택이며 해당 논문의 전체 kernel family를 동일하게 재현한다고 하지 않는다. M01–M05의 nonreciprocal active chemical potentials 또는 그 NN decomposition이 이 논문에 그대로 제시된 것은 아니다.

## R4 — Reciprocity가 coarse-graining 후 남는 조건

Alberto Dinelli, Jérémy O'Byrne, Agnese Curatolo, Yongfeng Zhao, Peter Sollich, Julien Tailleur.
**Non-reciprocity across scales in active mixtures.**
Nature Communications (2023); 출판일 2023-11-03.
DOI: `10.1038/s41467-023-42713-5`.

역할: microscopic mediated nonreciprocity와 coarse-grained integrability의 차이, reciprocal effective description이 복구될 수 있다는 경고. 이 연구는 mixtures를 다루므로 여기의 단일 scalar quadratic-source model을 그대로 검증한 연구처럼 인용하지 않는다. 본 명세의 single-scalar functional curl은 주어진 chemical potential을 직접 미분하여 검증한다.

## R5 — Differential-flow chemotaxis의 물리적 근거

Javier Muñoz-García, Zoltán Neufeld.
**Aggregation of chemotactic organisms in a differential flow.**
Physical Review E **80**, 061902 (2009).
DOI: `10.1103/PhysRevE.80.061902`. arXiv: `0901.1831`.

역할: 입자가 직접 advect되지 않아도 신호의 상대적인 advection이 anisotropic/moving aggregation에 영향을 줄 수 있는 물리 설정. M03/M04는 이를 fast-mediator scalar kernel로 단순화한 확장이다. constant-noise AMB, Gaussian kernel EPR trace, neural shell attribution을 원 논문의 결과로 돌리지 않는다.

## R6 — Neural EPR inference

Dong-Kyum Kim, Youngkyoung Bae, Sangyun Lee, Hawoong Jeong.
**Learning Entropy Production via Neural Networks.**
Physical Review Letters **125**, 140604 (2020).
DOI: `10.1103/PhysRevLett.125.140604`. arXiv: `2003.04166`.

역할: stationary forward/reverse pair probability에 기반한 neural EP inference의 일반적 배경. 사용자 제공 shell_force.py에는 loss 코드가 없으므로 실제 trainer가 이 논문의 objective와 같다고 가정하지 않는다. 물리 channel 분해와 learned branch attribution의 동일성도 자동으로 따라오지 않는다.

## R7 — Angular nonreciprocity에서 vector field가 남을 수 있다는 경고

Zhi-Feng Huang, Michael te Vrugt, Jonas Mayer Martins, Raphael Wittkowski, Hartmut Löwen.
**Active pattern formation emergent from single-species nonreciprocity.**
arXiv: `2404.10093v2` (2025-10-07).

역할: single-species vision-cone nonreciprocity의 microscopic-to-continuum 사례. 해당 Active Model N은 density와 polarization을 결합하므로, 임의의 vision-cone 모델이 단일 scalar Markov theory로 즉시 환원된다는 근거로 쓰지 않는다. 본 패키지는 느린 orientation을 피하기 위해 고정 substrate director/외부 transport direction을 사용한다.

## 사용자 제공 구현

`shell_force.py`: 클래스 ShellAbsoluteConv2d, _ShellForceBranch2D, ShellForceKNEEP2D.
SHA-256: `baea2ddf87ed10ac5c4150617e8a24556f6d6490c54009b671d4254d1a79cf79`.

확인한 계약: Chebyshev shell mask, periodic padding, pointwise head, input pair의 midpoint/increment, 공간 평균 branch score, force 반환 옵션. loss, simulator, 실제 hyperparameter, normalization은 이 파일만으로 확정할 수 없다.

## 증거 수준 표기

- **문헌 근거**: 위 R1–R7에 실제로 있는 일반 체계 또는 물리 설정.
- **직접 유도**: 명세한 방정식에서의 functional derivative/curl, mass projection, kernel partition, Gaussian covariance/EPR.
- **검증 스크립트**: validation/의 finite-grid algebra와 OU 행렬 계산. 실제 stochastic sampling/NN 학습 아님.
- **실험 가설**: EPR의 크기, pattern, branch recovery, parameter dependence. 실제 사용자의 실험으로 확인할 대상.