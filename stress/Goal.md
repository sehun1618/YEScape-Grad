# PBC DC-DFT Analytic Stress Goal

## 0. 문서의 지위

이 문서는 PBC DC-DFT stress/cell derivative를 기존 gradient 구현과 분리된
독립 작업 공간에서 개발하기 위한 goal이다.

현재 성능 최적화와 contracted GPU route의 production 승격 작업은
`stress/Goal_Stress_Optimization.md`에서 별도로 관리한다. 이 문서의 이론, convention과
지원 범위가 그 optimization goal보다 우선한다.

- 모든 stress 전용 source, test, benchmark와 문서는 `stress/` 아래에 둔다.
- 기존 `grad/`, `pbc/cpu/`, `pbc/gpu/`는 검증된 reference implementation으로
  읽고 활용하되, stress 구현을 위해 해당 경로에 조건문이나 stress 전용 함수를
  추가하지 않는다.
- 기존 PBC gradient goal의 완료 상태를 변경하거나 다시 열지 않는다.
- 공통화가 필요한 helper는 stress 경로에서 먼저 검증한다. 두 구현에서 실제로 같은
  contract가 확인된 뒤에만 별도 변경으로 공통 module 이동을 검토한다.
- 한 단계에서는 하나의 DF backend, sampling 범위와 물리 항만 확장한다.
- FD stress는 correctness oracle이고, production 및 MLIP 데이터 생성은 analytic stress를
  목표로 한다.

향후 source 구조는 다음을 기본안으로 한다. 이 문서를 작성하는 시점에는 빈 package를
미리 만들지 않는다.

```text
stress/
    Goal.md
    cpu/src/pyscf_pbc_dcdft_stress/
    gpu/src/gpu4pyscf_pbc_dcdft_stress/
    benchmark/development/
    benchmark/user/
    docs/
```

## 1. 1차 지원 범위

첫 release의 범위는 다음으로 제한한다.

- restricted closed-shell RHF/KRHF source density
- fixed-density DC-PBE 및 DC-r2SCAN target energy
- GTH pseudopotential
- Gamma GDF를 첫 analytic backend로 사용
- full symmetric stress tensor 6성분
- CPU analytic reference 이후 GPU 경로
- energy, nuclear gradient와 stress를 같은 calculation state에서 반환

다음 항목은 1차 release의 non-goal이다.

- open-shell/UHF 및 spin-polarized stress
- multigrid stress
- FFTDF stress
- all-electron PBC stress
- NPT integrator 또는 cell optimizer 자체 구현
- elastic constant와 2차 cell derivative
- 사용자 water64 계산을 development gate로 사용하는 것
- 수치 오차를 보정하는 fitted stress 또는 system-dependent correction

## 2. 정의와 convention lock

Cell matrix를 $h$, cell volume을 $\Omega=\det h$, 대칭 strain을
$\epsilon$이라 둔다. Cartesian position은 fractional coordinate $s_A$를 고정한 채

$$
h(\epsilon)=(I+\epsilon)h, \qquad R_A(\epsilon)=h(\epsilon)s_A
$$

로 변형한다. Stress는 tensile-positive convention으로

$$
\sigma_{\alpha\beta}
=\frac{1}{\Omega}\frac{\partial E}{\partial\epsilon_{\alpha\beta}},
\qquad
P=-\frac{1}{3}\operatorname{Tr}\sigma
$$

를 사용한다.

- 내부 canonical representation은 대칭 $3\times3$ tensor다.
- Voigt 출력 순서는 `[xx, yy, zz, yz, xz, xy]`로 고정한다.
- shear는 engineering strain과 tensor strain을 혼용하지 않는다. FD perturbation과
  analytic assembly가 사용하는 off-diagonal factor를 unit test로 고정한다.
- 기본 단위는 `Eh/Bohr^3`; report에는 GPa 변환값도 함께 기록한다.
- MLIP용 virial을 출력할 경우 $W=-\Omega\sigma$ convention과 부호를 metadata에 쓴다.
- k-point grid의 fractional reciprocal 좌표와 weight는 strain 동안 고정하고,
  Cartesian k-vector만 reciprocal lattice와 함께 변환한다.

## 3. Gradient 코드와의 관계

Stress Lagrangian은 기존 DC-DFT nuclear gradient와 다음 항목을 공유한다.

- source HF density와 target DFA state construction
- target Fock의 virtual-occupied block
- matrix-free HF response
- CPHF/Z-vector solver
- GDF/RSDF source, metric 및 auxiliary response의 수학적 구성
- PBE/r2SCAN XC density channels와 normalization

Z-vector는 perturbation 종류에 의존하지 않으므로 원자 displacement와 cell strain에
같은 converged Z를 사용할 수 있다. Analytic stress는 strain 6성분마다 CPHF를 다시
풀지 않고, Z-vector 1회와 6개의 explicit derivative contraction으로 구성한다.

그러나 다음 항목은 nuclear gradient에서 그대로 복사할 수 없는 stress 고유 항이다.

- lattice vector와 reciprocal lattice derivative
- cell volume 및 integration-weight derivative
- Gaussian lattice image와 basis-center Pulay response
- kinetic, overlap 및 GTH local/nonlocal PP의 homogeneous strain derivative
- Ewald/ion-ion cell derivative
- GDF/RSDF 2-center metric과 3-center integral의 cell derivative
- reciprocal mesh와 k-vector response
- XC grid coordinate 및 volume response

Stress package는 gradient package의 public state/response API만 사용한다. Private helper를
사용해야 할 경우 adapter 한 곳에 격리하고 upstream version 및 signature test를 둔다.

## 4. 작업 단계

### Stage S0: Workspace와 contract 고정

1. `stress/` 전용 CPU/GPU package, test, benchmark namespace를 만든다.
2. cell/strain convention, Voigt ordering, pressure와 virial 부호를 dataclass로 고정한다.
3. 기존 PBC source/target state를 read-only로 받는 adapter를 만든다.
4. stress option이 기존 energy/gradient state fingerprint를 오염시키지 않게 한다.
5. component timing과 host/device memory report schema를 정의한다.

Completion gates:

- zero strain에서 geometry, cell, k-point와 mesh가 변하지 않음
- tensor/Voigt round trip 및 shear factor test
- isotropic strain에서 $dE/d\eta=\Omega\operatorname{Tr}\sigma$ convention 일치
- stress package를 import해도 기존 gradient package에 monkey patch가 발생하지 않음

### Stage S1: Central-FD stress oracle

1. 대칭 strain 6성분에 대해 $+\delta/-\delta$ 중앙차분을 구현한다.
2. 각 strained cell에서 source HF SCF와 target DC-DFT energy를 다시 계산한다.
3. fractional atomic coordinates, fractional k-point grid와 k-point weight를 고정한다.
4. $+\delta/-\delta$ pair는 동일한 integer mesh를 사용한다. `ke_cutoff`에서 mesh를
   각각 다시 반올림해 생기는 discontinuity를 FD derivative에 섞지 않는다.
5. $\delta$, $\delta/2$, $\delta/4$ step convergence와 SCF tolerance convergence를
   report에 기록한다.
6. warm start는 timing option으로만 사용하고 FD 정의나 reference energy를 바꾸지 않는다.

FD는 중앙 energy를 포함하면 최대 13개 state, 이미 중앙 state가 있으면 12개의 추가
energy calculation을 요구한다. Production stress로 노출하지 않고 validation namespace에
둔다.

Completion gates:

- diagonal 및 shear 6성분 step-size plateau
- stress tensor antisymmetric residual `<= 1e-10 Eh/Bohr^3`
- rigid translation한 구조의 converged stress parity
- fixed-mesh와 cutoff-derived-mesh 결과를 구분해 report
- failed SCF 또는 mesh change를 stress 값으로 조용히 수용하지 않음

### Stage S2: Frozen-density explicit cell derivative

Orbital response를 넣기 전에 고정된 HF density에서 explicit derivative를 항별로 만든다.

1. ion-ion/Ewald contribution
2. overlap 및 kinetic contribution
3. GTH local/nonlocal pseudopotential contribution
4. target Hartree와 XC explicit contribution
5. cell volume, real-space grid와 reciprocal-vector contribution
6. GDF metric, 3-center integral 및 auxiliary-basis explicit contribution

각 항은 단독 FD와 비교할 수 있는 decomposition을 제공한다. 전체 합만 맞추기 위해 서로
다른 항의 오차가 상쇄되는 상태를 completion으로 인정하지 않는다.

Completion gates:

- small cubic/non-cubic cell에서 component FD parity
- isotropic 및 pure shear perturbation 모두 통과
- PBE Gamma/GDF frozen-density stress maximum error `<= 1e-6 Eh/Bohr^3`
- PP, one-electron, Hartree와 XC owner가 report와 실제 호출 경로에서 일치

### Stage S3: DC-DFT orbital response와 analytic CPU GDF

1. 기존 target Fock VO RHS와 source HF response로 Z-vector를 한 번 푼다.
2. strain-dependent source Fock와 overlap derivative를 Z와 수축한다.
3. explicit S2 stress와 non-variational response stress를 조립한다.
4. 6개 strain을 원자별 gradient tensor처럼 모두 보존하지 않고 component별로 즉시
   contraction한다.
5. nuclear gradient와 stress를 같은 state에서 요청할 때 HF, target Fock와 Z-vector를
   재사용한다.

Completion gates:

- analytic-FD maximum difference `<= 1e-6 Eh/Bohr^3` for PBE small systems
- Z-vector residual `<= 1e-8`
- analytic stress 계산 중 추가 HF SCF 없음
- stress 6성분을 위해 Z-vector를 한 번만 풂
- 기존 CPU GDF energy/gradient regression 무변경

### Stage S4: MGGA/r2SCAN grid response

1. density, density-gradient와 $\tau$ channel의 strain response를 구현한다.
2. integration coordinate, cell volume와 quadrature weight derivative를 포함한다.
3. uniform-grid mesh convergence와 egg-box/phase sensitivity를 분리 측정한다.
4. target XC mesh와 source HF/GDF integral mesh ownership을 report에 별도로 기록한다.

Completion gates:

- DC-r2SCAN analytic-FD maximum difference `<= 1e-5 Eh/Bohr^3`
- PBE 경로 무회귀
- mesh 증가에 따른 stress convergence 기록
- translation phase에 따른 energy/stress amplitude를 숨기지 않음

### Stage S5: RSDF CPU 확장

1. GDF에서 검증한 one-electron, PP, XC와 Z-vector assembly를 재사용한다.
2. RSDF short-range/long-range source, metric과 auxiliary cell derivative를 분리한다.
3. source omega, metric omega와 mesh를 report에 고정한다.
4. full derivative tensor를 보존하지 않고 6개 stress scalar/tensor component로 즉시
   contraction하는 bounded-memory route를 사용한다.

Completion gates:

- GDF/RSDF converged energy 및 stress 비교
- RSDF analytic-FD maximum difference `<= 1e-6 Eh/Bohr^3` for PBE
- range-separation parameter convergence
- 기존 RSDF gradient release tests 무변경

### Stage S6: Sampled k-point CPU

1. fractional k-point를 고정하고 reciprocal lattice derivative를 구현한다.
2. complex density/Fock contraction에서 실수부를 임의 cast하지 않는다.
3. $1/N_k$ normalization과 time-reversal pair를 primitive test로 고정한다.
4. Gamma limit와 sampled-k 결과의 연속성을 확인한다.

Completion gates:

- imaginary residual과 antisymmetric stress residual 명시
- sampled-k analytic-FD gate
- Gamma route 무회귀
- k-point weight와 reciprocal-coordinate convention metadata 고정

### Stage S7: GPU analytic stress

1. 검증된 CPU equations를 그대로 유지하고 contraction/backend만 GPU로 옮긴다.
2. source HF/target DFA/Z-vector가 기존 GPU DF state와 CDERI를 공유한다.
3. strain component별 derivative를 생성 즉시 stress tensor에 contraction한다.
4. memory estimator, bounded batching, OOM retry와 peak VRAM tracking을 구현한다.
5. CPU fallback은 explicit diagnostic option 외에는 허용하지 않는다.

Completion gates:

- CPU/GPU maximum difference `<= 1e-7 Eh/Bohr^3`
- PBE 및 r2SCAN, Gamma 및 sampled-k gate
- forced small-batch parity
- no silent CPU fallback, dropped component, NaN/Inf 또는 complex cast warning
- small-system에서 FD보다 end-to-end 시간이 짧음

### Stage S8: MLIP output와 release

1. energy, force, stress, virial, volume와 units를 한 report에 저장한다.
2. force는 `-gradient`, virial은 선택한 convention으로 한 번만 변환한다.
3. ASE/extxyz 등 출력 adapter는 core stress tensor와 분리한다.
4. user benchmark와 development benchmark를 별도 폴더에 유지한다.
5. NVT용 energy/force-only와 NPT용 energy/force/stress dataset을 구분한다.

Release gates:

- energy/force/stress가 동일한 source HF state와 numerical settings를 사용
- tensor, Voigt, GPa와 virial 변환 round-trip test
- analytic-FD, CPU/GPU와 k-point release matrix 통과
- wheel/import/example smoke test
- documentation에 sign, units, fixed-fractional-coordinate convention 명시

## 5. Benchmark 정책

Development system은 H2, H2O와 작은 water cluster의 cubic/non-cubic cell로 제한한다.
Water64는 사용자가 실행하는 scaling evidence이며 개발 중 자동 계산하지 않는다.

각 benchmark는 다음을 별도로 기록한다.

- source HF SCF time
- target DFA energy/Fock time
- Z-vector time과 iteration/residual
- explicit stress component time
- response stress time
- total analytic stress time
- FD pair별 SCF/energy time
- host RSS 및 sampled peak VRAM
- cell, volume, mesh, cutoff, k-point와 strain step
- stress tensor, pressure와 virial convention

Timing 비교는 독립 process에서 최소 3회 수행하고 median을 사용한다. FD 정확성 계산은
매 timing 반복마다 수행하지 않는다.

## 6. 작업 규칙

각 stage는 다음 순서를 따른다.

1. convention 또는 빠진 물리 항을 식과 component FD로 정의한다.
2. 작은 CPU reference를 먼저 구현한다.
3. 항별 analytic-FD parity를 확인한다.
4. 전체 stress를 조립한다.
5. 기존 energy/gradient 무회귀를 확인한다.
6. 그 후에만 다음 DF backend, k-point 또는 GPU로 확장한다.

정확성 gate를 성능 개선으로 완화하지 않는다. Mesh/cutoff 미수렴, SCF 미수렴과 egg-box는
analytic kernel 오차와 구분해 report한다. Stress 결과를 맞추기 위해 사후에 translation
sum, trace 또는 대칭성을 강제로 보정하지 않는다.

## 7. Progress Ledger

| Stage | State | Evidence | 다음 행동 |
| --- | --- | --- | --- |
| S0 workspace/convention | `complete` | 독립 CPU/GPU namespace, convention/strain/import-isolation tests | contract 유지 |
| S1 central-FD oracle | `complete` | CPU Gamma/GDF full tensor, diagonal/shear step, translation/mesh convergence 포함 12 tests | validation oracle 유지 |
| S2 frozen-density terms | `complete` | kinetic/XC/Ewald/GDF PP/J component parity, complete H2O PP matrix gate | decomposition 유지 |
| S3 CPU Gamma GDF analytic | `complete` | H2/H2O full analytic-FD error $8.46\times10^{-11}$/$5.11\times10^{-10}$ Eh/Bohr$^3$, no strained-cell production calls | 성능 최적화와 API 유지 |
| S4 r2SCAN grid response | `complete` | full analytic-FD $8.86\times10^{-11}$ Eh/Bohr$^3$; mesh 9/15/21 phase convergence | uniform-grid 수렴 metadata 유지 |
| S5 CPU RSDF | `complete` | PBE/r2SCAN H2 및 PBE H2O full analytic-FD; exact screened metric callback | parameter convergence 유지 |
| S6 sampled k-point | `complete` | PBE GDF/RSDF와 r2SCAN GDF full analytic-FD, complex pair/CDERI/PP/Madelung gates | fixed-scaled-k contract 유지 |
| S7 GPU analytic stress | `complete` | Gamma/sampled-k PBE/r2SCAN GDF/RSDF parity, low-memory gate, full GPU FD timing | production integral-owner contract 유지 |
| S8 MLIP/release | `complete` | same-state CPU/GPU workflow, one-Z reuse, JSON/extxyz conversion tests | release matrix 유지 |

## 8. 완료 정의

이 goal은 다음 조건을 모두 만족할 때 완료한다.

1. Restricted PBC DC-PBE 및 DC-r2SCAN stress가 GDF/RSDF에서 analytic하게 계산된다.
2. Gamma와 sampled k-point, CPU와 GPU가 명시된 정확성 gate를 통과한다.
3. 한 번의 HF state와 Z-vector로 energy, force와 stress를 함께 계산할 수 있다.
4. FD stress는 독립 oracle로 남고 production 경로에는 사용되지 않는다.
5. Stress 전용 코드는 `stress/`에 격리되고 기존 gradient package의 public behavior를
   변경하지 않는다.
6. MLIP/NPT 데이터에 필요한 stress, virial, volume, sign과 unit metadata가 완전하다.

## 9. 2026-07-21 S0/S1 implementation record

- `stress/cpu`와 `stress/gpu`를 기존 `grad/` 및 `pbc/` package와 분리했다. GPU
  namespace는 S7 이전에 계산 driver를 노출하지 않고 capability만 정직하게 보고한다.
- Tensile-positive stress, `[xx, yy, zz, yz, xz, xy]` Voigt ordering,
  engineering shear와 $W=-\Omega\sigma$ virial convention을 고정했다.
- PySCF row-vector lattice에 $a'=a(I+\epsilon)^T$를 적용하고 fractional atomic
  coordinate와 fractional reciprocal k-point를 보존한다.
- Production-default FD pair는 중앙 cell의 integer mesh를 $+\delta/-\delta$에 그대로
  사용한다. Cutoff-derived mesh 비교는 별도 option이며 plus/minus mesh를 각각 report한다.
- CPU S1 evaluator는 Gamma/GDF, restricted, uniform target integration만 허용한다.
  Strained cell마다 source HF SCF와 target DC-DFT energy를 새로 계산한다.
- Cubic H2/GTH-SZV/GDF/PBE, mesh `(9,9,9)`, $\delta=5\times10^{-4}$에서 full FD
  stress는 $\sigma_{xx}=\sigma_{yy}=1.36905844\times10^{-3}$,
  $\sigma_{zz}=-2.51148161\times10^{-4}$ Eh/Bohr$^3$였고 shear maximum은
  $7.11\times10^{-15}$ Eh/Bohr$^3$였다.
- Distorted H2의 마지막 두 step 차이는 $\sigma_{xx}$에서
  $4.18\times10^{-10}$, nonzero $\sigma_{xy}$에서 $3.38\times10^{-12}$
  Eh/Bohr$^3$였다.
- Rigid translation에 따른 $\sigma_{xx}$ 차이는 mesh 9, 15, 21에서 각각
  $9.15\times10^{-5}$, $1.35\times10^{-6}$, $1.62\times10^{-8}$
  Eh/Bohr$^3$로 감소했다. 이는 finite-grid phase error를 analytic stress 오차와
  구분하기 위한 baseline이다.
- Convention, strain, synthetic stress, non-convergence, real Gamma/GDF tensor,
  step convergence, translation과 dynamic-mesh reporting을 포함한 12개 test가 통과했다.
- Compact evidence는 `stress/validation/results/2026-07-21_s0_s1_fd.json`에 저장한다.

## 10. 2026-07-21 S2 partial implementation record

- 중앙 source-HF AO density를 그대로 유지하면서 target GDF energy를 kinetic, GTH PP,
  Coulomb, XC, Ewald로 분해하는 `GammaGDFFrozenDensityEvaluator`를 추가했다. Strain마다
  source HF SCF를 다시 돌리는 S1 oracle과 계산 의미를 분리했다.
- 고정 density component FD는 중앙 cell의 fractional atom coordinate와 integer mesh를
  보존하며 각 energy term의 strain derivative를 독립적으로 기록한다.
- PySCF 2.13 `rks_stress` dependency는 `upstream.py` 한 곳에 격리하고 version 및 function
  signature를 검사한다.
- Distorted H2/GTH-SZV/PBE/Gamma-GDF, mesh `(15,15,15)`, step
  $2.5\times10^{-4}$에서 analytic kinetic, XC, Ewald의 component-FD 최대 오차는 각각
  $1.89\times10^{-10}$, $5.31\times10^{-10}$, $9.96\times10^{-11}$
  Eh/Bohr$^3$다.
- PySCF의 현 PP stress primitive는 FFT representation이고 target hcore는 GDF PP를
  사용한다. 이를 GDF PP로 간주하면 같은 test에서 $2.92\times10^{-5}$
  Eh/Bohr$^3$ 차이가 생기므로 diagnostic으로만 노출하고 capability에는
  `gdf_pseudopotential`을 missing으로 유지한다.
- GDF PP와 Coulomb은 metric, 3-center source, reciprocal vector 및 lattice-image의
  homogeneous-strain response가 구현된 뒤에만 S2 완료로 바꾼다.

## 11. 2026-07-21 S3 development-reference record

- 기존 nuclear-gradient package의 context, target Fock, adjoint Z-vector와 HF response
  algebra는 `gradient_adapter.py` 한 곳을 통해 read-only로 재사용한다. Stress 전용
  조건문이나 method를 기존 `pbc/` package에 추가하지 않았다.
- GDF PP/J/K AO matrix만 strain pair로 재구성하는 matrix-level FD oracle을 추가했다.
  이 경로는 stress 중 source HF를 다시 실행하지 않으며 `production=false`다.
- Fixed-density explicit, target overlap/Pulay와 Z-vector response를 조립한 complete
  Gamma/GDF development stress가 distorted H2/GTH-DZVP에서 full strained-HF energy FD와
  최대 $1.62\times10^{-10}$ Eh/Bohr$^3$로 일치했다.
- 해당 test의 Z-vector real dimension은 18, residual은 $1.73\times10^{-18}$이며,
  orbital-response stress maximum은 $1.51\times10^{-4}$ Eh/Bohr$^3$로 0이 아니다.
- 중앙 energy 이후 semi-analytic reference는 1.56 s, full energy FD는 2.51 s였다.
  Semi-analytic 시간 중 1.28 s가 GDF PP/J/K matrix FD이므로 다음 병목과 analytic
  replacement owner가 명확하다.
- 이 결과는 Lagrangian 조립 gate 통과이지 S3 analytic 완료가 아니다. Metadata의
  `low_level_numerical_components`가 비워지기 전에는 production stress로 노출하지 않는다.
- GDF analytic replacement의 subterm 순서, matrix parity gate와 numerical-setting
  policy는 `docs/GDF_Analytic_Strain_Kernel_Plan.md`에 고정했다.

## 12. 2026-07-21 Gamma GDF analytic-kernel completion

- Auxiliary metric의 short-range image sum, reciprocal Fourier response, weighted Coulomb
  kernel과 $G=0$ volume response를 모두 analytic strain kernel로 교체했다.
- Three-center source의 AO-pair FT, auxiliary FT, reciprocal kernel, `vbar` 및 short-range
  double lattice sum을 analytic화했다. Short-range 항은 stress 전용 C fill callback이
  PySCF의 ME screening/OpenMP traversal 안에서 이미지 상대좌표를 즉시 수축하며, 원본
  PySCF module을 monkey patch하지 않는다.
- Complete pre-Cholesky source/metric에서 Cholesky response를 구성했고, fixed-density
  Gamma GDF $dJ/d\epsilon$과 fitted $dK/d\epsilon$을 analytic하게 얻는다.
- `exxdiv='ewald'`의 $mSDS$ 보정은 overlap response뿐 아니라 cell-dependent Madelung
  response도 포함한다. 3D Ewald real/self/reciprocal sum을 stress package 안에서 직접
  미분하며, Ewald 분할 parameter와 image/mesh는 중앙값에 고정한다.
- 실제 PySCF GDF provider matrix-FD와 비교한 최대 matrix 차이는 distorted H2에서
  $dJ$가 $1.55\times10^{-7}$ Ha, Ewald-corrected $dK$가
  $4.16\times10^{-7}$ Ha로 $10^{-6}$ Ha gate를 통과했다.
- GTH local-part1은 fake-nuclear three-center source와 representation-specific pseudo
  $G=0$ 항으로 analytic화했다. `_RSNucBuilder` strain FD와 최대 약
  $2.8\times10^{-8}$ Ha로 일치한다.
- GTH local-part2는 Gaussian-polynomial radial response와 AO strain을, nonlocal PP는
  reciprocal projector radial/angular response를 blockwise analytic하게 계산한다. Complete
  H2O PP matrix strain은 provider FD와 최대 약 $4.3\times10^{-8}$ Ha로 일치한다.
- H2/GTH-SZV/PBE full analytic stress는 full strained-HF FD와 최대
  $8.47\times10^{-11}$ Eh/Bohr$^3$, H2O/GTH-SZV/GTH-PBE는
  $5.10\times10^{-10}$ Eh/Bohr$^3$로 일치했다. 두 계산 모두 Z residual이 $10^{-8}$
  gate 아래이며 추가 source-HF SCF가 없다.
- 현재 `low_level_numerical_components`는 비어 있다. Canonical API는
  `gamma_gdf_analytic_stress`이고, 옛 `semi_analytic` 이름은 FD를 수행하지 않는 호환
  alias다. 이 production 표시는 restricted DC-PBE/Gamma/GDF 범위에만 적용한다.
- 42개 CPU test가 통과했다. Evidence는
  `stress/validation/results/2026-07-21_s3_gamma_gdf_analytic.json` 및
  `stress/validation/results/2026-07-21_s3_gamma_gdf_analytic_h2o.json`에 둔다.
- CPU reference는 correctness-first 구현이다. 최종 one-electron analytic 경로에서 H2는
  analytic 6.69 s 대 FD 1.32 s, H2O는 13.85 s 대 FD 2.79 s였으며, 대부분의 시간은
  image-resolved GDF PP와
  source/CDERI kernel이다. 이 수치를 속도 향상으로 제시하지 않는다.

## 13. 2026-07-21 S4 DC-r2SCAN completion

- PySCF 2.13 uniform-grid XC strain primitive의 MGGA branch가 density, density-gradient,
  grid-coordinate/volume 및 $\tau$ response를 포함하는지 source audit와 component FD로
  확인했다. PP/J/one-electron/Ewald는 이 upstream helper에 맡기지 않는다.
- Distorted H2의 fixed-density r2SCAN XC tensor는 component FD와 최대
  $5.58\times10^{-10}$ Eh/Bohr$^3$로 일치했다.
- Cubic H2 full DC-r2SCAN stress는 relaxed full FD와 최대
  $8.86\times10^{-11}$ Eh/Bohr$^3$, distorted H2는
  $2.20\times10^{-10}$ Eh/Bohr$^3$로 일치했다.
- Rigid shift `(0.23,0.17,0.11)` Bohr에 따른 stress 차이는 mesh 9, 15, 21에서
  각각 $1.05\times10^{-4}$, $2.18\times10^{-6}$,
  $1.28\times10^{-8}$ Eh/Bohr$^3$로 감소했다. 이는 uniform-grid phase error로 report하며
  stress를 사후 보정하지 않는다.
- Evidence는 `stress/validation/results/2026-07-21_s4_gamma_gdf_r2scan_analytic.json`,
  `2026-07-21_s4_gamma_gdf_r2scan_distorted_h2.json` 및
  `2026-07-21_s4_r2scan_mesh_phase.json`에 둔다.

## 14. 2026-07-21 S5 Gamma RSDF completion

- RSDF source와 metric이 서로 다른 `omega`와 compact mesh를 갖는 numerical contract를
  그대로 보존한다. Source는 `source_omega/source_mesh/precision_R`, metric은
  `metric_omega/metric_mesh/precision_j2c`를 사용한다.
- RSDF metric의 short-range 값은 일반 image sum과 동일시하지 않았다. PySCF
  `PBCsr2c`의 shell-pair screening/image list 안에서 값과
  $I^{ip1}_x(R_j-R_i+L)_y$를 함께 누적하는 stress 전용 C callback을 추가했다.
- Reconstructed CDERI와 실제 central HDF5 storage의 최대 차이는 H2에서
  $3.17\times10^{-10}$, H2O에서 $7.52\times10^{-10}$로 $10^{-7}$ gate 아래다.
- Full analytic-FD 최대 stress 차이는 H2 DC-PBE
  $5.90\times10^{-11}$, H2 DC-r2SCAN $5.56\times10^{-11}$, H2O DC-PBE
  $4.74\times10^{-10}$ Eh/Bohr$^3$다.
- H2에서 RSDF default와 GDF의 energy 차이는 $9.35\times10^{-7}$ Ha, stress 최대 차이는
  $2.41\times10^{-8}$ Eh/Bohr$^3$였다. 이는 각 backend의 finite numerical
  representation 차이이며 analytic-FD 오차와 구분한다.
- `(source,metric)` omega를 `(0.25,0.35)`, `(0.30,0.40)`, `(0.35,0.45)`로 바꾼
  convergence diagnostic에서 default stress 대비 최대 차이는 각각
  $1.91\times10^{-8}$, 0, $9.49\times10^{-8}$ Eh/Bohr$^3$였다. Default는 PySCF RSDF
  설정을 유지한다.
- S5 evidence는 `stress/validation/results/2026-07-21_s5_gamma_rsdf_*.json` 및
  `2026-07-21_s5_rsdf_range_separation.json`에 둔다. 45개 CPU test가 통과했다.

## 15. 2026-07-21 S6 sampled k-point CPU completion

- Strain 동안 fractional k-point와 weight를 고정하고 Cartesian k-vector와 momentum
  transfer $q=k_j-k_i$가 reciprocal lattice를 따라 변환되도록 했다.
- Long-range source는 $G+q$ momentum과 ket-image $e^{ik_j\cdot L}$를, short-range
  source는 $e^{-ik_i\cdot L_i+ik_j\cdot L_j}$를 사용한다. Stress 전용 complex C
  callback이 PySCF ME screening traversal 안에서 중앙값과 9개 strain 성분을 함께
  누적한다.
- GDF complex metric과 RSDF exact screened metric을 모두 구현했다. RSDF 2-center
  callback은 각 image에 $e^{iq\cdot L}$를 적용하며 PySCF 중앙 metric을 보존한다.
- Complex Cholesky response로 canonical CDERI를 만들고 reverse pair는 conjugate AO
  transpose로 복원한다. Reconstructed GDF/RSDF factors는 실제 central HDF5 storage와
  $10^{-7}$ gate 아래에서 일치한다.
- Sampled-k J/K에는 matrix construction의 $1/N_k$를, energy contraction에는 별도의
  $1/N_k$를 유지한다. 중간 complex tensor를 실수로 cast하지 않는다.
- GTH local-part1, local-part2와 nonlocal projector를 k-dependent Bloch AO/$G+k$ 경로로
  확장했다. `exxdiv='ewald'`는 PySCF가 사용하는 Monkhorst-Pack probe supercell의
  Madelung strain을 analytic하게 계산한다.
- Distorted H2/GTH-SZV, 2x1x1 k mesh에서 selected relaxed-FD 최대 stress 차이는
  DC-PBE GDF $3.62\times10^{-10}$, DC-PBE RSDF $1.91\times10^{-10}$,
  DC-r2SCAN GDF $3.71\times10^{-10}$ Eh/Bohr$^3$다.
- 63개 CPU test가 통과했다. Evidence는
  `stress/validation/results/2026-07-21_s6_kpoint_*.json`, 식과 phase/normalization
  설명은 `stress/docs/S6_Sampled_K_Analytic_Strain.md`에 둔다.
- Correctness-first CPU analytic 시간은 같은 작은 test에서 GDF 약 14.7 s, RSDF 약
  5.75 s였고 selected FD보다 느렸다. S6 completion은 정확성/API 범위이며 속도 향상을
  주장하지 않는다.

## 16. 2026-07-21 S7 GPU hybrid-primitive progress

- Source HF, target Fock, HF AX/Z-vector, PBE/r2SCAN XC stress, complex Cholesky response,
  sampled-k J/K와 최종 DC Lagrangian contraction을 CuPy/GPU 경로에 연결했다.
- PySCF/GPU4PySCF가 homogeneous-strain 2c/3c 및 GTH PP GPU integral kernel을 제공하지
  않으므로, 해당 primitive만 host analytic 경로에서 만든다. CPU SCF, CPU target energy,
  CPU Z-vector 또는 완성된 CPU stress로 되돌아가는 silent fallback은 없다.
- Source/metric derivative는 한 strain component씩 device에 올려 $dC$, $dJ$, $dK$에
  즉시 수축한다. Off-diagonal pair derivative는 사용 직후 해제하며 reverse pair는
  conjugate AO transpose로 복원한다.
- Canonical k-pair primitive는 private `cell/auxcell`을 사용하는 최대 4개의 bounded host
  worker로 병렬화했다. 2x1x1 H2에서 GDF host DF primitive는 11.50 s에서 4.40 s,
  stress 전체는 14.72 s에서 7.89 s로 줄었다. RSDF는 2.59 s에서 1.46 s,
  전체는 5.70 s에서 4.96 s로 줄었다.
- Sampled-k CPU/GPU 최대 stress 차이는 PBE GDF $2.88\times10^{-12}$, PBE RSDF
  $1.16\times10^{-12}$, r2SCAN GDF $2.88\times10^{-12}$ Eh/Bohr$^3$다.
- Host integral engine/device response 경계를 production contract로 명시적으로 채택했다.
  이 표시는 모든 연산이 GPU라는 뜻이 아니며 owner, worker, memory estimate와 실제 peak를
  결과에 계속 기록한다. 세부 owner와 evidence는
  `docs/S7_GPU_Hybrid_Analytic_Stress.md`에 둔다.

## 17. 2026-07-21 S7/S8 release completion

- Gamma와 sampled-k의 PBE/r2SCAN, GDF/RSDF CPU-GPU parity matrix를 채웠다. 최대 차이는
  sampled-k GDF/r2SCAN의 약 $2.88\times10^{-12}$ Eh/Bohr$^3$다.
- Diagonal $dC$ cache가 지정 byte limit을 넘으면 재계산 route로 자동 전환한다. Cache를
  강제로 0 byte로 둔 sampled-k GDF/PBE stress는 CPU와 $2.88\times10^{-12}$
  Eh/Bohr$^3$로 일치했고 시간 증가는 약 0.04 s였다.
- Allocation 전 DF memory estimator가 central factor, component response, J/K workspace와
  현재 free VRAM을 비교한다. Cache를 꺼도 허용 memory fraction을 넘으면 CuPy OOM 전에
  필요한 GiB와 가능한 GiB를 포함한 diagnostic을 낸다.
- Gamma GDF/PBE 3회 timing median은 source state 1.38 s, analytic stress 7.20 s,
  합계 8.58 s, full 6-component GPU energy FD 17.93 s였다. Analytic-FD 차이는
  $6.45\times10^{-11}$ Eh/Bohr$^3$다.
- CPU/GPU same-state workflow가 energy, force, stress, virial과 state fingerprint를 한
  record로 만든다. Force는 gradient의 부호를 한 번만 바꾸고 virial은
  $W=-\Omega\sigma$로 한 번만 계산한다.
- Gradient에서 구한 Z-vector를 stress에 재사용한다. JSON/extxyz writer와 volume/sign/unit
  validation test를 추가했다. 세부 형식은 `docs/S8_MLIP_Output.md`에 둔다.

## 18. 2026-07-21 S7 host-memory repair

- Water20/SZV/500 Ha GDF stress가 host primitive 단계에서 RSS 약 474 GiB, swap 약
  30 GiB까지 증가한 뒤 report 없이 종료되는 현상을 진단했다.
- 원인은 full reciprocal AO-pair strain `(3,3,nG,nao,nao)` materialization이며 해당
  입력의 이론적 배열 크기는 약 26.4 TiB다.
- Long-range source/metric을 G-block direct contraction으로 바꾸고 production source는
  audit-only decomposition 사본을 보존하지 않는다. 기본 추정 working set은 8 GiB다.
- GTH local-part2의 atom-resolved full-mesh 배열과 local/nonlocal PP의 full structure
  factor 배열도 blockwise contraction으로 교체했다.
- `stress_host_reciprocal_block_size`를 tuning option으로 제공하되 mesh, omega, precision과
  analytic 식은 바꾸지 않는다.
- Progress에 target Fock/Z, host PP, host DF, device response와 XC assembly substage를
  기록한다.
- CPU stress 67개, GPU response 1개, queue 4개 test와 forced-block/compact parity가
  통과했다. Water20 대형 timing/RSS gate는 development queue로 분리한다.

## 19. 2026-07-22 S7 DF/adjoint response optimization

- GDF/RSDF source와 metric strain derivative의 retained layout을 ordered Cartesian
  9성분에서 symmetric Voigt 6성분으로 바꿨다. 수학적 projection은 CPU stress와 같은
  convention을 사용하며 derivative array memory는 33.3% 줄어든다.
- 각 Voigt 성분과 canonical k-pair에서 CDERI derivative를 한 번만 만들고 같은 $dC$를
  auxiliary Coulomb response, $dJ$와 $dK$에 모두 사용한다. Factor derivative 평가 수는
  $6N_{pair}$로 고정되며 diagonal cache는 필요 없다.
- Factorization을 마친 host pair는 즉시 해제해 raw 9성분과 compact 6성분이 build 종료까지
  함께 남지 않게 했다. Report는 transient raw build payload와 retained Voigt payload를
  구분한다.
- Strain 6성분별 overlap-response density에 HF AX를 적용하던 경로를 self-adjoint HF
  response identity로 재배열했다. Z transition density에 AX를 한 번 적용하고 여섯 source는
  trace contraction으로 처리한다. 이는 근사나 screening threshold 변경이 아니다.
- Synthetic sampled-k J/K six-component parity와 explicit/adjoint response test가 통과했다.
  실제 Gamma PBE stress의 CPU-GPU 차이는 GDF $2.31\times10^{-12}$, RSDF
  $5.04\times10^{-13}$ Eh/Bohr$^3$다.
- 이번 end-to-end timing은 별도 장시간 GPU/CPU benchmark가 동시에 실행 중이어서 release
  성능 수치로 채택하지 않는다. 독립 process 3회 median gate는 외부 계산 종료 후 수행한다.

## 20. 2026-07-22 S7 host Fourier/pair-pipeline optimization

- Long-range source가 reciprocal block마다 translated cell, concatenated molecule, shell
  slice와 lattice phase를 다시 만들던 중복을 제거했다. Pair별 immutable image context를
  한 번 만들고 모든 G block에서 재사용한다. 작은 blocked source에서 setup reuse만의
  개선은 약 1.10배였고 tensor는 bitwise 동일했다.
- Fourier overlap과 center-i/center-j derivative를 stress 전용 7-component C callback으로
  합쳤다. Coordinate moment를 포함한 integral driver 호출은 image/block당 4회에서 2회로
  줄었다. 동일 process에서 long-range source는 3.53 s에서 median 1.97 s로 1.79배
  빨라졌고 derivative 최대 차이는 $1.11\times10^{-16}$이다.
- PP local-part1의 source-vbar와 pseudo-G0 항이 같은 G=0 overlap strain을 공유하게 했다.
  Numerical parameter와 PP representation은 바꾸지 않았다.
- Sampled-k raw source/metric pair를 모두 host dict에 쌓던 경로를 bounded producer-consumer로
  바꿨다. 최대 `host_workers`개의 raw pair만 대기하며, 완료 pair는 즉시 Voigt 압축과 GPU
  factorization 후 해제한다.
- Extension runner는 binary 존재 여부뿐 아니라 C source mtime과 fused ABI symbol을 검사해
  stale `_sr3c_strain`을 자동 재빌드한다.
- Gamma 및 2x1x1 sampled-k PBE GDF/RSDF CPU-GPU parity가 모두 유지됐다. Sampled-k 최대
  차이는 GDF $2.88\times10^{-12}$, RSDF $1.16\times10^{-12}$ Eh/Bohr$^3$다.

## 21. 2026-07-22 Water8 performance/accuracy gate

- Water8/SZV-GTH/r2SCAN/Gamma/400 Ha에서 GDF와 RSDF stress-only 및 xx/xy selected FD를
  development scope로 측정했다. 두 case 모두 정상 완료했고 최대 analytic-FD 차이는 각각
  $6.88\times10^{-11}$, $6.90\times10^{-11}$ Eh/Bohr$^3$다.
- Analytic stress는 GDF 2226.2 s, RSDF 1705.9 s였다. 공통 host PP primitive가 각각
  1565.8 s와 1575.8 s를 사용해 주 병목이며, GDF host DF 651.1 s, RSDF host DF 121.3 s가
  다음 병목이다. Device DF는 두 route 모두 0.07 s 미만이다.
- Host peak RSS는 약 12.6 GiB, peak CuPy pool reserved는 약 1.37 GiB로 OOM 없이 통과했다.
- 정확성/memory gate는 PASS이나 performance gate는 미완료다. 다음 작업은 numerical
  parameter를 바꾸지 않고 공통 PP strain primitive와 GDF reciprocal primitive를
  최적화하는 것이다. 세부 evidence는 `docs/S7_Water8_Evaluation_2026-07-22.md`에 둔다.

## 22. 2026-07-22 experimental contracted GPU response

- GTH-SCAN PP는 source/adjoint density가 공유하는 AO/grid strain work를 한 번만 수행하고,
  PP에 필요하지 않은 GGA/MGGA density-gradient와 $\tau$ channel을 만들지 않는 fused
  GPU contraction을 추가했다.
- DF는 2c/3c strain tensor 전체를 만들지 않고, target Coulomb과 adjoint RHF에 필요한 두
  fixed-density scalar만 6개 symmetric strain에서 centered derivative로 계산하는 opt-in
  route를 추가했다. 12개 displaced provider에서 J/K만 만들며 source HF, target DFA와
  Z-vector는 재계산하지 않는다.
- H2에서 기존 analytic DF 대비 stress 최대 차이는 GDF $2.13\times10^{-13}$, RSDF
  $6.85\times10^{-14}$ Eh/Bohr$^3$였다.
- Water8/SZV-GTH/r2SCAN/400 Ha에서 최종 stress 시간은 GDF 25.09 s, RSDF 23.72 s로,
  최초 analytic baseline 대비 각각 88.73배와 71.92배 빨라졌다. Selected xx/xy FD 최대
  오차는 각각 $9.33\times10^{-11}$, $6.74\times10^{-11}$ Eh/Bohr$^3$다.
- Host peak RSS는 GDF 4.77 GiB, RSDF 4.80 GiB로 baseline 약 12.6 GiB보다 감소했다.
  Process-local peak CuPy used는 각각 약 0.70 GiB와 0.81 GiB다. Global VRAM peak는 동시에 실행된 다른
  process를 포함하므로 단독 peak로 해석하지 않는다.
- PP route의 최초 GTH-HF H2 불일치는 `mesh=[9,9,9]`의 local-potential grid
  미수렴으로 판명됐다. 공통 `ke_cutoff_ha=400`에서 H2 exact/contracted stress 차이는
  최대 $2.9\times10^{-13}$ Eh/Bohr$^3$다. DF/PP fast route 모두 낮은 수준의 cell
  displacement를 포함하므로 기본 analytic 경로를 교체하지 않으며 explicit opt-in,
  `production=false`로 유지한다.
- 세부 구현, raw evidence와 제한은
  `docs/S7_GPU_Contracted_Response_2026-07-22.md`에 둔다.
- 후속 PP 최적화는 source/adjoint nonlocal projector를 한 projector에 batch 수축하고,
  비대칭 9성분 중앙차분을 symmetric Voigt 6성분으로 줄였다. 3D GTH local reciprocal
  derivative는 $dG^2/d\epsilon_{xy}=-2G_xG_y$ chain rule로 analytic하게 계산한다.
  동일 원소 radial projector와 strain-invariant structure factor는 재사용한다.

## 23. 2026-07-22 dedicated GPU timing gate

- 다른 CUDA process가 없는 RTX PRO 6000 Blackwell Server Edition에서 Water8
  GDF/RSDF를 각각 3회 반복했다. FD는 끄고 source state와 analytic stress timing만
  측정했다.
- GDF stress median은 16.18 s, 범위는 16.09--16.19 s다. Source HF+target을 포함한
  workflow median은 18.18 s다.
- RSDF stress median은 15.70 s, 범위는 15.64--15.78 s다. Workflow median은
  17.70 s다.
- PP median은 GDF 13.56 s, RSDF 13.44 s이며 nonlocal projector가 각각 7.87 s와
  7.78 s로 남은 가장 큰 단일 병목이다. DF median은 GDF 2.31 s, RSDF 1.94 s다.
- Host RSS median은 GDF 4.80 GiB, RSDF 4.80 GiB다. CuPy pool used median은 두
  backend 모두 0.62 GiB다.
- Raw evidence는
  `benchmark/development/results/water8_gpu_dedicated_timing_2026-07-22`에 둔다.

## 24. 2026-07-22 sampled-k contracted GPU gate

- Water8/SZV-GTH/r2SCAN/400 Ha, 2x1x1 k-mesh에서 GDF/RSDF contracted PP/DF 경로를
  검증했다.
- GDF analytic stress는 52.28 s, source state는 6.61 s였다. Selected xx FD 최대
  차이는 $6.91\times10^{-11}$ Eh/Bohr$^3$다.
- RSDF analytic stress는 52.61 s, source state는 6.68 s였다. Selected xx FD 최대
  차이는 $6.58\times10^{-11}$ Eh/Bohr$^3$다.
- Sampled-k AO/XC는 GPU4PySCF `krks_stress`, 공통 GTH local/nonlocal primitive는
  `rks_stress` namespace를 사용하도록 owner를 분리했다. Complex density를 실수로
  cast하지 않는다.
- 두 backend 모두 tolerance $10^{-8}$ Eh/Bohr$^3$를 통과했다. Raw evidence는
  `benchmark/development/results/water8_gpu_sampled_k_validation_2026-07-22`에 둔다.

## 25. 2026-07-22 O1 profiler와 O2 PP phase-cache optimization

- PP/DF build 및 contraction count, transfer payload, explicit workspace와 benchmark
  stage별 RSS/VRAM peak를 report에 추가했다. Opt-in CUDA event profiler는 Python wall과
  current-stream interval을 분리하며 Z iteration/wall/residual과 target-XC timing도 노출한다.
- Nonlocal PP의 strain-invariant structure factor를 device에 한 번 유지하고, 원소별
  projector template을 displacement/k-point당 한 번만 전송하도록 바꿨다. 원자마다 full
  `(nprojector,nG)` workspace를 host에서 다시 보내지 않는다.
- Water8 Gamma/GDF 3회 median에서 stress는 14.81 s, nonlocal PP는 5.02 s다. 변경 전
  단일 paired run 대비 각각 16.1%, 35.9% 감소했고 명시적 PP H2D payload는 79.2%
  감소했다. Selected xx/xy FD 오차는 $9.37\times10^{-11}$ Eh/Bohr$^3$다.
- Water8 2x1x1에서 GDF/RSDF stress는 이전 52.28/52.61 s에서 35.73/36.46 s로
  감소했다. Selected xx FD 오차는 각각 $6.92\times10^{-11}$,
  $6.66\times10^{-11}$ Eh/Bohr$^3$다. VRAM/RSS 증가는 모두 10% 이하다.
- Density-axis batched local PP 실험은 호출 수를 줄였지만 local density wall을 72.9%,
  total stress를 7.5% 늘려 폐기했다. 원래 per-density GEMM 경로를 복원했다.

## 26. 2026-07-22 O4 analytic direct-contracted DF

- `host_analytic_direct_contracted` route를 추가했다. Analytic source/metric derivative를
  canonical k-pair별로 만들고 six directional Cholesky response를 target-J와 adjoint-RHF
  scalar에 즉시 수축한다.
- Pair 처리가 끝나면 source/metric derivative, differentiated CDERI와 central factor를
  폐기한다. Retained pair derivative, central factor와 six AO J/K response bytes는 0이다.
- H2 Gamma/k211 GDF/RSDF 네 case에서 matrix-producing exact route와 최대 차이는
  $1.0\times10^{-18}$ Eh/Bohr$^3$ 이하고, six-component FD 최대 오차는 약
  $8.8\times10^{-11}$ Eh/Bohr$^3$다.
- Host analytic PP와 조합한 이 gate는 numerical component가 없고 `production=true`다.
  Water8 memory/performance와 host builder의 pair 내부 directional streaming은 후속 gate다.
- Water8 Gamma/RSDF에서 selected `xx` FD 오차는
  $6.93\times10^{-11}$ Eh/Bohr$^3$로 통과했고, retained pair derivative, central
  factor와 six AO J/K response는 모두 0 byte다. 다만 총 129.64 s 중 117.13 s가 DF이며,
  그중 116.58 s가 host source primitive 생성이다. 따라서 이 route는 현재 memory gate를
  닫았지만 performance gate는 닫지 못했다.
- 대응 GDF Water8 route도 selected `xx` FD 오차
  $7.12\times10^{-11}$ Eh/Bohr$^3$로 통과했다. Stress/DF/host-source 시간은
  653.41/640.84/639.02 s이고 peak RSS는 12.67 GiB다. RSDF의
  129.64/117.13/116.58 s와 9.03 GiB보다 훨씬 크므로, pair 내부 GDF source streaming을
  우선하고 같은 구조를 RSDF에 확장한다.
- Direct route의 long-range source는 Cartesian 9성분을 만든 뒤 투영하지 않고 symmetric
  Voigt 6성분을 block 내부에서 직접 누적하도록 바꿨다. Reciprocal block당 source matrix
  product는 27회에서 18회로 줄었다. Water8 GDF/RSDF source 시간은 각각 7.8%/7.6%,
  retained host source payload는 29.4% 감소했고 FD gate를 통과했다. 실제 peak RSS는 거의
  변하지 않아, 다음 memory target은 8 GiB reciprocal block workspace다.

## 27. 2026-07-22 O4 bounded reciprocal DF memory profile

- 기존 `stress_host_reciprocal_block_size`가 host PP와 DF를 함께 바꾸던 경계를 분리했다.
  새 `stress_df_host_reciprocal_block_size`는 DF source/metric에만 적용되며, 생략 시 기존
  global 값을 상속한다.
- Water8 DF block4096에서 GDF/RSDF peak RSS는 각각 52.8%/33.8% 감소했고 raw
  target-J parity는 $6.75\times10^{-14}$/$1.01\times10^{-13}$ Eh다. RSDF raw
  adjoint-RHF parity는 $6.66\times10^{-15}$ Eh이며 selected FD도 통과했다.
- 대신 total stress wall은 GDF 19.4%, RSDF 15.3% 증가했다. 따라서 block4096은
  memory/OOM opt-in으로 유지하고 speed default는 automatic block을 사용한다.
- GDF block8192는 RSS 31.5%를 줄였지만 wall이 30.8% 늘어 절충 기본값에서 제외했다.
  Block size와 CPU wall은 비단조적이므로 메모리 크기만으로 자동 선택하지 않는다.
- Fast PP numerical route의 독립 run 변동과 DF parity를 분리하기 위해 report에 raw
  six-component target-Coulomb과 adjoint-RHF contracted scalar를 추가했다.

## 28. 2026-07-23 device-contracted analytic nonlocal PP

- GPU4PySCF에 없는 periodic `ft_ao_ip1`을 기다리지 않고 Cartesian Gaussian의
  $(r_y-R_y)\phi_l=\phi_{l+1}$ recurrence를 사용했다. 모든 shell의 angular momentum을
  하나 올린 내부 cell을 GPU FT-AO로 평가해 exact first moment를 얻는다.
- 중앙 AO와 first moment, structure factor, six symmetric Voigt projector overlap 및
  source/adjoint density trace는 GPU에서 계산한다. 작은 원소별 radial/angular projector
  value-gradient template만 host가 소유하며 metadata에 명시한다.
- Full PP derivative AO matrix, strained cell, 추가 SCF, target DFA rebuild와 추가 Z solve는
  모두 0이다. 새 explicit opt-in backend는
  `gpu4pyscf_native_contracted_analytic_nonlocal_device`다.
- Random two-density Gamma, complex `kmesh=[2,1,1]`, DZVP d-polarization의 nonlocal PP가
  같은 revision의 host analytic matrix와 설정 tolerance 내 일치했다.
- Water1 GDF/RSDF six-component FD 오차는 각각 $6.32\times10^{-12}$,
  $6.31\times10^{-12}$ Eh/Bohr$^3$다. Water8 selected FD 오차는 각각
  $6.88\times10^{-11}$, $6.90\times10^{-11}$ Eh/Bohr$^3$다.
- Water8에서 host analytic matrix 대비 PP wall은 GDF 546.07 s에서 12.05 s,
  RSDF 565.57 s에서 12.06 s로 줄었다. Numerical fast route 대비 total stress 차이는
  0.5% 이하이고 sampled VRAM delta는 약 36% 작다.
- 두 Water8 route 모두 numerical component가 없고 `production=true`다. 다만 default
  승격에는 full sampled-k end-to-end FD, profiler-off 3회 median과 Water20/64 user
  scaling evidence가 더 필요하다.
- 같은 원소/angular channel의 원자를 GPU batch로 묶고 AO response를 block/Voigt당 한
  번만 계산했다. Water8 nonlocal-only wall은 6.42 s에서 1.96 s로 줄었고, 최종 RSDF
  PP/stress는 6.37/116.49 s, selected FD 오차는 $6.90\times10^{-11}$
  Eh/Bohr$^3$다.

## 29. 2026-07-23 exact GPU FT source와 package audit

- Reverse-Cholesky source adjoint를 GPU4PySCF의 blocked FT strain kernel에 직접
  수축하는 exact GDF/RSDF owner를 추가했다. Full AO Fourier derivative와 six projected
  source matrix를 만들지 않는다.
- H2/Water8, Gamma/`2x1x1`, GDF/RSDF selected FD가 모두 통과했고 H2
  `DZVP-GTH` 네 조합의 최대 오차는 $2.95\times10^{-11}$ Eh/Bohr$^3$다.
- Water8 Gamma profiler-off median stress/RSS는 RSDF 8.057 s/3.135 GiB, native GDF
  12.209 s/3.569 GiB다. 직전 bounded-forward 경로보다 stress와 RSS가 모두 크게
  감소했다.
- CPU/GPU wheel을 clean temporary target에 설치해 public package와 C extension import를
  검증했다. `_sr3c_strain`은 PySCF `libcgto`/`libcint`를 직접 link하고 wheel-relative
  RPATH를 사용하므로 runner의 사전 preload에 의존하지 않는다.
- MLIP JSON/extxyz에 stress production 상태와 PP/DF/source-HF/Z owner를 보존하고,
  두 wheel에 Apache/PySCF notice를 포함했다. GPU wheel은 base
  `gpu4pyscf-pbc-dcdft` dependency를 직접 선언한다.
- 최종 revision에서 CPU 73개, GPU 21개, benchmark 10개 test가 통과했다. 상세 audit는
  `docs/Release_Audit_2026-07-23.md`에 둔다. 수정된 Water20 GDF/RSDF user scaling은
  61.52/52.51 s stress와 5.43/5.20 GiB RSS로 validator를 통과해 마지막 gate를 닫았다.
- 3D local GTH reciprocal value/slope, FFT frequency ordering과 atom structure factor를
  GPU에서 analytic하게 만들도록 바꿨다. Water8 local setup은 1.315 s에서 0.125 s로
  줄었고 PySCF derivative 및 full FD gate를 통과했다.
- 식, ownership, 제한과 raw result 경로는
  `docs/S7_GPU_Analytic_Nonlocal_PP_2026-07-23.md`에 기록한다.
