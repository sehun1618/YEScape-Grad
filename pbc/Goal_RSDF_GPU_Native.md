# RSDF GPU-Native Continuation Goal

## 0. 문서의 지위

이 문서는 restricted closed-shell PBC DC-DFT의 RSDF GPU 경로에서 남아 있는
correctness hardening과 explicit host integral boundary 축소만을 다룬다.

- 이전 GDF/RSDF maintenance 단계와 그 completion evidence는
  `Goal_GDF_RSDF_Continuation.md`에 고정한다.
- 기준 source는 이 Git 저장소의 `pbc/cpu/src/`와 `pbc/gpu/src/`다.
- 현재 source-contracted, bounded-memory RSDF response contract를 유지한다.
- 한 번에 하나의 측정된 병목 또는 correctness defect만 수정한다.
- 각 물리 변경과 성능 변경은 분리된 commit 또는 PR로 남긴다.
- Codex는 Water64 계산을 시작하거나 재실행하지 않는다. 사용자가 생성한 Water64
  결과는 read-only scaling evidence로만 분석한다.

## 1. 현재 기준선

현재 RSDF 경로는 다음 기능을 지원한다.

- RHF/KRHF source density
- PBE, SCAN 및 r2SCAN target DFA
- Gamma 및 sampled k-point
- CPU/GPU energy와 fixed-cell analytic nuclear gradient
- independent source/metric omega 및 mesh
- source, metric 및 auxiliary-basis response
- matrix-free preconditioned Z-vector
- full `dCDERI`, `dJ`, `dK`를 보존하지 않는 source-contracted component stream
- bounded component batching과 transactional OOM rebatch

현재 GPU route는 완전한 GPU-native RSDF가 아니다. 다음 연산은 명시적인 host
boundary다.

- central RSDF two-/three-center periodic integral generation
- three-center source nuclear-derivative primitive generation
- two-center metric nuclear-derivative primitive generation
- 일부 GTH H1 primitive generation
- PySCF Cell/AuxCell metadata와 final host reporting

이 boundary는 silent CPU DC-DFT fallback이 아니라 현재 GPU4PySCF에 필요한 public
RSDF derivative primitive가 없기 때문에 존재한다.

## 2. 즉시 해결할 correctness defect

Sampled-k GPU RSDF의 Cholesky metric response에서 실수형 factor와 복소수
metric derivative가 `cupyx.scipy.linalg.solve_triangular`에 함께 전달된다. CuPy가
우변을 실수형으로 변환하면 다음 경고와 함께 허수부가 버려진다.

```text
ComplexWarning: Casting complex values to real discards the imaginary part
```

현재 `2 x 1 x 1` r2SCAN/RSDF test에서 버려진 허수부의 최대 절대값은
`5.69e-15`이고 CPU/GPU gradient 차이는 `8.90e-11 Ha/Bohr`였다. 현재 test 결과에는
영향이 작지만, 일반 momentum-transfer block에서는 허수부가 물리적으로 유의미할 수
있으므로 경고를 무시하거나 `.real`로 제거하지 않는다.

## 3. 작업 순서

### Stage N0: Sampled-k complex dtype safety

1. `CDERIFactorization.differentiate()`의 factor, source derivative 및 metric
   derivative를 공통 dtype으로 승격한다.
2. complex momentum-transfer block에서는 Cholesky 및 triangular solve 전체가
   `complex128` contract를 유지하도록 한다.
3. 최종 observable을 만들기 전에는 허수부를 제거하지 않는다.
4. `ComplexWarning`을 error로 승격한 sampled-k RSDF regression을 추가한다.
5. Gamma real-valued fast path와 central factorization을 회귀 검증한다.

Completion gates:

- sampled-k RSDF test에서 `ComplexWarning`이 발생하지 않음
- CPU/GPU energy difference `<= 1e-8 Ha`
- CPU/GPU gradient maximum difference `<= 1e-7 Ha/Bohr`
- Gamma와 sampled-k PBE/r2SCAN parity
- RSDF 및 MGGA 관련 test 통과

### Stage N1: Benchmark lifecycle and baseline integrity

1. benchmark 시작, 완료, 실패 및 외부 중단 상태를 구분한다.
2. 실행 process가 없는데 `progress.json`이 `running`으로 남은 case를 다음 queue
   진입 시 `interrupted`로 판정할 수 있게 한다.
3. 기존 실패 `report.json`과 새 실행 `progress.json`이 같은 output directory에서
   섞이지 않도록 run identity 또는 start timestamp를 기록한다.
4. Water8과 제한된 Water20에서 RSDF host/device stage timing을 다시 고정한다.
5. source/metric omega, mesh, transferred bytes, host RSS와 sampled VRAM peak를 report에
   유지한다.

Water64 GDF/RSDF 결과는 사용자가 제공한 경우에만 읽는다. Water64 실패를 small-system
development gate로 바꾸지 않는다.

### Stage N2: GPU-native source derivative primitives

Profile상 가장 큰 host boundary인 RSDF three-center source derivative를 먼저 다룬다.

1. CPU source primitive의 exact short-range/long-range decomposition과 AO/auxiliary
   center ownership을 equation-level reference로 고정한다.
2. short-range three-center derivative를 bounded GPU batches로 구현한다.
3. long-range reciprocal derivative를 source-specific `omega`와 `mesh`로 구현한다.
4. Gamma와 sampled-k canonical pair orientation 및 conjugation contract를 보존한다.
5. primitive를 host에 materialize하지 않고 기존 source-contracted component stream으로
   바로 전달한다.
6. allocation 실패는 같은 interval을 축소 재시도하며 CPU route로 전환하지 않는다.

Completion gates:

- primitive-level CPU/GPU maximum difference `<= 1e-9`
- source-contracted target-J 및 Z-bilinear parity `<= 1e-9 Ha/Bohr`
- Gamma와 sampled-k full-gradient parity
- forced-small-batch 및 transactional OOM test
- host source-derivative timing과 transferred bytes의 감소 기록

### Stage N3: GPU-native metric derivative primitives

Source derivative completion 후 two-center metric derivative를 다룬다.

1. CPU metric의 exact image list, screening 및 source-independent `metric_omega`와
   `metric_mesh` contract를 보존한다.
2. short-range two-center metric derivative를 GPU에서 생성한다.
3. long-range reciprocal metric derivative와 Hermitian momentum-transfer orientation을
   GPU에서 생성한다.
4. N0의 complex dtype-safe Cholesky response와 직접 연결한다.
5. full metric derivative tensor의 장기 보존을 피하고 component stream에 전달한다.

Completion gates:

- metric value/derivative primitive CPU/GPU parity `<= 1e-9`
- Cholesky/factor-response parity `<= 1e-9`
- Gamma 및 sampled-k full-gradient parity
- independent source/metric omega와 mesh 무회귀
- host metric-derivative timing과 transferred bytes의 감소 기록

### Stage N4: Conditional central CDERI and H1 work

N2와 N3 완료 후 profile에서 남은 exclusive stage 하나만 선택한다.

후보는 다음과 같다.

- central RSDF two-/three-center integral 및 CDERI setup
- remaining GTH H1 primitive generation
- target XC/grid evaluation
- host-device transfer, synchronization 또는 context rebuild

다음 규칙을 적용한다.

- end-to-end profile 없이 포팅을 시작하지 않는다.
- 같은 가설을 두 번 시험해 end-to-end 개선이 5% 미만이면 중단한다.
- public PySCF/GPU4PySCF numerical contract를 바꾸는 system-specific approximation을
  도입하지 않는다.

### Stage N5: RSDF GPU-native release gate

- CPU/GPU energy difference `<= 1e-8 Ha`
- CPU/GPU gradient maximum difference `<= 1e-7 Ha/Bohr`
- selected analytic-FD maximum difference `<= 1e-5 Ha/Bohr`
- Z-vector residual `<= 1e-8`
- Gamma와 sampled-k PBE, SCAN 및 r2SCAN regression
- warning-as-error complex dtype test
- bounded-memory, forced batch-1 및 transactional OOM parity
- package import, wheel 및 public example smoke
- no silent CPU fallback and no dropped component/block

Timing은 Water8 독립 process 최소 3회 median을 사용한다. 필요한 경우 Water20을 FD 없이
제한적으로 사용한다. 단일 best timing이나 Water64 한 번의 결과로 completion을 선언하지
않는다.

## 4. Non-goals

이 goal에는 다음을 포함하지 않는다.

- GDF 추가 최적화 또는 FFTDF 변경
- stress/cell derivative
- multigrid
- open-shell/UHF
- 새로운 functional, basis 또는 pseudopotential 범위 확장
- `j2c_eig_always=True` analytic response
- cutoff, mesh 또는 k-point 미수렴을 kernel optimization으로 처리하는 것
- displaced-geometry FD를 analytic kernel 내부에 삽입하는 것
- Water64를 Codex가 직접 실행하는 것

## 5. 작업 규칙

각 변경은 다음 순서로 수행한다.

1. 기존 report 또는 warning-as-error test로 문제를 재현한다.
2. 하나의 구체적인 원인 가설과 ownership boundary를 적는다.
3. 최소 범위 patch를 적용한다.
4. primitive/algebra test를 먼저 통과한다.
5. small-system energy, gradient, CPU/GPU 및 FD parity를 확인한다.
6. 동일 입력 timing과 host/device memory를 비교한다.
7. keep 또는 revert와 근거를 기록한다.

대형 계산은 최종 scaling evidence이며 development test가 아니다. 큰 system의 OOM이나
중단은 small synthetic-memory 또는 forced-block test로 먼저 재현한다.

## 6. Progress Ledger

| Stage | State | 현재 evidence | 다음 행동 |
| --- | --- | --- | --- |
| N0 sampled-k dtype safety | `complete` | common complex dtype solve; warning-as-error r2SCAN `2 x 1 x 1` RSDF, RSDF/MGGA 11-test 및 공유 GDF 36-test 통과 | regression 유지 |
| N1 benchmark integrity | `pending` | 외부 종료 후 stale `running` progress와 이전 failed report가 같은 directory에 존재 가능 | run identity와 interrupted-state contract 설계 |
| N2 GPU source derivative | `pending` | Water8에서 host source derivative가 RSDF response의 주요 부분; Water64 사용자 실행에서도 높은 CPU 사용 관측 | primitive ownership map과 small parity fixture 작성 |
| N3 GPU metric derivative | `pending` | metric derivative는 명시적 host boundary | N2 완료 후 시작 |
| N4 conditional remainder | `pending` | profile 전에는 대상 미정 | N2/N3 후 exclusive profile |
| N5 release gate | `pending` | 이전 R2 release evidence 존재 | N0--N4 완료 범위만 재검증 |

## 7. 완료 정의

이 goal은 다음 조건을 모두 만족할 때 완료한다.

1. sampled-k RSDF Cholesky response가 허수부를 암묵적으로 버리지 않는다.
2. source와 metric derivative primitive의 GPU ownership이 구현되거나, 측정 근거에 따라
   특정 host boundary 유지가 명시적으로 결정된다.
3. 기존 source-contracted bounded-memory response와 independent omega/mesh contract가
   유지된다.
4. Gamma 및 sampled-k CPU/GPU/FD correctness gate가 통과한다.
5. benchmark failure와 external interruption이 재현 가능하게 구분된다.
6. 모든 fallback과 unsupported option은 명시적이며 silent approximation이 없다.

## 8. 2026-07-17 N0 implementation record

- `CDERIFactorization.differentiate()`는 factor, central CDERI, source derivative와
  metric derivative의 공통 dtype을 사용한다.
- Production RSDF component traversal은 derivative dtype을 pair당 한 번 반영해 각
  Cartesian component에서 factor/value를 반복 복사하지 않는다.
- 실수 central factor와 유의미한 복소수 source/metric derivative를 결합한 algebra
  regression이 CPU reference와 `1e-10` 이내로 일치하고 허수부를 보존한다.
- r2SCAN `2 x 1 x 1` sampled-k RSDF CPU/GPU gradient test는 `ComplexWarning`을 error로
  승격한 상태에서 통과했다.
- RSDF/MGGA 관련 11개 test가 58.014 s에, 공유 GDF GPU 36개 test가 550.286 s에
  통과했다. Gamma real path, sampled-k, dense/matrix-free Z-vector와 full-gradient
  parity를 포함한다.
