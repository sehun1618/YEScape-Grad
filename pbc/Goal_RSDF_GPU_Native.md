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
| N1 benchmark integrity | `complete` | attempt identity, process-liveness reconciliation, interrupted state 및 prior-artifact archive; benchmark unit test 12개와 Water8 3회/Water20 1회 통과 | lifecycle regression 유지 |
| N2 GPU source derivative | `complete` | SR/LR+vbar native contraction, Gamma/sampled-k canonical orientation, source adjoint algebra, bounded reciprocal rebatch 및 shared two-observable traversal; RSDF 12개와 MGGA 4개 test 통과 | regression 유지 |
| N3 GPU metric derivative | `complete` | exact host SR `ip1/ip2` + GPU reciprocal metric/Cholesky adjoint; RSDF 14개와 MGGA 4개 test, Water20 35.2% N2 대비 개선 | regression 유지 |
| N4 conditional remainder | `complete` | Water20 source SR 21.330 s 병목을 exact fused adjoint로 10.676 s까지 축소; Water8 median 26.9%, Water20 34.7% end-to-end 개선 | fused/resolved regression 유지 |
| N5 release gate | `complete` | selected FD $2.70\times10^{-10}$ Ha/Bohr, Z residual $2.58\times10^{-14}$; wheel install/import, CPU/GPU example 및 GDF 교차 회귀 통과 | release regression 유지 |

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

## 9. 2026-07-17 N1 implementation record

- 각 queue 실행은 `case_run_id`, UUID `attempt_id`, KST 시작 시각과 hostname으로 된
  `run_identity`를 `run_input.json`, `attempt.json`, `progress.json` 및 `report.json`에
  공유한다.
- Queue runner는 PID뿐 아니라 Linux process start tick도 비교한다. 기록된 child가
  사라졌거나 PID가 재사용된 경우 이전 attempt와 progress를 `interrupted`로 바꾼다.
  살아 있는 child는 `skipped_running`으로 남겨 중복 계산을 막는다.
- 재실행 전 root의 input, log, progress, report 및 array를
  `attempts/<KST>_<attempt-id>/`로 함께 보존한다. 새 attempt와 identity가 다른 report는
  성공 결과로 읽지 않는다.
- lifecycle, stale-report rejection, live/dead process 판정, atomic progress와 archive를
  포함한 benchmark unit test 12개가 통과했다.
- `results/rsdf_n1_integrity_2026-07-17`에서 Water8 세 번과 FD 없는 Water20 한 번이 모두
  완료됐다. Water8 analytic-gradient wall-time 중앙값은 7.952 s, Water20은 58.377 s다.
- 모든 report/attempt identity가 일치했다. `source_omega=0.3`, `metric_omega=0.4`,
  source/metric mesh `21^3/31^3`, host-to-device bytes, host peak RSS 및 sampled VRAM peak가
  보존됐다. Water20의 host source/metric derivative time은 각각 28.999 s와 15.596 s로
  N2/N3의 측정 기준선이다.

## 10. 2026-07-17 N2 implementation record

- central relation $C=L^{-1}B^T$를 reverse-mode로 미분해 target-J와 restricted-HF
  Z-bilinear의 canonical $dB$ adjoint를 직접 만든다. 무작위 복소수 Gamma/sampled-k
  algebra가 기존 $dC$ contraction과 $10^{-10}$ 이내로 일치한다.
- short-range source derivative는 GPU4PySCF의
  `PBCsr_ejk_int3c2e_ip1`을 exact RSDF source omega, precision, image list와 BvK mesh로
  호출한다. reciprocal derivative는 `PBC_ft_aopair_ek_ip1`과 source mesh를 사용하며,
  GPU4PySCF가 $G=0$ Coulomb weight에 포함하는 vbar convention을 보존한다.
- source derivative primitive를 host에 만들거나 전송하지 않는다. target-J와
  Z-bilinear source adjoint는 한 번만 basis 변환하고, momentum context와 reciprocal
  AO/aux Fourier integral traversal을 공유한다. 여러 source를 받지 못하는 기존 CUDA
  integral kernel 호출만 observable별로 남는다.
- Gamma와 $2\times1\times1$ sampled-k의 SR 및 LR+vbar primitive, target-J와
  Z-bilinear contraction, materialized full-tensor reference 및 public full-gradient가
  모두 $10^{-9}$ 또는 해당 release tolerance 이내로 일치한다. forced block-1과
  transactional OOM retry도 통과한다.
- 전체 RSDF test 12개가 220.172 s, SCAN/r2SCAN을 포함한 MGGA test 4개가
  73.673 s에 통과했다. 관측된 `ResourceWarning`은 PySCF temporary-file 경로이며
  complex-casting 또는 RSDF numerical warning이 아니다.
- 동일 Water20 입력에서 analytic-gradient wall time은 58.377 s에서 50.136 s로
  14.1% 감소했다. host source derivative는 28.999 s에서 0 s, primitive transfer는
  2.350 GiB에서 0.032 GiB로 98.6% 감소했다.
- Water8 3회 median은 7.952 s에서 8.327 s로 약 4.7% 증가했다. 작은 계에서는 두
  observable에 대한 기존 single-source CUDA integral kernel launch 고정비가 CPU
  primitive materialization보다 크다. 이 사실을 숨기는 자동 CPU fallback은 넣지
  않았으며, true multi-source CUDA recurrence는 N3 이후 profile에서 N4 후보로만
  재평가한다.

## 11. 2026-07-17 N3 implementation record

- Central relation $C=L^{-1}B^T$의 plain-product $dC$ adjoint를 Cholesky gauge 뒤로
  전파해 $\operatorname{Re}\sum_{PQ}W^M_{PQ}dM_{PQ}$ 형태의 metric adjoint를 만든다.
  무작위 복소 Gamma/sampled-k algebra와 materialized $dM$ contraction은
  $10^{-10}$ 이내로 일치한다.
- Reciprocal metric derivative는 auxiliary Fourier value, Coulomb kernel 및
  $i(\mathbf G+\mathbf q)$ phase를 GPU에서 직접 수축한다. Atom-resolved
  `(natm, 3, naux, naux)` metric derivative와 full $dC$는 만들지 않는다.
- GPU4PySCF의 현재 public `int2c2e_ip1`은 production PySCF RSDF screened metric과
  H2 probe에서 최대 약 $5.29\times10^{-5}$ 차이가 났고 rcut 증가로 사라지지 않았다.
  이를 연결하면 $10^{-9}$ correctness contract를 깨므로, exact production host
  `ip1/ip2` 두 배열만 전송해 device에서 직접 수축한다. Water20에서 이 exact host
  short-range/context boundary는 0.109 s이며 full host metric derivative 15.121 s의
  1% 미만이다.
- Source와 metric native route는 같은 canonical $dC$ adjoint를 공유한다. 각각을 독립
  opt-out할 수 있어 host/host, native-source/host-metric,
  host-source/native-metric 및 native/native 네 조합을 full-tensor witness와 비교한다.
- RSDF test 14개가 261.847 s, SCAN/r2SCAN MGGA test 4개가 74.624 s에 통과했다.
  Gamma와 $2\times1\times1$ sampled-k 최종 gradient, complex dtype, batch-1 및 OOM
  transaction 회귀를 포함한다.
- Water8 독립 process 세 번의 analytic-gradient 중앙값은 N2 8.327 s에서 5.884 s로
  29.3% 감소했다. Water20은 50.136 s에서 32.468 s로 35.2% 감소했다.
  Water20 host metric derivative는 15.121 s에서 0.109 s로 99.3% 감소했고
  host-to-device primitive bytes는 34,675,200 B에서 16,275,392 B로 53.1% 감소했다.
- Default profile mode는 `native_source_native_metric_contracted_adjoint`이며 component
  batch 값은 `null`이다. Reference/mixed host derivative route에서만 bounded component
  stream과 transactional OOM rebatch를 사용한다. Silent CPU fallback은 없다.

## 12. 2026-07-17 N4 implementation record

- N3 이후 Water20 exclusive profile을 source setup, short-range, reciprocal, metric 및
  adjoint로 분리했다. 전체 32.958 s 중 native source가 23.243 s, 그중 short-range가
  21.330 s로 명확한 단일 병목이었다. Central CDERI, H1 또는 XC를 추측해 포팅하지
  않았다.
- 최종 DC gradient에는 target-J derivative와 restricted-HF Z-bilinear의 합만 필요하다.
  Reverse map의 선형성
  $R^*(W_J+W_Z)=R^*W_J+R^*W_Z$를 이용해 두 CDERI adjoint를 source/metric primitive
  contraction 전에 합쳤다. 따라서 현재 single-source CUDA short-range kernel은 canonical
  pair당 두 번이 아니라 한 번 호출된다. Cutoff, mesh, screening 또는 물리식은 바꾸지
  않았다.
- Production gradient는 `observable_mode="fused"`와
  `response_mode=native_source_native_metric_fused_adjoint`를 사용하고 최종 합을
  `coupled` contribution으로 조립한다. 성분별 target-J와 Z-bilinear가 필요한 진단은
  `observable_mode="resolved"`로 기존 2-pass reference를 그대로 사용할 수 있다.
- Fused direct contraction은 resolved 두 성분의 합과 $10^{-9}$ 이내로 일치했다.
  Water8/Water20의 archived N3 gradient와 최대 차이는 각각
  $7.42\times10^{-14}$와 $6.18\times10^{-14}$ Ha/Bohr였다.
- Water8 독립 process 세 번의 median은 5.884 s에서 4.299 s로 26.9% 감소했다.
  Water20은 32.468 s에서 21.202 s로 34.7% 감소했고, short-range source는
  21.330 s에서 10.676 s로 감소했다. 따라서 N4의 5% keep gate를 통과한다.
- RSDF test 14개와 SCAN/r2SCAN MGGA test 4개, 총 18개가 311.173 s에 통과했다.
  Gamma/sampled-k, mixed ownership, batch-1 및 transactional OOM 계약을 포함한다.
  Water64는 실행하지 않았다.

## 13. 2026-07-17 N5 release record

- Water8/Gamma/r2SCAN의 atom 0 z component를 $h=10^{-4}$ Bohr central FD로 다시
  검증했다. Fused analytic gradient와 FD의 차이는
  $2.696\times10^{-10}$ Ha/Bohr이고 Z-vector residual은
  $2.579\times10^{-14}$이다.
- Gamma 및 sampled-k PBE CPU/GPU, Gamma SCAN, sampled-k r2SCAN을 통과했다. Sampled-k
  r2SCAN test는 `ComplexWarning`을 error로 승격한다. Source reciprocal forced block-1,
  transactional OOM rebatch, host/native ownership 조합과 full-tensor witness도 통과했다.
- 공유 `grad/gdf.py`의 coupled 조립 회귀를 막기 위해 GDF Gamma/sampled-k fused
  full-gradient test 2개를 추가 확인했다. 두 test 모두 통과했다.
- CPU/GPU package wheel을 생성하고 임시 target에 설치해 source tree 밖에서 import했다.
  CPU/GPU sampled-k RSDF 공개 예제도 각각 완료됐다. Wheel SHA256은 validation JSON에
  고정했다.
- Public capability와 gradient metadata는 native source derivative, reciprocal
  metric/Cholesky adjoint, fused target-J/Z-bilinear, exact host short-range metric 및
  central-CDERI boundary를 구분해 보고한다. Silent CPU fallback과 누락 block은 없다.
- N0--N5 전체 goal이 완료됐다. Water64는 실행하지 않았다.

## 14. 2026-07-17 conditional central follow-up

- 이후 사용자가 SZV-GTH에 한해 Water64 실행을 명시적으로 허용했다. 이 section의
  Water64 한정 scaling evidence는 위의 기존 non-goal을 변경하지 않으며 DZ 계산은
  실행하지 않았다.
- GPU4PySCF의 unmodified central builder는 PySCF CDERI와 최대 약
  $7.2\times10^{-6}$ 차이가 났다. 분해 결과 GPU three-center source는
  $7.74\times10^{-12}$ 이내로 일치했고, 차이는 GPU screened two-center metric에
  있었다.
- `exact_gpu_source` opt-in은 큰 source와 reciprocal metric/Cholesky를 GPU에서,
  작은 exact screened metric만 PySCF에서 계산한다. Gamma, heterogeneous H2O 및
  $2\times1\times1$ central/full-gradient gate를 통과했다. CPU parameter witness의
  `_cderi`는 `None`이다.
- Upstream source builder가 free VRAM 대부분을 batch로 잡지 않도록 reported working
  memory를 512 MiB로 제한했다. Water64 HF-stage peak는 첫 exact probe의
  19.52 GiB에서 10.17 GiB로 감소했다.
- Water8 3회 median의 analytic total은 2.06%, host RSS는 7.75% 감소했다. Water64
  host RSS는 10.24% 감소했지만 analytic total은 3.32% 증가하고 전체 sampled VRAM
  peak도 감소하지 않았다.
- 따라서 `reference`를 default로 유지하고 `exact_gpu_source`는 host-memory 제약을
  위한 명시적 opt-in으로 보존한다. 5% end-to-end promotion gate를 통과하지 않은
  경로를 자동 선택하거나 silent fallback하지 않는다.
- 최종 회귀 검증은 RSDF GPU 25개, SCAN/r2SCAN MGGA 4개, benchmark lifecycle 12개,
  CPU/GPU package-boundary 4개 test를 통과했다. Public sampled-k exact-central 예제도
  완료됐다. R5에서는 Water64 FD를 새로 실행하지 않았으며, Water64 정확성 판단은
  analytic reference parity이고 FD gate는 동일 response/assembly 경로를 사용하는
  소형 release system이 담당한다.
