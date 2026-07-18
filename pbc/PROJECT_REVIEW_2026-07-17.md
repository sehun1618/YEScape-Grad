# PBC DC-DFT 통합 프로젝트 리뷰

**작성일:** 2026-07-17
**검토 대상:** `agent/gdf-rsdf-continuation`의 현재 작업 트리
**검토 관점:** 구조/API, 수치 정확성, GPU 구현, benchmark/validation
**기준 commit:** `e8116bd6b855b92ef956eb78c77d4524288d4404`
**검증 artifact:** `benchmark/review_release_readiness_validation.json`

## 1. 목적과 적용 범위

이 문서는 현재 프로젝트에 대한 여러 검토 결과를 실제 개발 목표에 맞게 다시
정리한 통합 리뷰다. 현재 목표는 다음 범위의 정확성과 GPU 성능을 유지·개선하는
것이다.

- restricted closed-shell periodic DC-DFT
- PBE, SCAN, r2SCAN target DFA
- GTH pseudopotential
- Gamma 및 sampled k-point
- FFTDF, GDF, RSDF
- fixed-cell analytic nuclear gradient
- GPU production route와 CPU small-system reference
- Water8/Water20 correctness gate와 Water64 scaling evidence

다음 항목은 현재 개발 우선순위에서 제외한다.

- molecular/PBC unrestricted(UHF) 및 ROHF scheme
- NLC/VV10 계열 functional
- CPU large-system production scaling
- arbitrary diffuse/linearly dependent auxiliary basis까지의 일반화
- stress/cell derivative와 open-shell 확장
- GPU4PySCF가 권장하는 CUDA별 설치 방식 자체의 변경

범위 밖 기능은 지금 구현할 필요가 없다. 다만 잘못된 결과를 반환하지 않도록
unsupported route를 명시적으로 거부하는 최소 guard는 향후 public API 정리 단계에서
추가할 수 있다.

## 2. 종합 결론

PBC 구현의 핵심 수치 구조는 잘 설계되어 있다. Source HF, fixed-density target DFA,
orbital response, one-electron derivative와 DF별 response ownership이 분리되어 있고,
Gamma를 `Nk=1`로 정규화한 tensor convention도 일관적이다. 지원 범위 안의 small-system
analytic/FD 및 CPU/GPU parity는 충분히 설득력 있다.

현재 Water64 GDF/RSDF 결과도 앞서 제외한 U scheme, NLC/VV10 또는 GPU dependency
명명 방식 때문에 무효가 되지 않는다. 이 결과는 다음 용도로 유효하다.

- end-to-end timing 비교
- host/device memory scaling 비교
- 최적화 전후 analytic-gradient parity
- Z-vector convergence와 production route ownership 확인

다만 현재 Water64 실행은 `run_fd=false`이므로 absolute force-accuracy evidence가 아니라
성능 및 scaling evidence로 분류해야 한다. Absolute analytic/FD correctness는 Water8과
기존 small-system release fixture가 담당한다.

현재 목표에 가장 직접적인 개선 대상은 scientific kernel의 재작성보다 benchmark
artifact의 식별성, 정확도 상태의 의미, software provenance와 public object lifecycle이다.

## 3. 검증 현황

이번 리뷰에서 확인한 자동·추가 검증 결과는 다음과 같다.

| 검증 | 결과 |
| --- | ---: |
| PBC CPU unit tests | 49 passed, 112.518 s |
| PBC GPU unit tests | 91 passed, 1078.039 s |
| Benchmark lifecycle/evidence tests | 19 passed, 0.078 s |
| Python `compileall` | passed |
| `git diff --check` | passed |
| arbitrary complex-k FFTDF analytic/FD | 최대 오차 약 `2.38e-8 Ha/Bohr` |
| non-time-reversal GDF analytic/FD | 최대 오차 약 `3.69e-8 Ha/Bohr` |
| Water8 RSDF selected FD | 최대 오차 약 `2.70e-10 Ha/Bohr` |

따라서 기본 설정의 PBE/SCAN/r2SCAN, matching mesh, restricted closed-shell route는
회귀 상태가 좋다. 아래 지적 사항은 대부분 기존 test가 다루지 않는 상태 변경,
custom option 또는 benchmark lifecycle 경계에서 발생한다.

위 suite의 명령, test 수와 시간은
`benchmark/review_release_readiness_validation.json`에 고정했다. 표의 arbitrary
complex-k FFTDF와 non-time-reversal GDF 수치는 리뷰 당시 ad-hoc probe 결과이며 아직
독립 release artifact 또는 상시 자동 gate로 승격되지 않았다.

## 4. 현재 Water64 결과의 해석

### 4.1 RSDF 최적화 효과

`RSDF_Water64_SZ_Optimization.md`와 최종 bounded R4 report를 기준으로 한 변화는 다음과
같다.

| 항목 | Baseline | 최종 bounded R4 | 변화 |
| --- | ---: | ---: | ---: |
| analytic total | 234.4114 s | 164.3980 s | -29.87% |
| host peak RSS | 42.344 GiB | 6.691 GiB | -84.20% |
| sampled device peak | 92.873 GiB | 23.201 GiB | -75.02% |

이 개선은 의미가 크며, provider-owned central CDERI reuse, bounded reciprocal source
blocks, pseudopotential cache와 bounded GPU PP value route가 실제 Water64 scaling
문제를 완화했다는 근거로 사용할 수 있다.

### 4.2 정확도 evidence의 경계

현재 다음 queue는 이름에 `selected_fd_check`가 있지만 실제로는 FD를 실행하지 않는다.

- `benchmark/water64_benchmark_queue.json`
- `benchmark/water64_rsdf_benchmark_queue.json`

두 report 모두 다음 상태다.

```text
fd.enabled = false
accuracy = null
```

따라서 이 artifact를 Water64 FD gate 통과 자료로 표현해서는 안 된다. 권장 분류는
다음과 같다.

- Water8 selected FD: equation/algebra correctness evidence
- Water20: intermediate scaling 및 이전 gradient parity
- Water64: timing, memory 및 analytic-before/after parity evidence

### 4.3 Translation/grid drift

저장된 Water64 analytic gradient array의 원자별 합을 직접 계산하면 다음과 같다.

```text
GDF  ||sum_A dE/dR_A|| = 8.843511e-3 Ha/Bohr
RSDF ||sum_A dE/dR_A|| = 8.843481e-3 Ha/Bohr
GDF-RSDF max component gap = 1.900679e-5 Ha/Bohr
```

두 backend의 net-gradient drift가 거의 동일하므로 GDF/RSDF response 차이보다는 공통
XC/grid 경로가 원인이라는 증거다. 이후 Water8에서 비정상 grid phase를 선택해
rigid-translation energy FD와 analytic translation sum을 직접 비교했다. GDF/r2SCAN,
400 Ha에서 두 값은 각각 `5.601245648e-3`, `5.601212649e-3 Ha/Bohr`로
`3.30e-8 Ha/Bohr`까지 일치했다. 항별 합에서는 같은 `5.601245648e-3 Ha/Bohr`가
거의 전부 `target_xc`였고 one-electron, Coulomb, nuclear, Pulay 및 response channel은
수치 0이었다. RSDF도 같은 amplitude와 slope를 보였다.

따라서 이 drift는 누락된 GDF/RSDF analytic response가 아니라 finite uniform-grid에서
평가한 target-DFA XC energy의 egg-box 오차를 analytic하게 미분한 값이다. 400 Ha에서
r2SCAN의 phase-scan amplitude는 PBE보다 약 12배 컸고, 1000 Ha에서는 400 Ha 대비 약
90% 감소했지만 완전히 사라지지는 않았다. Cutoff가 바뀌면 atom-grid phase도 바뀌므로
한 geometry의 net sum은 단조 수렴할 필요가 없다. 상세 evidence와 운용 정책은
`TRANSLATION_SUM_DIAGNOSIS_2026-07-17.md` 및
`benchmark/translation_sum_diagnosis_water8.json`에 고정했다. 현재 성능 비교는
유효하지만 geometry optimization 또는 dynamics에서는 mesh/gradient convergence와
net-gradient gate가 필요하다.

## 5. 현재 범위에서의 우선 개선사항

### P1. Benchmark input/source fingerprint

`benchmark/run_benchmark_queue.py`의 완료 판정은 report의
`status == "complete"`만 검사한다 (`260-266`, `484-488`). Case directory 이름도
index, structure stem, job name, device와 backend로 구성되며 numerical settings와 source
revision은 포함하지 않는다 (`469-471`).

따라서 같은 output root와 job name을 재사용하면서 다음 값을 바꾸면 이전 report를
현재 결과로 잘못 skip할 수 있다.

- basis, pseudo, auxbasis
- cutoff/mesh, kmesh
- XC, SCF/Z-vector tolerance
- GDF/RSDF setup mode
- source code revision
- XYZ 파일 내용

현재 개발 과정에서는 매 최적화마다 다른 output root를 사용해 이 문제를 수동으로
회피했다. 장기적으로 다음 fingerprint를 `run_input.json`, `attempt.json`,
`progress.json`, `report.json`에 동일하게 저장해야 한다.

```text
SHA256(canonical effective config
       + resolved XYZ coordinates/lattice
       + git commit and dirty-state digest)
```

완료 report를 skip하기 전 fingerprint 일치를 강제하는 것이 가장 중요하다.

### P1. 실행 완료와 정확도 통과의 분리

`benchmark/dcdft_benchmark.py`는 FD error를 계산하더라도 report status를 항상
`complete`로 기록한다 (`967-993`, `1116-1120`, `1144-1146`). Queue exit status도
실행 status만 본다 (`run_benchmark_queue.py:525-528`).

다음 두 상태를 분리해야 한다.

```text
execution_status = complete | failed | interrupted
accuracy_status  = not_requested | pass | fail
```

선택적으로 다음 입력을 지원하면 release queue를 자동 gate로 사용할 수 있다.

```json
{
  "fd_max_abs_tolerance_ha_per_bohr": 1e-5,
  "net_gradient_tolerance_ha_per_bohr": null
}
```

Water64 timing queue는 `accuracy_status=not_requested`가 정상이며 실패로 취급할 필요가
없다. 대신 job 이름과 summary가 timing-only임을 명확하게 표시해야 한다.

### P1. Software provenance

Benchmark는 checkout의 `pbc/cpu/src`와 `pbc/gpu/src`를 `sys.path` 앞에 추가해 실행한다
(`benchmark/dcdft_benchmark.py:28-35`). 그러나 report에는 현재 다음 정보가 없다.

- git commit과 dirty state
- Python, NumPy, PySCF, GPU4PySCF, CuPy version
- CUDA runtime/driver version
- package source/wheel mode
- queue/config fingerprint

현재 timing 결과를 같은 장비에서 비교하는 데에는 충분했지만, 장기 보존되는 release
evidence에는 위 정보를 자동 기록해야 한다. GPU dependency 명명 방식을 바꾸라는 뜻이
아니라, GPU4PySCF 권장 설치가 실제로 어떤 version 조합으로 실행됐는지를 남기자는
의미다.

### P1. Public driver state invalidation

CPU/GPU `DCRestrictedBase`의 `density_fit`, `fft_density_fit`, `rs_density_fit`은
`with_df`만 교체하고 이전 `source_scf`, `target_ks`, `converged` state를 무효화하지
않는다.

- CPU: `cpu/src/pyscf_pbc_dcdft/dft/base.py:60-83`
- GPU: `gpu/src/gpu4pyscf_pbc_dcdft/dft/base.py:59-83`
- Context gate: `cpu/src/pyscf_pbc_dcdft/grad/context.py:76-126`

또한 두 번째 `kernel()`의 SCF가 실패해도 이전 성공 state가 남을 수 있다. 이 경우
이전 energy state와 새 provider 또는 geometry의 gradient를 섞을 수 있다.

현재 benchmark는 각 case와 displaced geometry마다 새 method를 생성하므로 기존
Water64 artifact에는 영향이 없다. Public API 안정화 단계에서 다음을 적용한다.

1. `kernel()` 진입 시 이전 계산 state 무효화
2. provider modifier 호출 시 state 무효화
3. energy completion 시 input signature 저장
4. gradient context 생성 시 source/target/current signature 일치 확인

### P1. FFTDF custom mesh 일관성

Public FFTDF provider는 `mesh=`를 받을 수 있지만 energy는 `with_df.mesh`, derivative는
여러 위치에서 `cell.mesh`를 사용한다.

- Provider: `cpu/src/pyscf_pbc_dcdft/df/fft/fft.py:29-35`
- Grid derivative: `cpu/src/pyscf_pbc_dcdft/df/fft/grid.py:17-38`
- J/K derivative: `cpu/src/pyscf_pbc_dcdft/df/fft/jk_derivative.py:117-153`
- Nonlocal PP derivative:
  `cpu/src/pyscf_pbc_dcdft/pseudo/fft_nonlocal_derivative.py:55-57`

H2 Gamma/PBE에서 `cell.mesh=[15,15,15]`, provider mesh `[9,11,13]`을 사용하면 한
gradient component의 analytic-FD 오차가 약 `1.80e-2 Ha/Bohr`였다. 현재 Water64
GDF/RSDF 결과에는 영향이 없고, benchmark FFTDF도 provider와 cell mesh를 같게
설정하므로 현 작업의 blocker는 아니다.

Public FFTDF API를 유지하려면 effective provider mesh를 모든 FFT derivative primitive에
전달해야 한다. 그렇지 않으면 kernel preflight에서 `with_df.mesh == cell.mesh`를
강제하는 편이 안전하다.

### P2. Restricted-scope guard

Public PBC driver는 restricted closed-shell을 전제로 하지만 spin, electron parity와
occupation을 package boundary에서 검사하지 않는다.

- `cpu/src/pyscf_pbc_dcdft/dft/base.py:34-58`
- `cpu/src/pyscf_pbc_dcdft/scf/reference.py:53-80`

통제된 Water 시스템은 항상 neutral closed-shell이므로 현재 결과에는 영향이 없다.
향후 잘못된 입력이 조용히 계산되지 않도록 다음 guard를 추가한다.

```text
cell.spin == 0
cell.nelectron % 2 == 0
converged occupations are exactly 0 or 2
sum(occupations)/Nk matches cell.nelectron
```

### P2. Queue concurrency와 FD resource lifetime

두 queue runner가 동시에 같은 case를 시작하지 못하게 하는 lock이 없다.
`_run_case()`는 기존 artifact를 archive한 뒤 child를 먼저 시작하고 attempt 파일을 쓴다
(`benchmark/run_benchmark_queue.py:284-327`). Case 단위 exclusive lock을 child lifetime
전체에 유지하는 것이 안전하다.

또한 FD helper의 `EnergyRun`은 사용되지 않는 full `method` reference를 보존한다
(`benchmark/dcdft_benchmark.py:57-65`, `783-805`). 다음 displacement 계산 중 이전
provider/CDERI가 함께 살아 있어 large-system FD memory를 왜곡할 수 있다. Water64 FD를
켜기 전에는 이 field를 제거하고 displacement별 resource를 명시적으로 release해야 한다.

## 6. 현재 비범위 또는 후순위 항목

### 6.1 UHF/ROHF와 molecular package

현재 목적에는 필요하지 않으므로 molecular UHF/ROHF end-to-end FD coverage와 GPU
UCPHF 진단은 후순위로 둔다. PBC restricted GPU 작업의 release gate에 포함하지 않는다.

### 6.2 NLC/VV10

현재 target functional이 PBE/SCAN/r2SCAN으로 고정되어 있으므로 NLC derivative 구현은
필요 없다. Public API 정리 시 `target.do_nlc()`를 명시적으로 거부하는 guard만 있으면
충분하다.

### 6.3 GPU dependency naming

GPU4PySCF가 권장하는 CUDA별 설치 순서를 프로젝트 전제로 유지한다. Dependency 이름을
임의로 재설계하지 않는다. CI에서는 다음 실제 사용자 경로만 확인하면 된다.

1. 권장 방식으로 CUDA별 CuPy/GPU4PySCF 설치
2. CPU package wheel 설치
3. GPU extension wheel 설치
4. import smoke와 한 개의 small GPU energy/gradient 실행

### 6.4 CPU large-system production

CPU Z-vector는 dense real-linear Hessian을 만들고 GDF/RSDF CPU response는 full derivative
tensor를 보존한다. CPU를 small-system equation/FD reference로 사용하는 현재 역할에는
문제가 없다. CPU Water64 production을 목표로 바꾸기 전까지 matrix-free/contracted
response 포팅은 필요하지 않다.

### 6.5 Eigen-metric fallback

PySCF가 auxiliary metric Cholesky 실패 후 eigen-factorization으로 자동 fallback하는
경우 project response가 reduced rank를 처리하지 못한다. 현재 검증된 GTH auxiliary basis
범위에서는 발생하지 않았으므로 일반 diffuse auxiliary basis 지원을 시작할 때 다룬다.

## 7. 권장 실행 순서

### Stage A: Evidence pipeline 정리

Scientific kernel을 건드리지 않고 먼저 다음을 적용한다.

1. Effective-config/source fingerprint 기록 및 skip 검증
2. `execution_status`와 `accuracy_status` 분리
3. Water64 timing job 이름에서 `fd_check` 제거
4. git/software/CUDA provenance 자동 기록
5. report에 net-gradient vector와 norm 추가

### Stage B: Public API guard

1. Driver state invalidation과 input signature 검증
2. Restricted closed-shell preflight
3. FFTDF custom-mesh 일관성 또는 명시적 제한
4. mesh-only Cell에서 `ke_cutoff=None` metadata 처리

### Stage C: Release automation

1. GPU4PySCF 권장 설치 방식 그대로 clean-wheel smoke CI 구성
2. Small Water8 selected FD와 Gamma/sampled-k parity 자동 gate
3. Full GPU suite는 self-hosted 또는 scheduled job으로 실행
4. Water64는 timing/memory scaling job으로 별도 유지

## 8. 완료 기준

현재 목표의 release-ready 상태는 다음 조건으로 정의할 수 있다.

- PBE/SCAN/r2SCAN restricted PBC의 small-system FD/parity gate 통과
- GDF/RSDF Gamma 및 sampled-k GPU test 통과
- Z-vector residual gate 통과
- Water64 timing artifact에 input/source/software fingerprint 존재
- 완료 report가 다른 config/source에서 재사용되지 않음
- Water64 timing과 small-system accuracy evidence가 명확히 구분됨
- 권장 GPU4PySCF 설치 방식의 clean-wheel import/example smoke 통과

이 기준에서는 UHF/ROHF, NLC/VV10, CPU Water64 production 또는 arbitrary auxiliary
basis 일반화가 완료 조건에 포함되지 않는다.

## 9. 2026-07-17 반영 결과

리뷰 후 scientific kernel을 바꾸지 않고 evidence pipeline과 public API boundary를
다음과 같이 정리했다.

- Effective config, XYZ content, Git commit, source-only dirty state와 source-tree를
  합친 SHA256 fingerprint를 `run_input`, `attempt`, `progress`, `report` 및 manifest에
  공통 기록한다. 동일 fingerprint의 completed report만 skip한다.
- `execution_status`와 `accuracy_status`를 분리했다. Optional FD 및 net-gradient
  tolerance가 요청되면 queue exit code가 accuracy failure도 반영한다.
- 모든 report에 analytic-gradient translation-sum vector/norm과 Python, package,
  CUDA software provenance를 기록한다.
- Case lock을 child lifetime 전체에 유지한다. FD helper는 더 이상 이전 displacement의
  full method/provider를 다음 displacement 동안 보존하지 않는다.
- CPU/GPU provider modifier와 `kernel()` 진입 시 이전 state를 무효화한다. 성공한
  계산에는 geometry, lattice, k-point, XC 및 provider input signature를 저장하고,
  density/gradient 요청 전에 현재 signature와 비교한다.
- Spin 0, even electron count, 0/2 occupation restricted contract를 public boundary에서
  검증한다.
- Custom FFTDF provider mesh가 `cell.mesh`와 다르면 조용히 잘못된 gradient를 반환하지
  않고 kernel preflight에서 명시적으로 거부한다. Matching/default mesh는 기존 경로를
  유지한다.
- Water8 rigid-translation probe로 Water64에서 관찰한 net-gradient drift의 공통 원인을
  target-DFA uniform-grid egg-box로 확정했다. Diagnostic은 analytic sum, total-energy FD,
  항별 translation sum과 cutoff/XC/DF backend scan을 같은 public route로 계산한다.

검증 결과는 CPU 49개, GPU 91개, benchmark 19개 test가 모두 통과했다. H2 CPU FFTDF
queue smoke에서 두 번째 동일 fingerprint 실행은 실제로 skip됐고 translation-sum norm은
`1.30e-14 Ha/Bohr`였다. 사용자 소유 Water64 queue의 기존 job 이름은 이번 변경에서
수정하지 않았다. Clean-wheel CI와 Water64 artifact 재생성은 Stage C로 남는다.
