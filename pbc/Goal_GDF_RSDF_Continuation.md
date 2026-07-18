# GDF and RSDF Continuation Goal

## 0. 문서의 지위

이 문서는 restricted closed-shell PBC DC-DFT의 GDF 및 RSDF 경로를 유지보수하고
최적화하기 위한 후속 goal이다. 기존 구현의 물리 기능과 정확성 completion을
취소하거나 다시 정의하지 않는다.

이 문서에서 완료된 G6--G8 및 R0--R2 이후의 RSDF GPU-native 작업은
[`Goal_RSDF_GPU_Native.md`](Goal_RSDF_GPU_Native.md)에서 별도로 추적한다.

- 기준 source는 이 Git 저장소의 `pbc/cpu/src/`와 `pbc/gpu/src/`다.
- 이전 개발 workspace는 회귀 evidence와 구현 이력을 읽는 용도로만 사용한다.
- GDF와 RSDF의 correctness를 보존한 상태에서 memory 안정성, 재현 가능한 timing,
  대형 문제의 scaling을 개선한다.
- 한 번에 하나의 측정된 병목만 수정하고, 각 변경은 별도 commit 또는 PR로 남긴다.
- Codex는 water64 계산을 직접 시작하거나 재실행하지 않는다. 사용자가 생성한
  water64 결과는 read-only evidence로 분석한다.

## 1. 현재 기준선

이미 지원되는 범위는 다음과 같다.

- RHF/KRHF source density
- PBE, SCAN, r2SCAN target DFA
- GTH pseudopotential
- Gamma 및 sampled k-point
- CPU/GPU GDF와 RSDF
- fixed-cell analytic nuclear gradient
- matrix-free preconditioned Z-vector solver

현재 작업은 새 물리 기능 구현이 아니라 성능 및 안정성 maintenance track이다.

### 1.1 GDF 완료 항목

- matrix-free MINRES Z-vector
- nuclear CPHF source batching
- signed two-density one-pass derivative
- target Veff/Fock workspace reuse
- self-adjoint HF response contraction
- bounded Cartesian H1 streaming
- explicit opt-in `one_pass_fused` target-J/Z-bilinear traversal
- Gamma 및 sampled-k correctness gates

### 1.2 RSDF 완료 항목

- CPU/GPU analytic energy and gradient
- Gamma 및 sampled k-point
- independent source/metric omega and mesh
- source, metric 및 auxiliary-basis response
- PBE, SCAN 및 r2SCAN gates

`j2c_eig_always=True`는 현재 지원 범위가 아니며 silent approximation이나 CPU
fallback을 사용하지 않는다.

## 2. 즉시 해결할 GDF memory incident

Water64/DZVP-GTH, Gamma, r2SCAN, 500 Ha의 기존 사용자 실행은
`gdf_bilinear_z_response`에서 `cudaErrorMemoryAllocation`으로 종료됐다. 실패 allocation은
Gamma reciprocal context의 다음 tensor였다.

```text
occupied_g.shape = (nocc, nocc, 2 * ngrid_block)
```

현재 block-size 추정은 주로 `pair_g`와 `tmp`의 크기를 반영하지만 다음 동시 live
allocation을 충분히 포함하지 않는다.

- `occupied_g`, which scales as `nocc**2 * ngrid_block`
- `aux_g`와 `aux_weighted`
- Fourier-transform evaluator의 임시 workspace
- 이미 보존 중인 density/state/J3C tensor와 CuPy memory-pool reservation

따라서 available memory가 충분해 보이더라도 실제 peak보다 큰 reciprocal block을
선택할 수 있다. 이 문제는 SCF, target-DFA energy 또는 Z-vector convergence failure가
아니라 GDF bilinear derivative의 block-memory policy 문제다.

## 3. 작업 순서

### Stage G6: Gamma GDF bounded-memory repair

1. reciprocal block의 모든 명시적 live tensor를 byte 단위로 계산하는 memory estimator를
   만든다.
2. FT evaluator와 기존 device state를 위한 명시적 safety reserve를 둔다.
3. 최소 block 크기조차 안전하지 않으면 allocation 전에 설명 가능한 `MemoryError`를
   발생시킨다.
4. allocation failure 시 block을 bounded하게 축소해 재시도할 수 있지만 CPU route로
   자동 전환하지 않는다.
5. state별 `tmp`와 `occupied_g` lifetime을 최소화하고, 가능하면 contraction을
   streaming하여 `occupied_g` materialization 자체를 줄인다.
6. 선택된 block 크기, estimated peak bytes, free bytes와 retry 횟수를 timing metadata에
   저장한다.

Completion gates:

- forced-low-memory unit test에서 deterministic block 축소
- Gamma multi-block parity
- sampled-k route 무회귀
- water8 및 제한된 water20 energy/gradient parity
- no silent CPU fallback, no dropped block, no NaN/Inf
- water64 실행은 사용자가 수행하며 Codex release gate에 포함하지 않음

### Stage G7: Post-closure GDF timing

이미 구현된 workspace reuse, self-adjoint response, H1 streaming 및
`one_pass_fused`의 실제 end-to-end 효과를 같은 입력과 hardware에서 측정한다.

1. water8을 독립 process에서 최소 3회 반복한다.
2. 필요한 경우 water20을 FD 없이 제한적으로 3회 반복한다.
3. component-resolved `one_pass`와 opt-in `one_pass_fused`를 비교한다.
4. energy, gradient, Z-vector residual과 peak host/device memory를 함께 기록한다.
5. median과 run-to-run drift를 사용하고 단일 best timing을 사용하지 않는다.

Primary metrics:

- end-to-end analytic-gradient wall time
- GDF target-J derivative time
- GDF bilinear Z-response time
- target workspace reuse hit/miss
- H1 streaming peak memory
- Z-vector iterations and residual

`one_pass_fused`는 target-J와 Z-bilinear의 합은 보존하지만 두 성분의 사후 분리를
제공하지 않는다. 의미 있는 speedup이 확인되더라도 reporting contract를 검토하기
전에는 explicit opt-in으로 유지한다.

### Stage G8: Profile-driven GDF optimization

G7 결과에서 가장 큰 exclusive stage 하나만 선택한다. 후보는 다음과 같지만 profile
없이 구현을 시작하지 않는다.

- native GDF source/metric/auxiliary derivative traversal
- remaining host periodic-integral boundaries
- CDERI/context setup and reuse
- target XC/grid evaluation
- allocation, synchronization or host-device transfer

같은 가설을 두 번 시험해 end-to-end 변화가 5% 미만이면 해당 micro-optimization을
중단하고 다음 상위 병목으로 이동한다.

### Stage R0: Reproducible RSDF baseline

GDF timing과 같은 형식으로 water8 및 제한된 water20 RSDF baseline을 만든다.

- source and metric omega/mesh를 report에 고정
- CDERI setup, target energy, AX/Z-vector, source response, metric response,
  auxiliary response를 분리
- host/device ownership과 transferred bytes 기록
- FD는 매 timing run에서 반복하지 않고 기존 release evidence를 사용

### Stage R1: RSDF source-contracted response

Profile에서 derivative tensor materialization이 병목으로 확인되면 다음을 수행한다.

1. full `dCDERI`, `dJ` 또는 `dK` tensor를 만들지 않고 density/source와 즉시
   contraction한다.
2. source, metric 및 auxiliary response를 bounded block으로 처리한다.
3. GDF의 workspace reuse, self-adjoint response와 H1 streaming을 재사용할 수 있는
   공통 interface로 정리한다.
4. RSDF 고유의 independent omega/mesh contract를 보존한다.
5. GPU-native periodic derivative primitive는 host integral generation이 profile상 실제
   최대 병목일 때만 포팅한다.

### Stage R2: RSDF release gate

- CPU/GPU energy difference `<= 1e-8 Ha`
- CPU/GPU gradient maximum difference `<= 1e-7 Ha/Bohr`
- analytic-FD maximum difference `<= 1e-5 Ha/Bohr`
- Z-vector residual `<= 1e-8`
- Gamma와 sampled-k parity
- PBE, SCAN과 r2SCAN smoke/regression
- bounded-memory batch 축소 parity
- package import, wheel and examples smoke

## 4. Non-goals

이 continuation goal에는 다음을 포함하지 않는다.

- FFTDF 추가 최적화
- stress/cell derivative
- multigrid (별도 `Goal_GDF_RSDF_Multigrid.md`에서 추적)
- open-shell/UHF
- 새로운 functional, basis 또는 pseudopotential 범위 확장
- system-specific threshold나 fitted force correction
- displaced-geometry FD를 analytic kernel 내부에 삽입
- water64를 Codex가 직접 실행하는 것
- cutoff, mesh 또는 k-point 미수렴을 kernel optimization으로 취급하는 것

## 5. 작업 규칙

각 변경은 다음 순서를 따른다.

1. 기존 report 또는 profiler로 병목을 수치화한다.
2. 하나의 구체적인 원인 가설을 적는다.
3. 최소 범위 patch를 적용한다.
4. small-system unit/parity test를 실행한다.
5. 동일 입력 timing을 비교한다.
6. keep 또는 revert를 명시한다.

Water64와 같은 large-system 결과는 최종 scaling evidence이지 개발 중 unit test가 아니다.
대형 계산 실패는 먼저 small forced-block 또는 synthetic-memory test로 재현한다.

## 6. Progress Ledger

| Stage | State | 현재 evidence | 다음 행동 |
| --- | --- | --- | --- |
| G6 bounded-memory repair | `complete` | complete live-memory estimator/reserve, transactional OOM rebatch, planner 7-test 및 GDF GPU 35-test 통과, water8 selected-FD $2.09\times10^{-9}$ Ha/Bohr, water20 prior-gradient gap $3.23\times10^{-14}$ Ha/Bohr | 사용자 Water64 evidence는 optional follow-up이며 자동 gate가 아님 |
| G7 post-closure timing | `complete` | water8 3회 median: one-pass 6.2552 s, fused 4.7703 s; water20 limited no-FD 32.1472 s | compact validation evidence 유지 |
| G8 profile-driven GDF optimization | `complete` | CUDA profile에서 SR integral kernel 82%; Gamma derivative omega 1.5배로 water8 one-pass/fused median 30.3%/23.4%, water20 34.2% 개선, FD/parity 및 35-test 통과 | sampled-k는 변경하지 않고 현재 Gamma 정책 유지 |
| R0 RSDF baseline | `complete` | water8 3회 gradient median 224.3317 s; one-electron 146.1063 s, RSDF response 76.1478 s; host/device timing 및 byte ownership 기록 | compact evidence 유지 |
| R1 RSDF contracted response | `complete` | bulk source/metric derivative, native H1 reuse, canonical-pair direct contraction으로 full `dCDERI/dJ/dK` 제거; water8 median 7.3928 s, retained derivative 0 B, FD $2.72\times10^{-10}$ Ha/Bohr | final package gate 유지 |
| R2 RSDF release | `complete` | Gamma/sampled-k PBE 및 SCAN/r2SCAN parity, forced batch-1와 transactional OOM rebatch, 관련 29-test, wheel/import/example, 공유 GDF 35-test 통과 | compact evidence 유지 |

## 7. 완료 정의

이 goal은 다음 조건을 모두 만족할 때 완료한다.

1. GDF Gamma reciprocal route가 계산된 memory bound 안에서 동작한다.
2. GDF 후속 최적화의 반복 가능한 end-to-end 효과가 기록된다.
3. GDF와 RSDF 모두 Gamma 및 sampled-k correctness를 유지한다.
4. RSDF의 실제 최대 병목에 대해 bounded-memory 개선을 구현하거나, 측정 결과 현
   구현 유지가 타당함을 기록한다.
5. 모든 실패는 명시적이며 silent CPU fallback이나 task truncation이 없다.
6. large-system 계산 없이도 memory policy와 block completeness를 검증하는 회귀 테스트가
   존재한다.

## 8. 2026-07-16 G6 implementation record

- `df/gdf/reciprocal_memory.py`가 zero-order와 derivative의 전체 명시적 live tensor를
  별도로 산정하고 더 큰 peak를 block 계획에 사용한다.
- 가용 memory의 65% 이하만 사용하며 최소 512 MiB를 남긴다.
- zero-order와 derivative block은 allocation/계산 성공 후에만 commit되며,
  `cudaErrorMemoryAllocation`이면 같은 interval을 절반 block으로 재시도한다.
- `tmp`, `occupied_g`, weighted auxiliary tensor를 재사용하고 derivative의 별도
  `exchange_source` materialization을 제거했다.
- 성공 report는 `gdf_reciprocal_memory`에 계획과 두 phase의 실행 결과를 기록한다.
- Pure scheduler/planner test 7개와 GDF GPU module 35개가 통과했다. 여기에는 Gamma 강제
  multi-block, sampled-k, one-pass/fused, small-batch 및 dense/matrix-free Z-vector parity가
  포함된다.
- Water64는 실행하지 않았다. Water8 선택 FD, water20 이전 gradient parity와 반복 timing이
  G6/G7 gate를 통과했으며, 사용자 Water64-DZ 재실행은 optional scaling evidence다.

## 9. 2026-07-16 G7 timing record

- Water8 `one_pass` gradient wall time: 6.3322, 6.2552, 6.2453 s; median 6.2552 s.
- Water8 `one_pass_fused`: 4.9564, 4.7703, 4.7667 s; median 4.7703 s.
- 이전 동일 water8 `one_pass` median 7.0863 s 대비 bounded-memory workspace/lifetime patch는
  11.7% 빨랐다. Fused는 새 one-pass보다 23.7% 빨랐다.
- 모든 water8 route/run gradient의 최대 차이는 $7.50\times10^{-14}$ Ha/Bohr였고 선택
  FD 오차는 $2.09\times10^{-9}$ Ha/Bohr였다.
- Water20 limited no-FD gradient는 32.1472 s, Z residual은
  $5.72\times10^{-14}$, 이전 검증 gradient와 최대 차이는
  $3.23\times10^{-14}$ Ha/Bohr였다.
- G8의 첫 측정 대상은 fused GDF derivative traversal이다. Median 2.5030 s로 water8 fused
  gradient의 약 52.5%를 차지한다.

## 10. 2026-07-16 G8 optimization record

- CUDA-event profile은 fused traversal 2.5020 s 중 short-range derivative 2.0606 s,
  그중 `PBCsr_ejk_int3c2e_ip1` 2.0569 s와 source contraction 0.0015 s를 보였다.
- Shell/auxiliary batching 완화는 kernel을 2.0569 s에서 2.0549 s로만 바꿔 5% keep gate를
  넘지 못했으므로 revert했다.
- Gamma derivative에서 upstream `_guess_omega(cell)`의 1.5배를 사용해 SR workload를 LR로
  균형 이동했다. Water8에서 SR kernel은 0.8561 s, LR derivative는 0.2409 s였다.
- 2.0배는 reciprocal vector 19683개와 LR 0.5005 s로 다시 느려져 채택하지 않았다.
- 최종 water8 3회 median은 one-pass 4.3592 s, fused 3.6517 s다. G7 대비 각각 30.3%,
  23.4% 개선됐다.
- Water20은 32.1472 s에서 21.1607 s로 34.2% 개선됐다.
- Water8 선택 FD 오차는 $2.09\times10^{-9}$ Ha/Bohr, 이전 gradient와 water8/20 최대
  차이는 $7.38\times10^{-12}$/$1.65\times10^{-11}$ Ha/Bohr다.
- Gamma/k-point 전체 GDF GPU 35-test module이 551.310 s에 통과했다. sampled-k omega
  policy는 변경하지 않았다.

## 11. 2026-07-16 R0 RSDF baseline record

- Water8/SZV-GTH/GTH-SCAN/r2SCAN/400 Ha/Gamma를 독립 process 세 번 측정했다.
- Analytic-gradient wall time은 222.3333, 224.3317, 231.9235 s였고 median은
  224.3317 s였다.
- One-electron median은 146.1063 s, RSDF response median은 76.1478 s였다.
- Response 중 source derivative median은 71.3266 s, metric derivative median은
  4.5648 s였다.
- Full response는 device derivative output 382,205,952 B를 보존했다. 따라서 R1은 host
  source 반복, H1 tensor, full derivative materialization 순서로 범위를 좁혔다.
- Water20 과거 evidence는 read-only scaling 참고로만 사용했고 재실행하지 않았다.

## 12. 2026-07-16 R1 RSDF optimization record

- Source와 metric primitive는 원자마다 integral context를 다시 만들지 않고 all-atom
  derivative를 한 번 생성한다.
- RSDF one-electron H1은 DF backend와 무관한 기존 bounded native GDF H1 streaming을
  재사용하고 RSDF two-electron response만 별도로 더한다.
- Production `RSDFContractedResponse`는 canonical k-pair와 nuclear component 하나의
  $dC$를 만든 즉시 target-J와 Z-vector bilinear scalar로 수축한다. Full `dCDERI`,
  `dJ`, `dK`는 반환하거나 보존하지 않는다.
- Full-tensor reference, production scalar assembly, bounded-memory execution을 각각
  `response.py`, `contracted_response.py`, `component_stream.py`로 분리했다. Public API는
  유지되며 memory/OOM 정책은 물리 응답 조립과 독립적으로 테스트된다.
- Component upload의 기본 상한은 3이며 free-memory preflight를 거친다. OOM batch는 결과를
  commit하지 않고 절반으로 줄여 같은 interval부터 재시도한다. Forced batch-1 parity와
  synthetic 4-to-2 OOM retry가 통과했다.
- 최종 Water8 wall time은 6.9598, 7.3928, 7.5430 s, median 7.3928 s로 R0 대비
  96.70% 감소했다.
- One-copy full $dC$ equivalent는 63,700,992 B지만 retained derivative는 0 B이고,
  명시적 component working estimate는 6,303,744 B다.
- 선택 FD 오차는 $2.717\times10^{-10}$ Ha/Bohr, Z residual은
  $2.581\times10^{-14}$ 이하, R0 gradient gap은 $4.337\times10^{-11}$ Ha/Bohr다.
- Gamma와 sampled-k PBE CPU/GPU parity, SCAN Gamma와 r2SCAN sampled-k regression이
  통과했다. Water64는 실행하지 않았다.

## 13. 2026-07-16 R2 release record

- CPU RSDF 5, CPU MGGA 4, GPU RSDF 7, GPU MGGA 4, package/import/benchmark 9개로
  구성된 관련 회귀 29개가 모듈 분리 후 다시 통과했다.
- CPU와 GPU wheel을 build isolation 없이 각각 빌드하고 임시 target에 설치한 뒤 source
  tree가 아닌 wheel package에서 import했다.
- CPU sampled-k RSDF와 GPU sampled-k RSDF public example이 wheel 설치본으로 실행됐다.
- 공유 `grad/gdf.py` 변경의 GDF 무회귀를 확인하기 위해 GDF GPU 35-test 전체 모듈을
  재실행했고 550.958 s에 통과했다.
- Release evidence는
  `pbc/gpu/validation/results/2026-07-16_rsdf_optimization.json`에 고정했다.
- 모든 completion gate가 small/medium test로 닫혔고 Water64는 실행하지 않았다.

## 14. 2026-07-18 DZ larger-basis repair record

- 사용자 Water64/DZVP 네 경로 evidence에서 GDF는 reciprocal derivative의 AO reshape,
  RSDF는 source HF exchange의 allocation에서 실패했다. Water64는 재실행하지 않고 생성된
  report를 read-only evidence로 사용했다.
- GDF 실패는 spherical physical AO 1472와 GPU4PySCF `SortedCell`의 Cartesian AO
  1536을 혼동한 문제였다. 산소의 d shell이 spherical 5개에서 Cartesian 6개로 확장되어
  산소 64개만큼 차이가 났다. Gamma와 sampled-k signed derivative가 모두 transformed
  density의 sorted-Cartesian AO 차원을 사용하도록 수정했다.
- H2O/DZVP는 physical AO 23, sorted AO 24로 같은 조건을 작게 재현한다. One-pass와
  two-pass analytic derivative의 최대 차이는 $10^{-8}$ Ha/Bohr gate 안에서 통과했다.
- Canonical GDF/RSDF exchange는 auxiliary 전체의
  `einsum("pij,jk,pkl->il")`을 제거하고 free-VRAM 기반 auxiliary block 두 단계 matmul로
  교체했다. Allocation failure는 결과를 commit하지 않고 같은 auxiliary interval을 절반
  block으로 재시도하며 CPU fallback은 없다.
- 22.5 GiB free-memory Water64/DZ planner, forced block-1 complex parity와 synthetic OOM
  retry 테스트가 통과했다. 기존 Gamma/sampled-k GDF와 RSDF exact-source energy/gradient
  회귀도 통과했다.
- Public H2O/DZVP DC-PBE analytic 경로는 GDF와 RSDF 모두 완료됐다. 두 경로의 energy
  차이는 $3.40\times10^{-7}$ Ha, gradient 최대 차이는
  $5.01\times10^{-7}$ Ha/Bohr였다.
- Water64/DZVP 최종 completion은 사용자가 수정 후 queue를 재실행해 확인한다.

## 15. 2026-07-18 Gamma RSDF source-adjoint memory candidate

- 수정된 Water64/DZVP report에서 RSDF uniform과 `multigrid_xc`가 모두
  `gdf_response_tensors` 단계의 `build_pair_cderi_adjoints()`에서 실패했다. 첫 번째
  `target` allocation 뒤 `bilinear = cp.zeros_like(target)`가 실패했으며, 이는 source HF,
  target DFA 또는 Z-vector 수렴 문제가 아니다.
- Gamma 중앙 CDERI, reference density와 response density는 `float64`였지만
  `auxiliary_density_contraction()` 결과 container가 `complex128`이라 adjoint 전체가
  불필요하게 복소화됐다. $n_\mathrm{aux}=n_\mathrm{AO}=1472$에서 한 장은
  47.527 GiB이며, 같은 shape의 target, bilinear와 exchange temporary를 동시에 만들 수
  없었다.
- Production gradient는 target-J와 restricted-HF bilinear의 합만 사용한다. 새 경로는
  두 adjoint를 따로 만든 뒤 더하지 않고 Coulomb/exchange 항을 하나의 fused output에
  순차 누적한다. Exact-real Gamma 입력은 `float64`를 유지하므로 fused adjoint 한 장은
  23.764 GiB다. Exchange temporary도 한 번에 한 장만 존재한다.
- Cholesky reverse solve와 metric adjoint는 같은 fused array를 공유한다. Native source
  basis transform 직전 pair execution과 source-adjoint map의 이전 참조를 제거하고,
  observable 하나를 `cp.stack`으로 복사하지 않는다. Physical source 23.764 GiB,
  Cartesian AO transform 25.875 GiB와 fully sorted transform 27.000 GiB가 세대별로
  소비되도록 바뀌었다.
- 이 변경은 sampled-k complex algebra를 실수로 cast하지 않는다. Gamma에서 core input이
  모두 exact-real일 때만 real dtype을 선택하고, sampled-k와 complex central block은 기존
  complex path를 유지한다.
- Fused-vs-resolved algebra, single-source ownership consumption, H2O/DZVP 실제 RSDF
  sorted transform, full-tensor contracted-response witness 및 Gamma/sampled-k exact-source
  end-to-end gradient test가 모두 통과했다.
- 상태는 `candidate; Water64/DZVP rerun required`다. 이 수리 후에도 fragmentation 또는
  다른 provider-owned allocation 때문에 실패하면 다음 단계는 AO-pair reverse solve와
  native source contraction의 진짜 streaming/fusion이며, silent CPU fallback은 사용하지
  않는다.

### 15.1 First Water64 rerun and auxiliary-block repair

- Commit `e2bf2e5`의 첫 사용자 재실행은 이전 `bilinear=zeros_like(target)`를 통과했지만
  같은 `gdf_response_tensors` 단계에서 다시 실패했다. 새 traceback은 fused output에 두
  번째 Coulomb outer product를 더하기 위한
  `cp.multiply(auxiliary[:,None,None], density.T[None])` 전체 temporary였다.
- 이는 23.764 GiB fused output은 성공적으로 할당됐지만, 남은 free VRAM 21.514 GiB보다
  큰 23.764 GiB temporary를 추가로 요구했기 때문이다. 따라서 real/fused dtype 수리는
  유효했지만 전체 auxiliary 축을 한 번에 계산하는 연산이 하나 남아 있었다.
- Fused adjoint assembly를 auxiliary block 단위로 변경했다. 기본 block 상한은 32이며
  Water64/DZVP real tensor에서 한 block은 약 0.516 GiB, output/temporary/workspace 세 배
  보수 추정은 약 1.55 GiB다. Planner는 현재 free VRAM의 20%만 사용하고 최소 2 GiB를
  남긴다.
- 각 block은 별도 working array에서 Coulomb과 exchange 네 항을 모두 완성한 뒤 output에
  commit한다. Allocation failure이면 미완성 block을 버리고 같은 구간을 절반 크기로
  다시 계산하므로 중복 누적이나 누락이 없다.
- Forced block-1 Gamma parity, sampled-k complex pair parity, synthetic OOM 4-to-2 retry와
  21.5 GiB free-memory Water64 planner test가 통과했다. 상태는
  `candidate v2; post-blocking Water64/DZVP rerun required`다.

### 15.2 Reverse-solve and sorted-source lifetime repair

- Blocked fused output 23.764 GiB를 완성한 뒤 기존
  `solve_triangular(L.T, adjoint_flat)`은 C-order RHS를 Fortran order로 복사해 같은 크기의
  source adjoint를 추가로 만들 수 있었다. CuBLAS right-side triangular solve로
  $A^T L^{-1}$을 C-contiguous fused storage에 직접 덮어쓰도록 변경했다. 반환 source
  adjoint는 이 storage의 transpose view이며 추가 dense allocation이 아니다.
- Metric adjoint는 overwrite 전에 먼저 계산한다. 따라서 source와 metric reverse map은
  같은 fused array를 순서대로 공유하며 수식과 Cholesky gauge는 바뀌지 않는다.
- Source sorted-basis 변환에는 physical source, Cartesian-AO intermediate와 fully-sorted
  output이 순서대로 필요하다. 현재 free VRAM으로 두 변환 phase가 들어가지 않을 때만
  이미 마지막 J/K 사용이 끝난 provider central CDERI cache를 해제한다. Provider는
  release byte와 이유를 metadata에 기록하며 이후 J/K 요청에서 factors를 재구축한다.
- Fully-sorted `(naux,nao,nao)` 결과를 `(nao,nao,naux)` C-order로 다시 복사하지 않고
  strided transpose view로 native SR/LR contraction에 전달한다. Existing kernels는
  auxiliary slice 또는 einsum에서 필요한 block만 materialize한다.
- Real/complex in-place solve parity, pressure-triggered cache release와 H2O/DZVP consumed
  transform이 통과했다. Fused block plan과 central-release decision은
  `device_fused_adjoint_memory` metadata로 report에 남는다.
- 상태는 `candidate v3; Water64/DZVP rerun required`다. 현재 실행 중이던 두 번째 RSDF
  process는 `e2bf2e5`를 이미 import한 뒤 시작됐으므로 v2/v3 patch의 검증으로 사용하지
  않는다.

### 15.3 Second Water64 rerun and metric dtype repair

- Commit `090a259`의 Water64/DZVP RSDF 재실행은 blocked fused assembly를 통과했다.
  다음 실패는 metric reverse map의
  `value_flat.astype(common_dtype, copy=False)`였으며, 당시 최소 free VRAM은
  17,099,718,656 bytes(15.925 GiB)였다. 즉 v3가 이전 outer-product OOM을 해결했고,
  metric adjoint가 large `(naux, nao**2)` operand를 공통 dtype으로 복사하는 다음
  allocation이 드러났다.
- Restricted Gamma Z-response는 물리적으로 실수지만 matrix-free solver와 MO gauge 때문에
  complex container에 roundoff 허수부가 남을 수 있다. Central CDERI가 실수이고 reference
  및 response density의 최대 허수부가 `1e-10` 이하일 때만 네 density/auxiliary operand를
  zero-copy real view로 정규화한다. 허수부가 이 gate보다 크면 complex path를 유지한다.
- Metric adjoint는 이 Gamma 정규화와 독립적으로 mixed dtype large operand를 더 이상
  통째로 cast하지 않는다. 실수부/허수부 행렬곱을 원래 dtype view에서 수행하고 작은
  `(naux,naux)` gauge adjoint만 complex로 조립한다. GPU4PySCF Gamma unpacker의
  auxiliary-fast strided CDERI layout도 그대로 소비한다.
- Roundoff-only Gamma, significant-imaginary Gamma, strided real-CDERI/complex-adjoint metric
  parity와 exact GPU-source Gamma/sampled-k 전체 gradient가 통과했다. RSDF GPU 전체 suite는
  35/35 통과했으며 wall time은 433.240 s였다.
- 상태는 `candidate v4; Water64/DZVP rerun required`다. v4 이후 실패가 source sorted-basis
  transform에서 발생하면 다음 단계는 fully-sorted source materialization을 제거하는
  auxiliary/source contraction streaming이다.
