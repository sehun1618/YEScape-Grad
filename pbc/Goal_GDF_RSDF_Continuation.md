# GDF and RSDF Continuation Goal

## 0. 문서의 지위

이 문서는 restricted closed-shell PBC DC-DFT의 GDF 및 RSDF 경로를 유지보수하고
최적화하기 위한 후속 goal이다. 기존 구현의 물리 기능과 정확성 completion을
취소하거나 다시 정의하지 않는다.

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
- multigrid
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
| R0 RSDF baseline | `active` | 기능/정확성 release gate만 존재 | 고정 water8/20 timing queue와 component/ownership profile 작성 |
| R1 RSDF contracted response | `pending` | full response/intermediate 비용 미분해 | R0 profile 후 범위 확정 |
| R2 RSDF release | `pending` | 기존 correctness evidence 존재 | 최적화 후 전체 gate 재실행 |

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
