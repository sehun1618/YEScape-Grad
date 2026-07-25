# PBC DC-DFT Stress Optimization Goal

## 0. 문서의 지위

이 문서는 이미 구현된 restricted PBC DC-DFT analytic stress의 정확성을 유지하면서
wall time과 host/device memory를 줄이기 위한 독립 active goal이다.

- 전체 stress 이론, 구현 역사와 지원 범위는 `stress/Goal.md`가 소유한다.
- 이 문서는 성능 profile, 최적화, production 승격 조건만 소유한다.
- 모든 구현, test, benchmark와 문서는 `stress/` 아래에서 작업한다.
- 검증된 `grad/`, `pbc/cpu/`, `pbc/gpu/` 코드는 read-only reference로 취급한다.
- numerical parameter, mesh, cutoff, range-separation omega 또는 수렴 tolerance를 완화해
  얻은 시간 감소는 기본 최적화 성과로 인정하지 않는다.
- Water20/64와 그 DZ scaling 계산은 사용자가 실행한다. 자동 development gate는 H2와
  Water8까지만 사용하며 H2 DZ correctness는 허용한다.

현재 상태 (2026-07-23):

- 외부 package/algorithm survey와 equation-owner map, profiler/ownership schema 완료
- O2 exact GPU nonlocal GTH PP owner 완료: Gamma, complex sampled-k와 DZVP
  d-polarization parity 및 Water1/Water8 FD 통과
- O4 reverse-Cholesky GPU FT source owner 완료: GDF/RSDF, Gamma/`2x1x1`, H2/Water8
  selected FD 통과
- H2 `DZVP-GTH` GDF/RSDF Gamma/`2x1x1` 네 gate의 최대 FD 오차는
  `2.95e-11 Eh/Bohr^3`
- Water8 profiler-off Gamma median stress/RSS는 RSDF 8.057 s/3.135 GiB, native GDF
  12.209 s/3.569 GiB. 직전 bounded-forward 대비 stress는 79.9%/95.5%, RSS는
  64.7%/71.3% 감소
- Water8 `2x1x1` FD 오차는 RSDF `6.74e-11`, native GDF
  `6.73e-11 Eh/Bohr^3`; total stress는 41.33/48.67 s
- package API 기본값은 보수적인 host-analytic route를 유지한다. 최신 GPU user queue는
  exact optimized PP/DF backend를 명시적으로 선택하며 report에 owner를 남긴다
- 첫 Water20 user-run의 Gamma sampled-k/BvK layout 오류와 과대 G-block을 수정했다.
  재실행은 GDF/RSDF 모두 완료됐고 scaling validator를 통과했다. 자동 block은
  `16384`에서 `4656`, 예상 active work는 21.33 GiB에서 6.06 GiB로 감소했다
- Water20 stress/RSS는 GDF 61.52 s/5.43 GiB, RSDF 52.51 s/5.20 GiB다. 두 report는
  exact owner, `production=true`, numerical component 없음, silent fallback 없음을
  기록한다. 이 결과로 system-scaling release gate는 완료됐다
- Release audit은 `docs/Release_Audit_2026-07-23.md`, machine-readable scaling evidence는
  `validation/results/2026-07-23_water20_scaling_gate.json`에 둔다. Python API 기본값은
  자동 변경하지 않았으며, 검증된 exact GPU route는 명시적 production opt-in으로 유지한다
- 최소 Water20 GDF/RSDF stress-only queue와 scaling validator를 추가했다. Validator는
  exact owner, `production=true`, numerical/fallback 부재와 wall/RSS/VRAM evidence를
  모두 요구한다

## 1. 현재 실행 경로와 지위

현재 GPU stress에는 서로 다른 지위를 가진 세 경로가 있다.

1. **Exact analytic reference**
   - one-electron, PP, GDF/RSDF, XC와 adjoint response를 analytic strain derivative로
     계산한다.
   - 정확성 기준점이며 최적화 중 항상 보존한다.
2. **Exact optimized contracted route**
   - analytic nonlocal PP와 reverse-source GPU FT로 필요한 six-Voigt scalar를 직접
     수축해 큰 derivative tensor를 피한다.
   - numerical cell displacement가 없고 `production=true`를 보고한다.
   - H2/Water8와 Gamma/`2x1x1`, GDF/RSDF FD gate를 통과한 현재 user-queue 선택 경로다.
3. **Legacy numerical contracted route**
   - 일부 PP/DF low-level cell derivative에 centered numerical displacement를 사용한다.
   - `production=false`이며 성능 역사와 독립 validation 용도로만 보존한다.

이번 goal의 analytic 교체, small/medium correctness와 Water20 scaling gate는 완료했다.
Python API 기본값 변경은 별도 compatibility 결정으로 남기며, 현재 user queue의 명시적
production opt-in과 API 기본값을 계속 구분한다.

## 2. 외부 package와 알고리즘 조사

구현 전에 `stress/docs/Stress_Optimization_Reference_Survey.md`를 만들고 각 reference를
우리 stress 식과 code owner에 대응시킨다.

### 2.1 조사 대상

- **PySCF**: `pyscf.pbc.grad.rks_stress`의 kinetic, overlap, Ewald, Coulomb/XC 및
  symmetric Voigt assembly를 직접 기준으로 사용한다.
- **GPU4PySCF**: PBC stress의 GPU array ownership, AO/grid blocking, RSJK/AFTDF stress와
  host fallback 경계를 조사한다.
- **CP2K Quickstep/GPW**: Gaussian product의 multigrid task placement, collocation,
  integration, FFT/Poisson pipeline, block-sparse matrix와 MPI/OpenMP/GPU task 분할을
  조사한다.
- **Quantum ESPRESSO 등 plane-wave package**: kinetic, local/nonlocal PP, Hartree, XC,
  Ewald stress decomposition과 reciprocal-space contraction을 독립적인 식 검산에 쓴다.
- **수치 선형대수**: matrix-free Krylov, block preconditioner, reverse-mode/adjoint,
  direct contraction, low-rank update, screening과 iterative refinement를 검토한다.

### 2.2 채택 조건

외부 전략은 다음 표를 작성한 뒤에만 구현 후보가 된다.

| 항목 | 필수 기록 |
|---|---|
| 수학적 대상 | 우리 식의 PP/DF/XC/Pulay/Z-vector 중 어느 항인지 |
| 자료구조 | AO, auxiliary, G-vector, grid, k-pair 중 어느 축을 줄이는지 |
| 정확성 | exact transformation인지 approximation/screening인지 |
| 기대 효과 | FLOP, kernel call, transfer, retained bytes 중 무엇을 줄이는지 |
| 적용 제한 | Gamma, sampled-k, GDF, RSDF, basis/PP 범위 |
| provenance | package/version, file 또는 paper, license |

CP2K와 Quantum ESPRESSO의 GPL source는 복사하거나 line-by-line 번역하지 않는다. 공개된
알고리즘과 식만 참고해 독립 구현한다. PySCF/GPU4PySCF의 Apache-2.0 source를 적응할 경우
기존 `NOTICE`와 `THIRD_PARTY_NOTICES` provenance 절차를 따른다.

참고 시작점:

- CP2K GPW: https://manual.cp2k.org/cp2k-2025_2-branch/methods/dft/gpw.html
- CP2K code structure: https://manual.cp2k.org/cp2k-2026_2-branch/development/code-structure.html
- PySCF PBC stress API: https://pyscf.org/pyscf_api_docs/pyscf.pbc.grad.html
- PySCF PBC DF: https://pyscf.org/user/pbc/df.html
- GPU4PySCF repository: https://github.com/pyscf/gpu4pyscf
- Quantum ESPRESSO repository: https://github.com/QEF/q-e

## 3. 공통 불변조건

- tensile-positive stress와 Voigt 순서 `[xx, yy, zz, yz, xz, xy]`를 유지한다.
- fractional atom coordinate, fractional k-point와 integer mesh의 strain contract를
  바꾸지 않는다.
- source HF, target DFA와 Z-vector는 중앙 state에서 한 번만 계산한다.
- six strain component 때문에 HF AX나 target Fock를 여섯 번 만들지 않는다.
- full derivative tensor를 만드는 reference route와 direct-contracted route는 같은 식을
  계산해야 한다.
- CPU fallback, device-to-host transfer, numerical derivative와 approximation은 metadata에
  명시한다. 조용한 fallback은 허용하지 않는다.
- OOM을 피하기 위한 block 축소는 동일 interval을 정확히 한 번 commit해야 한다.
- mixed precision, integral screening 또는 sparse truncation은 exact default와 분리된
  opt-in research stage에서만 검토한다.

## 4. Benchmark와 측정 규칙

### Stage O0: baseline lock

현재 source tree에서 exact analytic과 contracted fast route를 다시 측정한다. 과거 timing은
비교 자료로만 사용하고 새 최적화의 baseline으로 자동 승계하지 않는다.

필수 case:

- H2: PBE, Gamma 및 `kmesh=[2,1,1]`, GDF/RSDF, six-component FD
- Water8: r2SCAN, Gamma, GDF/RSDF, selected `xx/xy` FD
- exact analytic route와 contracted route를 동일 input으로 분리 측정

측정 규칙:

- 다른 GPU process가 없는 상태에서 1회 warm-up 후 3회 반복 median
- source energy, target Fock/Z, one-electron/PP, DF, XC, total stress를 분리
- host RSS, process-local CuPy used/reserved, global VRAM을 구분
- kernel-launch 수, host/device transfer bytes와 retained workspace를 가능한 범위에서 기록
- user benchmark와 development benchmark를 물리적으로 분리
- checkpoint 재사용 run은 성능 median에서 제외

Baseline 결과는 `stress/docs/Stress_Optimization_Log.md`와 development result summary에
기록한다.

## 5. 최적화 단계

### Stage O1: profiler와 ownership 완성

1. PP local/nonlocal, GDF/RSDF source/metric, AX와 XC의 exclusive timing을 추가한다.
2. CUDA event timing과 Python wall timing을 분리한다.
3. AO/grid/projector build 횟수와 k-pair/G-block 반복 수를 metadata에 기록한다.
4. peak가 발생한 stage와 live tensor estimate를 report에 남긴다.

Completion gates:

- stage timing 합과 total wall 차이 설명 가능
- CPU-only stage, host boundary와 GPU owner가 report와 실제 호출 경로에서 일치
- profiler를 껐을 때 결과 bitwise 또는 configured tolerance 내 동일

### Stage O2: nonlocal PP projector 병목

현재 fast Water8 route에서 nonlocal PP가 가장 큰 단일 병목이다.

1. source density와 adjoint density를 한 projector batch에서 수축한다.
2. 원소별 radial projector, angular channel, strain-invariant structure factor를 재사용한다.
3. atom/projector/channel loop를 GPU batch와 fused contraction으로 바꾼다.
4. symmetric Voigt 6성분만 계산하고 비대칭 9성분 temporary를 만들지 않는다.
5. AO/grid block은 free VRAM 기반 planner와 adaptive rebatching을 사용한다.

CP2K의 Gaussian-to-grid task list와 GPU batch scheduling은 loop organization의 참고로만
사용한다. 우리 GTH projector 식과 PySCF basis layout은 그대로 유지한다.

Completion gates:

- H2/Water8 exact PP contribution parity
- Gamma와 sampled-k parity
- PP wall time 25% 이상 감소 또는 total stress 15% 이상 감소
- peak RSS/VRAM 10% 초과 증가 없음

### Stage O3: local PP와 grid contraction

1. local-part1/part2, pseudo-G0, AO collocation과 structure factor의 중복 build를 제거한다.
2. source/adjoint density가 AO values와 reciprocal phase를 공유한다.
3. grid point와 G-vector를 block streaming하고 결과만 device에 유지한다.
4. CP2K multigrid의 task assignment를 분석하되 이번 stage에서 multigrid stress를 새 지원
   범위로 추가하지 않는다.
5. uniform grid의 mesh/phase 정확성을 바꾸지 않는다.

Completion gates:

- local PP와 total stress의 exact-route parity
- mesh별 결과 무회귀
- Water8 local PP wall time 감소와 transfer-byte 감소를 함께 입증

### Stage O4: GDF/RSDF direct-contracted analytic derivative

1. numerical cell displacement로 계산 중인 contracted DF source/metric derivative를
   analytic directional derivative로 교체한다.
2. full `dCDERI`, full PP strain matrix와 `(6,naux,nao,nao)` retained tensor를 만들지 않는다.
3. target-J와 adjoint source observable을 한 pair/G-block traversal에서 함께 수축한다.
4. GDF reciprocal source/metric context를 pair와 G-block 사이에서 재사용한다.
5. RSDF short-/long-range source와 metric을 별도 profile하고 bounded pipeline으로 처리한다.
6. sampled-k canonical pair와 Hermitian reverse pair를 중복 계산하지 않는다.
7. AO-pair strain은 large-image case에서 center/moment projection을 먼저 수행해
   `(nG,nao,nao)` image response의 반복 조립을 피한다. Auto 선택은 계산식이나
   numerical parameter를 바꾸지 않고 working-set 크기만 사용한다.

Completion gates:

- numerical low-level derivative 없이 `production=true` 가능
- exact analytic DF contribution과 maximum difference `<= 1e-10 Eh/Bohr^3`
- full selected energy FD `<= 1e-8 Eh/Bohr^3`
- GDF/RSDF 모두 host/device memory bound 준수

### Stage O5: Z-vector, AX와 k-point scaling

1. gradient와 stress가 같은 Z-vector 및 target Fock workspace를 공유한다.
2. self-adjoint HF response identity로 AX 호출을 최소화한다.
3. matrix-free Krylov preconditioner를 gap, orbital energy와 backend response에 맞춰 profile한다.
4. 여러 k-point와 density observable을 block matvec로 처리한다.
5. solver tolerance를 완화하지 않고 residual과 stress error를 함께 측정한다.

Completion gates:

- 중앙 state당 Z solve 1회
- strain six component당 AX 재호출 없음
- Gamma 이득과 sampled-k 이득을 별도 보고
- residual `<= 1e-8`, stress parity 유지

### Stage O6: optional research algorithms

다음은 exact production route 완료 후 별도 opt-in으로만 검토한다.

- Schwarz/locality screening과 sparse block contraction
- low-rank auxiliary/response representation
- FP32/TF32 tensor contraction 후 FP64 iterative refinement
- multi-GPU 또는 MPI G-vector/k-pair distribution
- automatic differentiation을 이용한 selected kernel 검산

각 approximation은 error estimator, strict fallback과 convergence study가 없으면 user API에
노출하지 않는다.

## 6. 정확성 gate

필수 조합은 다음과 같다.

| System | Sampling | XC | Backend | Reference |
|---|---|---|---|---|
| H2 | Gamma | PBE | GDF/RSDF | CPU analytic + six-component energy FD |
| H2 | 2x1x1 | PBE | GDF/RSDF | CPU analytic + selected FD |
| Water8 | Gamma | r2SCAN | GDF/RSDF | exact GPU analytic + `xx/xy` energy FD |
| Water8 | 2x1x1 | r2SCAN | GDF/RSDF | exact GPU analytic + `xx` energy FD |

Release tolerance:

- CPU/GPU exact analytic difference `<= 1e-10 Eh/Bohr^3`
- optimized/exact analytic difference `<= 1e-10 Eh/Bohr^3`
- selected full-energy FD difference `<= 1e-8 Eh/Bohr^3`
- Z-vector residual `<= 1e-8`
- stress antisymmetric residual과 pressure/virial convention test 통과
- energy와 nuclear gradient regression 무변경

## 7. 성능 채택 기준

개별 최적화는 다음 중 하나를 만족해야 채택한다.

- Water8 total stress median 15% 이상 감소
- 해당 target stage median 25% 이상 감소
- 동일 시간에서 peak host/device memory 25% 이상 감소
- 더 큰 계에서 OOM을 제거하면서 작은 계 overhead 10% 이하

동시에 다음을 모두 만족해야 한다.

- 정확성 gate 통과
- peak memory 10% 초과 증가 없음, 또는 증가 이유와 scaling 이점 명시
- 코드 경로와 metadata ownership 일치
- user default에 hidden numerical derivative, approximation 또는 CPU fallback 없음

Water20/64와 DZ는 사용자가 제공한 결과로 scaling gate를 닫는다. 해당 계산을 Codex가
자동 실행하지 않는다.

## 8. Production 승격 조건

Contracted fast route는 다음 조건을 모두 만족한 뒤에만 default 후보가 된다.

1. PP와 DF의 production-critical 항이 analytic derivative임
2. `production=true` metadata가 실제 호출 경로와 일치
3. Gamma/sampled-k, PBE/r2SCAN, GDF/RSDF gate 통과
4. H2/Water8 correctness와 dedicated 3-run median evidence 보유
5. Water20 또는 Water64 user-run memory/scaling evidence 1개 이상 보유
6. implementation 문서, equation-code map, example과 changelog 최신화
7. reference analytic route를 validation option으로 보존

## 9. Non-goals

이번 optimization goal에서 새로 구현하지 않는다.

- D2C/dispersion stress
- multigrid stress를 새 production backend로 추가
- FFTDF stress
- open-shell/UHF stress
- elastic constants 또는 second cell derivative
- NPT integrator와 cell optimizer
- system-dependent fitted correction
- exact gate 없이 mixed precision/screening을 기본값으로 사용

## 10. 완료 정의

이 goal은 단순히 더 빠른 benchmark 하나를 얻었을 때 완료되지 않는다. 다음이 모두 필요하다.

- reference survey와 equation/code ownership map 완료
- exact analytic reference 보존
- fast route의 PP/DF production-critical numerical derivative 제거
- 정확성, performance와 memory gate 통과
- user/development benchmark 분리와 checkpoint evidence 유지
- Water20/64 중 하나의 사용자 scaling 결과 확보
- README, optimization log, API metadata와 examples 최신화
