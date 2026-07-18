# GDF/RSDF with Target Multigrid Goal

## 0. 문서의 지위

이 문서는 restricted closed-shell PBC DC-DFT에서 이미 검증된 GDF/RSDF source-HF
경로를 유지하면서, fixed-density target DFA의 일부 또는 전체를 GPU multigrid로
평가하는 독립적인 후속 goal이다.

- GDF/RSDF correctness와 GPU-native 최적화 이력은
  `Goal_GDF_RSDF_Continuation.md`와 `Goal_RSDF_GPU_Native.md`에 고정한다.
- 이 goal은 source HF backend를 대체하지 않는다. Source RHF/KRHF의 $J$와 $K$는
  선택한 GDF 또는 RSDF가 계속 소유한다.
- Multigrid는 source SCF가 만든 $D_\mathrm{HF}$를 받아 target $J$, $E_\mathrm{xc}$,
  $V_\mathrm{xc}$ 및 그 nuclear derivative를 계산하는 별도 축이다.
- 기존 uniform-grid target route가 default이자 reference로 남는다. 새 route는
  correctness와 성능 gate를 통과하기 전까지 explicit opt-in이다.
- 한 단계에서 하나의 ownership 변경만 적용한다. Target XC-only와 target J+XC를
  동시에 구현하거나 검증하지 않는다.
- Water64는 development gate가 아니다. Water8과 제한된 Water20을 먼저 통과하고,
  Water64는 사용자가 요청하거나 실행한 경우에만 scaling evidence로 사용한다.

## 1. 용어와 현재 구조

이 문서에서 서로 다른 두 종류의 grid를 구분한다.

1. **Periodic integral mesh**: GDF/RSDF의 long-range Coulomb, Fourier transform 및 GTH
   pseudopotential 표현에 필요한 reciprocal/FFT mesh.
2. **Target XC quadrature grid**: $E_\mathrm{xc}[D_\mathrm{HF}]$와
   $V_\mathrm{xc}[D_\mathrm{HF}]$를 평가하는 real-space numerical integration grid.

Source HF는 XC quadrature를 사용하지 않지만, PBC J/K와 pseudopotential을 위해 첫 번째
mesh를 사용할 수 있다. 현재 target DFA는 다음과 같이 계산된다.

```text
source HF hcore/J/K       selected FFTDF/GDF/RSDF
source density            D_HF
target hcore              selected DF provider
target J                  selected DF provider
target XC                 GPU NumInt/KNumInt uniform grid
Z-vector Hessian/AX       source HF GDF/RSDF response
target dJ/dR              selected DF derivative
target dExc/dR            GPU uniform-grid derivative
```

현재 Water64/SZV/r2SCAN profile에서 target-DFA energy는 약 33--35 s,
`target_explicit_device_assembly`는 약 73 s다. 후자는 대부분 target-XC grid
derivative다. 따라서 target multigrid는 현재 확인된 큰 병목을 직접 겨냥한다.

## 2. 지원할 ownership 조합

### U0: Existing uniform reference

```text
source HF J/K       GDF or RSDF
target J            same GDF or RSDF
target XC           NumInt/KNumInt uniform grid
target PP/hcore     selected DF provider
```

현재 production/reference route다. 모든 새 route의 energy, Fock, gradient 및 timing
비교 기준으로 유지한다.

### M1: Target XC-only multigrid

```text
source HF J/K       GDF or RSDF
target J            same GDF or RSDF
target XC           GPU multigrid_v2
target PP/hcore     selected DF provider
```

이 route는 multigrid의 `nr_rks(..., with_j=False)`와
`get_veff_ip1(..., with_j=False,
with_pseudo_vloc_orbital_derivative=False)`를 사용한다. Target Coulomb energy/Fock와
$dJ/dR$은 기존 GDF/RSDF가 계속 소유한다.

Upstream `nr_rks`는 `with_j=False`에서도 density의 reciprocal Coulomb energy를 계산해
tag할 수 있다. M1 adapter는 이 `ecoul`을 사용하지 않고 selected-DF J에서 target
Coulomb energy를 다시 정의한다. Multigrid 반환값에서는 `exc`와 XC-only AO matrix만
채택한다.

M1은 target-XC quadrature만 바꾸므로 egg-box, MGGA $\tau$, k-point normalization 및
XC timing을 가장 분명하게 진단할 수 있다. M2보다 먼저 완료한다.

### M2: Full target J+XC multigrid

```text
source HF J/K       GDF or RSDF
target J+XC         GPU multigrid_v2
target PP/hcore     selected DF provider
Z-vector Hessian    source HF GDF or RSDF
```

이 route는 multigrid의 `nr_rks(..., with_j=True)`가 target Coulomb과 XC energy/Fock을
함께 만들고, `get_veff_ip1(..., with_j=True,
with_pseudo_vloc_orbital_derivative=False)`가 target $dJ/dR+dE_\mathrm{xc}/dR$을 함께
소유한다. 기존 GDF/RSDF target-J derivative를 더하지 않는다.

Source HF Hessian, source Fock nuclear derivative, $K$ response와 Z-vector AX는 여전히
source GDF/RSDF 경로다. Formal Coulomb operator는 같지만 finite auxiliary/mesh
representation은 다르므로 M2는 U0와 동일한 수치 모델이라고 가정하지 않고 별도
energy/gradient convergence gate를 둔다.

### M3: PP-only multigrid experiment

```text
source HF J/K       GDF or RSDF
source/target PP    GPU multigrid_v2
target J/XC         M1 or M2 policy
```

GPU4PySCF의 단순 `multigrid_numint()` 적용은 PP뿐 아니라 J ownership도 함께 바꿀 수
있으므로 사용하지 않는다. PP-only route는 별도 hcore provider로 다음 contract를
구현해야 한다.

```text
hcore = analytic kinetic + multigrid GTH PP
J/K   = selected GDF/RSDF get_jk
```

Central hcore뿐 아니라 explicit PP derivative, nuclear CPHF $h^{(1)}$, Z contraction과
FD displacement energy가 모두 같은 multigrid PP 표현을 사용해야 한다. 현재 Water64의
one-electron gradient는 약 9.2--9.4 s이고 translation-sum 분해에서 one-electron
channel은 수치 0이므로, M3는 M1/M2 이후 profile이 정당화할 때만 시작한다.

## 3. Public API와 source 설계

### 3.1 사용자 설정

GPU public driver와 benchmark에 다음 독립 option을 추가한다. CPU public driver는 현재
`uniform`만 허용하고 다른 값에는 명시적인 unsupported error를 낸다.

```python
target_integrator = "uniform"           # U0, default
target_integrator = "multigrid_xc"      # M1
target_integrator = "multigrid_j_xc"    # M2
```

PP-only 실험이 실제로 시작될 때만 별도 option을 추가한다.

```python
one_electron_backend = "selected_df"     # default
one_electron_backend = "multigrid_pp"    # future M3 opt-in
```

`target_integrator`, multigrid mesh/task policy 및 one-electron ownership은
`state.py` input signature, benchmark fingerprint, report metadata와 package capability에
반드시 기록한다. Energy 계산 후 option을 바꾸면 이전 state를 무효화한다.

### 3.2 구현 경계

GPU target integrator는 예를 들어 다음 namespace에 둔다.

```text
gpu/src/gpu4pyscf_pbc_dcdft/dft/target_integrators/
    __init__.py
    uniform.py
    multigrid.py
    result.py
```

공통 result는 다음을 명시적으로 반환한다.

```text
target hcore
target J 또는 J+Vxc AO matrix
target Coulomb energy
target XC energy
target Fock
J owner / XC owner / PP owner
mesh and multigrid task metadata
```

Upstream `target.multigrid_numint()`를 직접 호출하지 않는다. GPU4PySCF의
`get_hcore()`는 `_numint`가 multigrid이면 PP owner까지 바꿀 수 있고, `get_j()`도
multigrid로 전환할 수 있기 때문이다. Project adapter가 low-level multigrid call과
selected-DF hcore/J를 명시적으로 조합한다.

### 3.3 Gradient 경계

Gradient assembly는 target ownership에 따라 정확히 한 경로만 선택한다.

- U0: existing selected-DF $dJ$ + uniform $dE_\mathrm{xc}$
- M1: existing selected-DF $dJ$ + multigrid $dE_\mathrm{xc}$
- M2: multigrid combined $dJ+dE_\mathrm{xc}$; selected-DF target $dJ$ disabled
- M3: multigrid PP derivative; existing selected-DF PP derivative disabled

GPU4PySCF `multigrid_v2.get_veff_ip1()`은 현재 k-point 수만큼 scale된 값을 반환한다는
upstream contract가 있으므로, Gamma 및 sampled-k에서 $1/N_k$ normalization을
primitive-level test로 고정한다. MGGA에서는 density, gradient와 $\tau$ channel을 모두
같은 multigrid object와 mesh로 계산한다.

Target integrator는 target Fock VO RHS만 바꾼다. Matrix-free Z-vector operator $A$는
source HF Hessian이므로 GDF/RSDF `apply_hf_response()`를 그대로 사용한다. Target
`nr_rks_fxc` 부재는 현재 DC-DFT adjoint equation의 blocker가 아니다.

## 4. 작업 순서

### Stage MG0: Capability and ownership lock

1. 설치된 PySCF/GPU4PySCF multigrid API와 version을 report에 기록한다.
2. `nr_rks`의 `with_j`와 `get_veff_ip1`의 `with_j`, PP derivative flag 및 $N_k$
   normalization을 작은 synthetic density로 고정한다.
3. GDF/RSDF source HF energy, density와 orbitals가 target-integrator 선택에 따라 바뀌지
   않는 regression을 추가한다.
4. 같은 target term이 두 번 더해지면 실패하는 ownership assertion을 만든다.
5. GPU multigrid를 사용할 수 없는 environment에서는 silent NumInt/CPU fallback 대신
   명시적인 capability error를 발생시킨다.

Completion gates:

- target option 변경 시 state invalidation
- source HF energy difference `<= 1e-10 Ha`
- source density maximum difference `<= 1e-10`
- owner metadata와 actual call path 일치
- no PP/J double counting and no silent fallback

### Stage MG1: GPU target XC-only route

1. `MultiGridNumInt`를 target workspace가 소유하고 energy/gradient에서 재사용한다.
2. M1 target energy는 selected-DF J와 multigrid XC matrix를 조합한다.
3. Target Fock를 workspace에 보존해 Z-vector RHS에서 다시 XC를 계산하지 않는다.
4. `get_veff_ip1(with_j=False, with_pseudo...=False)`를 target-XC atom gradient로
   연결한다.
5. PBE와 r2SCAN의 Gamma 및 sampled-k normalization을 검증한다.

Completion gates:

- H2/H2O GDF 및 RSDF target energy/Fock convergence against tight U0
- PBE/r2SCAN analytic-FD maximum difference `<= 1e-5 Ha/Bohr`
- CPU/GPU 또는 tight-uniform reference gradient difference `<= 1e-7 Ha/Bohr` when
  the quadratures are converged to the same tolerance
- Z-vector residual `<= 1e-8`
- GDF/RSDF source and target-J derivative tests unchanged

### Stage MG2: GPU full target J+XC route

1. M2 energy/Fock를 `nr_rks(with_j=True)` 결과로 만든다.
2. M2 explicit target gradient를 combined multigrid $dJ+dE_\mathrm{xc}$로 연결한다.
3. GDF/RSDF target-J response가 호출되지 않았음을 metadata와 mocked ownership test로
   검증한다.
4. Source HF GDF/RSDF response와 M2 target response가 최종 gradient에서 한 번씩만
   들어가는 decomposition test를 추가한다.
5. M1과 M2의 target Coulomb energy/Fock 차이를 mesh/auxbasis convergence 관점에서
   기록한다.

Completion gates:

- GDF-source/M2와 RSDF-source/M2 각각 Gamma 및 sampled-k FD gate
- PBE, SCAN 및 r2SCAN smoke/regression
- target J+XC analytic-FD maximum difference `<= 1e-5 Ha/Bohr`
- source-HF energy/density identity with U0
- no selected-DF target-J derivative call in M2
- no PP ownership change

### Stage MG3: Translation, timing and memory evaluation

U0, M1과 M2를 같은 source density와 numerical settings에서 비교한다.

1. Water8에서 $x,y,z$ rigid phase scan을 수행한다.
2. 한 geometry의 translation sum뿐 아니라 phase-scan energy amplitude와 complete
   gradient mesh convergence를 비교한다.
3. Water8 독립 process 최소 3회 timing을 측정한다.
4. 제한된 Water20을 FD 없이 측정하고 target energy, target explicit gradient,
   end-to-end gradient, host RSS와 sampled VRAM을 기록한다.
5. Multigrid task 수, per-level mesh/work estimate와 build/reuse time을 metadata에 남긴다.

Keep criteria:

- analytic-FD 및 Z residual correctness gate 통과
- U0보다 translation phase amplitude가 악화되면 production default로 승격하지 않음
- end-to-end median speedup이 5% 이상이거나, 같은 force tolerance에서 memory/egg-box가
  의미 있게 개선됨
- single best run이 아니라 median과 run-to-run drift 사용

Multigrid는 finite-grid quadrature이므로 translation invariance를 자동 보장한다고
가정하지 않는다. Egg-box 개선과 속도 개선은 별도 결과로 판정한다.

### Stage MG4: Conditional PP-only experiment

MG1--MG3 완료 후 profile에서 one-electron/PP가 상위 병목일 때만 시작한다.

1. Selected-DF J/K를 강제하는 source HF subclass 또는 hcore adapter를 만든다.
2. Central source/target hcore를 동일한 multigrid PP로 만든다.
3. Explicit PP, nuclear CPHF source와 Z-response를 같은 representation으로 포팅한다.
4. One-electron component-level FD와 rigid-translation phase scan을 수행한다.
5. End-to-end 개선이 5% 미만이거나 translation error가 악화되면 route를 revert하거나
   diagnostic opt-in으로만 남긴다.

## 5. Benchmark와 evidence

Benchmark queue는 다음 field를 지원해야 한다.

```json
{
  "df_backend": "gdf",
  "target_integrator": "multigrid_j_xc",
  "target_multigrid_mesh": null,
  "run_fd": true,
  "net_gradient_tolerance_ha_per_bohr": 1e-5
}
```

`df_backend`는 source HF backend를 뜻한다. `target_integrator`는 fixed-density target
evaluation backend를 뜻한다. 두 field를 case 이름, summary, report와 fingerprint에
독립적으로 기록한다.

필수 비교 matrix:

| Source | Target | XC | Sampling | 역할 |
| --- | --- | --- | --- | --- |
| GDF | U0/M1/M2 | PBE | Gamma | 기본 algebra/FD |
| RSDF | U0/M1/M2 | PBE | Gamma | backend independence |
| GDF | U0/M1/M2 | r2SCAN | Gamma | MGGA and egg-box |
| RSDF | U0/M1/M2 | r2SCAN | sampled-k | complex-k normalization |

Water64 결과는 최종 scaling 참고이며 release correctness를 대신하지 않는다.

## 6. CPU reference policy

현재 CPU PySCF multigrid derivative는 sampled-k와 MGGA에서 GPU
`multigrid_v2.get_veff_ip1()`과 같은 범위를 제공하지 않는다. 따라서 이 goal은
GPU-first로 진행한다.

- U0 CPU analytic gradient는 equation/reference 역할을 유지한다.
- Multigrid target energy는 가능한 범위에서 CPU/GPU parity를 사용한다.
- r2SCAN multigrid gradient는 GPU analytic-vs-full-energy FD를 primary gate로 사용한다.
- CPU에 unsupported multigrid MGGA gradient를 임시 finite difference나 silent fallback으로
  package API에 넣지 않는다.

## 7. Non-goals

이 goal에는 다음을 포함하지 않는다.

- Source HF 전체를 multigrid로 대체하는 것
- Source HF의 J만 multigrid, K만 GDF/RSDF로 바꾸는 혼합 response
- Hybrid target DFA 또는 target exact exchange
- NLC/VV10
- stress/cell derivative
- open-shell/UHF
- phase-averaged/interlaced-grid correction 구현
- system-specific force projection 또는 translation-sum 강제 제거
- MG0--MG3 profile 없이 PP-only multigrid를 시작하는 것
- upstream PySCF/GPU4PySCF object를 monkey patch하여 ownership을 숨기는 것

## 8. 작업 규칙

각 변경은 다음 순서를 따른다.

1. U0 baseline과 변경하려는 term owner를 적는다.
2. Energy, Fock, explicit gradient와 response 중 영향을 받는 항을 열거한다.
3. 한 owner만 바꾸는 최소 patch를 적용한다.
4. Primitive/ownership test를 먼저 통과한다.
5. Small-system energy, gradient, FD와 translation scan을 실행한다.
6. 같은 input의 반복 timing과 memory를 비교한다.
7. Keep, opt-in 유지 또는 revert를 수치 근거와 함께 기록한다.

Energy와 gradient가 서로 다른 target integrator를 사용하는 configuration은 허용하지
않는다. Diagnostic 목적의 비교는 별도 script에서만 수행하고 production driver에는
연결하지 않는다.

## 9. Progress Ledger

| Stage | State | 현재 evidence | 다음 행동 |
| --- | --- | --- | --- |
| MG0 capability/ownership | `complete` | PySCF 2.13.1/GPU4PySCF 1.7.4 version report, state fingerprint, owner tests; same-process RSDF `max|Delta D_HF|=3.3e-16` | 유지 |
| MG1 target XC-only | `complete, opt-in keep` | GDF/RSDF, Gamma/sampled-k, PBE/r2SCAN FD max `<3.3e-7 Ha/Bohr`; Water8 gradient median 24--26% speedup | default 승격 없이 explicit opt-in 유지 |
| MG2 target J+XC | `complete, diagnostic opt-in` | same correctness matrix FD max `<3.5e-7`; selected-DF target dJ disabled and owner test 통과 | finite-auxbasis GDF/RSDF와 다른 target-J model임을 명시 |
| MG3 translation/performance | `complete` | Water8 3회 timing, Water20 no-FD scaling, x/y/z phase scan, host RSS/VRAM/task metadata 기록 | evidence 유지 |
| MG4 PP-only | `deferred by profile` | Water20 one-electron 약 1.8 s / M1 total 12.8--16.7 s; PP는 그 일부 | 5% end-to-end 이득 근거가 생길 때만 재개 |

### 2026-07-17 implementation decision

- Production default는 기존 `uniform`을 유지한다.
- `multigrid_xc`는 source GDF/RSDF와 target J를 그대로 두고 target XC만 교체하는
  검증된 explicit opt-in route다.
- `multigrid_j_xc`는 analytic derivative가 맞지만 finite-auxbasis target J와 다른
  reciprocal Coulomb model을 사용한다. Water8/SZV-GTH에서 M1과 약 3.10 Ha 차이가
  있으므로 일반적인 GDF/RSDF drop-in replacement가 아니라 diagnostic opt-in이다.
- Water8 phase-scan peak-to-peak amplitude는 uniform과 multigrid XC가 모두 x/y에서
  약 `1.92e-4 Ha`, z에서 약 `2.52e-4 Ha`다. Multigrid는 속도는 개선하지만 현재
  egg-box를 체계적으로 줄이지 않는다.
- 상세 수식, 소유권 및 benchmark 수치는
  `docs/implementation/13_target_multigrid.md`에 고정한다.

## 10. 완료 정의

이 goal은 다음 조건을 모두 만족할 때 완료한다.

1. GDF와 RSDF source HF가 target-integrator 선택과 독립적으로 같은 density를 만든다.
2. M1과 M2의 energy/Fock/gradient ownership이 metadata와 test에서 명시적이다.
3. GDF/RSDF, Gamma/sampled-k, PBE/r2SCAN analytic-FD gate가 통과한다.
4. Target energy와 gradient가 동일한 multigrid mesh/task contract를 사용한다.
5. Translation phase amplitude와 complete-gradient mesh convergence가 기록된다.
6. Water8/Water20 반복 timing과 host/device memory가 U0와 비교된다.
7. Silent CPU fallback, PP/J double counting, dropped k-point normalization이 없다.
8. 성능 또는 수치적 이득이 없는 route는 default로 승격하지 않고 근거와 함께
   opt-in 또는 revert 상태로 남긴다.
