# Orbital response and Z-vector

## 1. 왜 response가 필요한가

HF source orbital은 HF energy에는 stationary하지만 target DFA energy에는 일반적으로
stationary하지 않다. Target Fock의 virtual--occupied block

$$
g_{ai\mathbf k}
=\left(C_{v\mathbf k}^{\dagger}
F_{\mathrm{DFA}}[D_{\mathrm{HF}}]
C_{o\mathbf k}\right)_{ai}
$$

가 0이 아니므로 nuclear displacement가 만드는 HF orbital response가 DC gradient에
기여한다. Geometry component마다 CPHF를 직접 풀 수 있지만, adjoint Z-vector를 한 번
풀면 모든 $3N_{\mathrm{atom}}$ component에 재사용할 수 있다.

Theory references: `eq:target-orbital-gradient`, `eq:hf-hessian-action`,
`eq:cphf-source`, `eq:zvector-equation`, `eq:response-gradient`.

## 2. MO partition

`grad/orbitals.py::build_mo_partitions`는 각 k-point의 MO를 occupied와 virtual로 나눈다.

| Quantity | Shape per k-point | Unit |
| --- | --- | --- |
| $C_o$ | `(nao, nocc)` | dimensionless |
| $C_v$ | `(nao, nvir)` | dimensionless |
| $\epsilon_a-\epsilon_i$ | `(nvir, nocc)` | Ha |
| target $g_{ai}$ | `(nvir, nocc)` | Ha |

Restricted occupation은 `mo_occ > 0`을 occupied로, `mo_occ == 0`을 virtual로
정의한다. Occupied 또는 virtual space가 비어 있으면 response equation 자체가
정의되지 않으므로 명시적으로 실패한다.

`target_fock_ao_blocks`는 저장된 HF density에서 target hcore와 Veff를 재평가하고,
`target_fock_vo_blocks`가 $C_v^\dagger F_{\mathrm{DFA}}C_o$를 만든다.

## 3. Restricted HF Hessian action

Trial VO block $X_{\mathbf k}$로부터 spin-summed first-order density를 만든다.

$$
D_{\mathbf k}^{(1)}[X]
=C_{v\mathbf k}X_{\mathbf k}C_{o\mathbf k}^{\dagger}
+C_{o\mathbf k}X_{\mathbf k}^{\dagger}C_{v\mathbf k}^{\dagger}.
$$

PySCF `gen_response`가 반환하는

$$
G[D^{(1)}]=J[D^{(1)}]-\frac{1}{2}K[D^{(1)}]
$$

를 VO로 project한다. `HFOrbitalHessian.coupling`은 restricted spatial-orbital
convention의 factor 2를 적용하고, `HFOrbitalHessian.__call__`은 orbital-energy gap을
더한다.

$$
(\mathcal A X)_{ai\mathbf k}
=(\epsilon_{a\mathbf k}-\epsilon_{i\mathbf k})X_{ai\mathbf k}
+2\left(C_v^\dagger G[D^{(1)}[X]]C_o\right)_{ai\mathbf k}.
$$

GDF/RSDF transition density에서는 PySCF low-rank MO optimization이 empty imaginary
factor를 만들 수 있다. `_response_environment`는 algebraically identical
CDERI-density contraction을 선택해 representation-only failure를 피한다. 이는 CPU
DC-DFT fallback이 아니라 같은 provider 내부 J/K contraction 선택이다.

## 4. Complex k-point의 real-linear 표현

Hermitian density가 $X$와 $X^*$를 함께 포함하므로 $\mathcal A$는 일반적으로
complex-linear가 아니라 real-linear다. Flat complex vector $x$를

$$
x_R=
\begin{pmatrix}
\operatorname{Re}x\\
\operatorname{Im}x
\end{pmatrix}
$$

로 바꾼다. `BlockLayout`은 k-point별 VO shape와 offset을 보존하며 singleton Gamma
block도 같은 packing을 사용한다. Realified dimension은

$$
N_R=2\sum_{\mathbf k}n_{v\mathbf k}n_{o\mathbf k}
$$

이다.

GPU route는 동일한 `HFOrbitalHessian` action에 대해 두 solver를 제공한다.

- `dense`: 각 real basis vector에 operator를 적용해 $A_R$을 명시적으로 만든다. 작은
  reference 및 parity test용이다.
- `matrix_free`: $A_R$을 저장하지 않고 Krylov iteration이 요구하는
  $y=A_Rx$만 계산한다. Production route다.
- `auto`: 기본값이며 $N_R\le 256$에서만 dense를 쓰고, 그보다 크면 matrix-free로
  전환한다.

Dense matrix 한 개의 예상 저장량은 $8N_R^2/2^{30}$ GiB다. 따라서 $N_R=65{,}536$이면
행렬 하나만 32 GiB이며, matrix copy와 SVD workspace까지 고려하면 production에
부적합하다. `zvector_preflight` progress event와 `ZVectorResult.dense_matrix_gib`가 이
예상량을 실행 전에 보고한다.

## 5. Adjoint equation

Production equation은

$$
A_R^{T}z_R=-g_R
$$

이다. `solve_z_vector(..., equation="adjoint")`가 이 식을 풀며 `"direct"` option은
Hessian symmetry diagnostic에만 사용한다. Restricted HF Hessian은 real orbital-rotation
좌표에서 self-adjoint이므로 matrix-free route에서는 같은 operator action이 direct와
adjoint를 표현한다. Gamma와 sampled-k dense parity test가 이 convention을 보호한다.

Matrix-free solver는 orbital-energy gap

$$
M^{-1}_{ai\mathbf k,bj\mathbf k'}
=\frac{\delta_{ab}\delta_{ij}\delta_{\mathbf k\mathbf k'}}
{\epsilon_{a\mathbf k}-\epsilon_{i\mathbf k}}
$$

을 preconditioner로 사용하는 CuPy MINRES다. 기본 Krylov tolerance는 $10^{-12}$,
maximum cycle은 80이다. 첫 solve 뒤 실제 residual을 다시 계산하고, 필요하면 residual
equation을 최대 세 번 더 풀어 iterative refinement한다. 최종 residual은

$$
\lVert A_R^Tz_R+g_R\rVert_\infty
\le 10^{-8}\max(1,\lVert g_R\rVert_\infty)
$$

를 만족해야 하며, 넘으면 gradient를 반환하지 않는다.

반환되는 `ZVectorResult`는 다음을 포함한다.

| Field | Meaning |
| --- | --- |
| `blocks` | k-point별 complex $Z_{ai}$ |
| `residual_max_abs` | $\|A_R^Tz_R+g_R\|_\infty$ |
| `condition_number` | dense reference에서만 계산 가능한 exact condition number |
| `condition_number_status` | exact SVD 수행, memory/int32 위험으로 skip, matrix-free 미평가 등의 상태 |
| `matrix_asymmetry_max_abs` | dense reference의 $\|A_R-A_R^T\|_{\max}$ |
| `dimension` | realified matrix dimension |
| `solver`, `solver_reason` | 실제 선택된 route와 선택 이유 |
| `iterations` | matrix-free Krylov iteration 수 |
| `solver_wall_time_s` | Z-vector solve 전체 wall time |
| `mean_iteration_wall_time_s` | iteration당 평균 wall time |
| `dense_matrix_gib` | 같은 문제를 dense로 만들었을 때 행렬 한 개의 예상 GiB |

Condition number는 gradient 식의 필수 항이 아니다. Dense exact-SVD 진단이 legacy
cuSOLVER integer/workspace 한계나 device memory 부족으로 안전하지 않으면 이를 skip하고,
residual gate를 통과한 Z-vector와 gradient 계산은 계속한다.

Small residual만으로 전체 gradient가 맞는 것은 아니다. Explicit hcore/J/XC/Pulay
derivative가 빠져도 Z residual은 작을 수 있다.

## 6. Nuclear source와 final response

Atom $A$, Cartesian $x$의 nuclear perturbation source를 $B^{A_x}_{ai\mathbf k}$라
하면 response correction은

$$
E_{\mathrm{response}}^{A_x}
=\frac{4}{N_k}\operatorname{Re}
\sum_{\mathbf k ai}
Z_{ai\mathbf k}^{*}B_{ai\mathbf k}^{A_x}.
$$

Factor 4는 restricted occupation과 Hermitian VO/OV rotation convention을 함께 반영한다.
Nuclear source는 full hcore derivative, source HF J/K derivative, overlap derivative와
occupied-space metric response를 포함한다. AO-row H1을 full H1로 잘못 취급하거나
target Fock과 source HF Fock을 바꾸면 이 contraction이 틀어진다.

Native GPU GDF route는 $3N_{\mathrm{atom}}$ source를 component마다 별도 J/K 호출로
계산하지 않는다. `nuclear_response_source_vo_sets`가
`(nset,nkpts,nao,nao)` density batch를 만들고, `apply_hf_response_sets`가 같은 CDERI
traversal에서 여러 source를 처리한다. Batch size는 현재 free device memory의 5%와
component당 보수적인 work-array 12개를 기준으로 정한다. 이 정책은 tensor 크기가 커질
때 자동으로 batch를 줄이며, 물리식이나 component 순서는 바꾸지 않는다. 사용자는
`z_nuclear_batch_memory_fraction`으로 이 memory budget을 더 보수적으로 만들거나,
`z_nuclear_batch_max_components`로 memory-derived batch의 upper bound를 줄일 수 있다.
후자는 production default를 바꾸는 magic number가 아니라 bounded-memory parity와 제한된
memory 환경을 위한 명시적 cap이다.

Preflight는 component 수, $N_k$, $N_{AO}$, 계획 당시 free/total device memory,
component당 예상 byte, 선택된 batch size와 batch count를 기록한다. 실행 중 allocation이
실제로 OOM을 내면 실패한 범위를 commit하지 않고 batch를 절반으로 줄여 같은 component부터
다시 시작한다. Batch size 1에서도 OOM이면 component index와 함께 hard failure를 낸다.
따라서 silent CPU fallback, component drop 또는 truncated response는 허용되지 않는다.

각 성공 batch 뒤에는 device free memory와 CuPy memory-pool used/reserved bytes를
관측한다. Report의 값은 sampling 시점의 **observed high-water**이며 CUDA kernel 내부의
순간적인 exact peak라고 과장하지 않는다. `z_nuclear_response_batch_plan`과
`z_nuclear_response_batch_execution` metadata에서 계획, 실제 rebatch 횟수, 완료 component
수 및 관측 memory를 함께 확인할 수 있다.

## 7. CPU/GPU ownership

CPU reference는 NumPy dense real-linear matrix와 `numpy.linalg.solve`를 사용한다.
GPU는 같은 packing과 equation을 CuPy로 구현하며, 작은 문제의 dense reference와 큰
문제의 matrix-free MINRES를 제공한다. 각 Krylov iteration의 AX는 선택된
FFTDF/GDF/RSDF provider의 transition-density J/K를 사용한다. Periodic derivative
integral primitive가 host에서 왔더라도 AX, iterative solve와 response contraction이
device에서 수행됐는지 timing/ownership metadata로 확인한다.

## 8. Verification

- `tests/test_zvector_algebra.py`: complex/real packing, adjoint solve, residual
- `tests/test_gradient_assembly.py`: response factor와 final component sum
- `tests/test_fftdf_gradient.py`, `tests/test_gdf_gradient.py`,
  `tests/test_rsdf_gradient.py`: Gamma/k-point full-energy FD
- `pbc/gpu/tests/test_zvector_gpu.py`: route preflight, synthetic
  real-linear solve, dense/matrix-free Gamma 및 sampled-k parity
- `pbc/gpu/tests/test_nuclear_response_gpu.py`: memory-bounded nuclear
  batch policy, OOM retry/no-drop scheduler와 batched contraction
- `pbc/gpu/tests/test_gdf_gpu.py`: H2 Gamma 및 sampled-k에서 automatic
  batch와 one-component forced batch의 final-gradient parity
- GPU backend tests: CPU/GPU Z residual, energy와 gradient parity

Debug order는 target VO block, Hessian action on random trial, $A_R$ asymmetry,
condition number, Z residual, nuclear source projection, final response contraction 순서다.
