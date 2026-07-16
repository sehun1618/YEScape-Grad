# DC Lagrangian과 gradient assembly

## 책임

이 문서는 target energy $E_{\mathrm{DFA}}[D_{\mathrm{HF}}]$의 nuclear analytic
gradient를 backend-independent contribution으로 조립하는 방법을 설명한다. FFTDF, GDF,
RSDF는 $J/K$ value와 derivative tensor를 제공하고, 이 문서의 Lagrangian/Z-vector
contract는 공통이다.

관련 이론식은 Eq. `dc-lagrangian`, `dc-gradient-master`,
`explicit-gradient-decomposition`, `response-gradient`, `final-gradient-assembly`다.

## 전체 계산 순서

1. `RestrictedGradientContext.from_dc_method`가 converged source HF, target KS, density,
   $k$-points와 DF provider를 정규화한다.
2. `build_mo_partitions`가 각 $k$-point의 occupied/virtual coefficient와 energy를 만든다.
3. `target_fock_kpts`와 `target_fock_vo_blocks`가 target orbital gradient
   $g_{ai}^{\mathbf k}=F_{\mathrm t,ai}^{\mathbf k}$를 만든다.
4. `solve_z_vector`가 $A_R^Tz_R=-g_R$을 푼다.
5. Backend가 one-electron, $J/K$, XC와 overlap derivative를 만든다.
6. `nuclear_response_source_vo`가 변위별 HF CPHF source $B^{Ax}$를 VO 공간에 투영한다.
7. `contract_z_vector`가 $z^TB^{Ax}$를 계산한다.
8. `GradientContributions.assemble`이 explicit와 response를 더한다.

## Canonical contribution

Final gradient는

$$
g_{Ax}=g_{Ax}^{1e}+g_{Ax}^{\mathrm H}+g_{Ax}^{\mathrm{xc}}
       +g_{Ax}^{\mathrm{Pulay,t}}+g_{Ax}^{\mathrm{NN}}
       +g_{Ax}^{\mathrm{response}}
$$

이다. `GradientContributions`는 최소한 `explicit`, `response`, `total`을 보존하며
`total = explicit + response`를 생성 시 검증한다.

## Tensor contract

| Quantity | Shape | Dtype | Unit/device |
| --- | --- | --- | --- |
| Density $D^{\mathbf k}$ | `(nkpts, nao, nao)` | real/complex | CPU NumPy 또는 GPU CuPy |
| Target Fock | `(nkpts, nao, nao)` | real/complex | source와 같은 device |
| Target VO RHS | k-point별 `(nvir, nocc)` | real/complex | source와 같은 device |
| $h^{Ax,\mathbf k}$ | `(natm, 3, nkpts, nao, nao)` | real/complex | Ha/Bohr |
| $S^{Ax,\mathbf k}$ | same | real/complex | Bohr$^{-1}$ |
| Explicit gradient | `(natm, 3)` | real | Ha/Bohr |
| Response gradient | `(natm, 3)` | real | Ha/Bohr |

Gamma에서도 내부 k-axis를 유지하고 public output만 `(natm, 3)`으로 축약한다.

## Factor와 sign

- Restricted density가 occupation factor $2$를 포함하는지 확인한다.
- Equal-weight k-mesh contraction에는 $1/N_k$를 정확히 한 번 적용한다.
- AO electronic-coordinate derivative를 nuclear-center derivative로 바꿀 때 sign을
  확인한다.
- Target overlap term은 $-\operatorname{Tr}[W_{\mathrm t}S^{Ax}]/N_k$다.
- Public result는 $\partial E/\partial R$, force는 그 음수다.
- Complex scalar는 모든 Hermitian partner를 합친 뒤 실수부를 취한다.

## Source owner

| Physical responsibility | CPU | GPU |
| --- | --- | --- |
| Context | `grad/context.py` | `grad/context.py` |
| MO partition/RHS | `grad/orbitals.py` | `grad/orbitals.py` |
| HF AX | `grad/response.py` | `grad/response.py` |
| Z-vector | `grad/zvector.py` | `grad/zvector.py` |
| Target overlap | `grad/pulay.py` | `grad/pulay.py` |
| Nuclear response source | `grad/nuclear_response.py` | `grad/nuclear_response.py` |
| Final assembly | `grad/assembly.py` | CPU `GradientContributions` contract |

## Failure policy

Source HF가 수렴하지 않았거나 target energy가 평가되지 않은 object에서 gradient를
호출하면 명시적으로 실패한다. 지원되지 않는 DF backend, hybrid target XC, open-shell
state도 silent fallback 없이 `NotImplementedError`를 발생시킨다.

## 검증

- `tests/test_gradient_assembly.py`
- `tests/test_zvector_algebra.py`
- `tests/test_fftdf_gradient.py`
- `tests/test_gdf_gradient.py`
- `tests/test_rsdf_gradient.py`
- GPU의 대응 `test_*_gpu.py`

Full completion evidence는 H2와 distorted H2O의 모든 Cartesian component에 대한
full-energy central FD다.
