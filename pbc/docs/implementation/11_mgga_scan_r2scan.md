# SCAN/r2SCAN target extension

## 1. Scope

restricted RHF/KRHF density에서 pure semilocal MGGA target인 SCAN과 r2SCAN energy 및
analytic nuclear gradient를 계산한다. FFTDF/GDF/RSDF, Gamma/sampled k-point, CPU/GPU를
지원한다. source density는 여전히 HF이며 target KS orbital optimization은 수행하지
않는다.

## 2. 새로 필요한 물리량

PBE의 GGA density vector는 $\rho$와 세 Cartesian derivative를 가진다. MGGA는
kinetic-energy density를 다섯 번째 component로 추가한다.

$$
\tau(\mathbf r)=\frac{1}{2N_k}\sum_{k\mu\nu}D^k_{\mu\nu}
\nabla\phi^*_{\mu k}\cdot\nabla\phi_{\nu k}.
$$

libxc가 반환하는 $v_\tau=\partial f_{xc}/\partial\tau$에는 PySCF의 tau 정의에 따른
factor `1/2`를 적용한다. nuclear AO-row derivative는 AO first derivative에 가중치를
곱한 뒤 second derivative와 contraction한다.

```text
wv[0] *= 0.5  # existing GGA bra/ket convention
wv[4] *= 0.5  # tau definition
GGA row term  = _gga_grad_sum_(ao, wv[:4])
tau row term  = _tau_grad_dot_(ao, wv[4])
```

CPU owner는 `pyscf_pbc_dcdft.grad.target::_target_mgga_xc_row_derivative`, GPU owner는
`gpu4pyscf_pbc_dcdft.grad.target::target_xc_row_derivative`이다. CPU upstream periodic
`krks.get_vxc`는 metaGGA에서 `NotImplementedError`를 내므로 CPU tau row derivative를
package가 직접 구현한다. GPU는 GPU4PySCF의 `_tau_grad_dot_` kernel을 재사용한다.

## 3. DC response에서 재사용되는 항

source HF Hamiltonian과 Hessian은 target functional과 무관하다. 따라서 다음 항은 PBE
route를 그대로 사용한다.

- HF SCF density와 MO partition
- FFTDF/GDF/RSDF J/K와 nuclear derivatives
- GTH one-electron/pseudopotential H1
- overlap/Pulay tensor 구조
- HF AX와 adjoint Z-vector solver

target functional에 따라 바뀌는 항은 target Fock/RHS와 explicit XC row derivative이다.
target MGGA Fock에 tau operator가 이미 포함되므로 target-Fock energy-weighted density와
Z-vector RHS도 일관되게 바뀐다.

## 4. Tensor and device contract

| Quantity | Shape | CPU | GPU |
| --- | --- | --- | --- |
| MGGA density | `(5,ngrids)` | NumPy | CuPy |
| AO through second derivative | `(nk,10,ngrids,nao)` | NumPy blocks | CuPy blocks |
| XC row derivative | `(3,nk,nao,nao)` | NumPy | CuPy |
| contracted target XC | `(natm,3)` | NumPy | CuPy |

uniform grid는 block loop로 처리하므로 전체 `ngrids x nao` tensor를 한 번에 저장하지
않는다. GPU route의 tau, XC row, atom contraction은 device-owned operation이다.

## 5. Validation

Full Cartesian evidence:
`pbc/gpu/validation/results/2026-07-14_mgga_gpu.json`.

| Case | GPU analytic-FD | CPU analytic-FD | CPU-GPU energy | CPU-GPU gradient |
| --- | ---: | ---: | ---: | ---: |
| H2 SCAN Gamma FFTDF SZV | $8.299\times10^{-8}$ | $8.307\times10^{-8}$ | $3.592\times10^{-11}$ | $7.431\times10^{-11}$ |
| H2 r2SCAN k211 GDF DZVP | $5.612\times10^{-8}$ | $5.567\times10^{-8}$ | $6.869\times10^{-12}$ | $4.548\times10^{-10}$ |
| H2O SCAN Gamma GDF SZV | $7.499\times10^{-7}$ | $7.499\times10^{-7}$ | $1.200\times10^{-11}$ | $1.553\times10^{-10}$ |
| H2O r2SCAN k211 FFTDF SZV | $8.787\times10^{-6}$ | $8.788\times10^{-6}$ | $1.037\times10^{-11}$ | $1.728\times10^{-10}$ |
| H2 SCAN Gamma RSDF SZV | $6.790\times10^{-8}$ | $6.790\times10^{-8}$ | $3.784\times10^{-11}$ | $2.471\times10^{-12}$ |
| H2 r2SCAN k211 RSDF DZVP | $5.595\times10^{-8}$ | $5.553\times10^{-8}$ | $6.915\times10^{-12}$ | $4.202\times10^{-10}$ |
| H2O SCAN Gamma RSDF SZV | $7.396\times10^{-7}$ | $7.395\times10^{-7}$ | $1.200\times10^{-11}$ | $5.438\times10^{-11}$ |
| H2O r2SCAN k211 RSDF SZV | $3.197\times10^{-7}$ | $3.198\times10^{-7}$ | $1.627\times10^{-12}$ | $3.748\times10^{-10}$ |

이 matrix는 MGGA target evaluation이 source-HF DF provider와 독립적으로 조합되며,
RSDF source/metric auxiliary response와 MGGA $\tau$ derivative가 동시에 포함될 때도 같은
full-energy FD를 재현함을 확인한다.

SCAN translation evidence:
`pbc/gpu/validation/results/2026-07-14_scan_grid_translation_gpu.json`.
non-XC net norm은 mesh $17^3$에서 $2.09\times10^{-10}$ Ha/Bohr이고, target-XC net norm은
$501^3$에서 $8.11\times10^{-8}$ Ha/Bohr이다. 중간 mesh에서는 aliasing 때문에 수렴이 단조롭지
않다. SCAN의 tau channel은 PBE보다 uniform-grid convergence가 더 엄격하므로 production
cutoff/mesh convergence를 반드시 별도로 확인해야 한다.

## 6. Known boundaries

- pure semilocal LDA/GGA/MGGA만 지원하며 hybrid MGGA와 NLC는 범위 밖이다.
- SCAN/r2SCAN은 PySCF의 `with_lapl=False` convention을 사용한다.
- fixed uniform grid의 egg-box error는 analytic formula 누락이 아니며 같은 finite-grid
  energy의 FD와 일치한 뒤 mesh convergence로 제거한다.
- fixed-cell nuclear gradient만 active 범위이며 cell deformation response는 archive에
  분리되어 있다.
