# Equation-to-code map

이 문서는 theory equation과 production source owner를 연결한다. 수식은
`docs/theory/PBC_DC_DFT_Gradient_Theory.tex`의 label을 기준으로 하며, 경로는 각
distribution의 `src` directory에 상대적이다. 표에 없는 experimental 또는 archived
route는 active public API가 아니다.

## 공통 DC-DFT 및 response

| Theory equation | 계산량 | CPU owner | GPU owner | Shape / unit | Validation |
| --- | --- | --- | --- | --- | --- |
| $E_{\mathrm{DFA}}[D^{\mathrm{HF}}]$, Eq. `eq:dc-energy-components` | fixed-density target energy | `dft/fixed_density.py::evaluate_energy` | `dft/fixed_density.py::evaluate_energy` | scalar, Ha | FFTDF/GDF/RSDF full-energy FD |
| $g_{ai\mathbf{k}}$, Eq. `eq:target-orbital-gradient` | target-Fock VO source | `grad/orbitals.py::target_fock_vo_blocks` | `grad/orbitals.py::target_fock_vo_blocks` | per-$\mathbf{k}$ $(n_v,n_o)$, Ha | orbital-rotation FD |
| $\mathcal A^{\mathrm{HF}}X$, Eq. `eq:hf-hessian-action` | HF orbital Hessian action | `grad/response.py::HFOrbitalHessian` | `grad/response.py::HFOrbitalHessian` | realified VO vector, Ha | linear-action and symmetry tests |
| $A_R^{T}z_R=-g_R$, Eq. `eq:zvector-equation` | adjoint Z-vector | `grad/zvector.py::solve_z_vector` | `grad/zvector.py::solve_z_vector` (`auto`: small dense, production matrix-free MINRES) | realified VO vector | residual $<10^{-8}$, dense/matrix-free Gamma/k parity |
| $B^{A_x}_{ai\mathbf{k}}$, Eq. `eq:cphf-source` | nuclear response source | `grad/nuclear_response.py::nuclear_response_source_vo` | `grad/nuclear_response.py::nuclear_response_source_vo_sets` + `grad/response.py::apply_hf_response_sets` | $(N_{\mathrm{atom}},3,n_v,n_o)$, Ha/Bohr, memory-bounded component batch | source-matrix FD and component/batch parity |
| $E^{A_x}_{\mathrm{resp}}$, Eq. `eq:response-gradient` | Z-vector response contraction | `grad/assembly.py::assemble_gradient` | backend gradient assembly | $(N_{\mathrm{atom}},3)$, Ha/Bohr | full-energy FD |
| $dE/dR_{A_x}$, Eq. `eq:final-gradient-assembly` | total nuclear gradient | `grad/fftdf.py`, `grad/gdf.py`, `grad/rsdf.py` | same paths in GPU package | $(N_{\mathrm{atom}},3)$, Ha/Bohr | Cartesian central FD |

## One-electron 및 GTH pseudopotential

| Theory equation | 계산량 | CPU owner | GPU owner | Shape / unit | Validation |
| --- | --- | --- | --- | --- | --- |
| $h^{A_x}_{\mathbf{k}}$, Eq. `eq:hcore-full-derivative` | kinetic + GTH local/nonlocal H1 | `grad/one_electron.py::build_one_electron_derivatives` | `grad/one_electron.py::build_one_electron_derivatives` | $(N_{\mathrm{atom}},3,N_k,n_{AO},n_{AO})$, Ha/Bohr | matrix FD and full-energy FD |
| $\partial V_{\mathrm{NL}}/\partial R_{A_x}$, Eq. `eq:gth-nonlocal-derivative` | FFT finite-mesh nonlocal GTH derivative | `pseudo/fft_nonlocal_derivative.py::fft_nonlocal_gth_derivative` | `pseudo/nonlocal_derivative.py::nonlocal_gth_derivative` | H1 matrix, Ha/Bohr | FFTDF matrix FD |
| $\partial V_{\mathrm{NL}}/\partial R_{A_x}$, Eq. `eq:gth-nonlocal-derivative` | exact-integral GDF/RSDF nonlocal derivative | `pseudo/nonlocal_derivative.py::nonlocal_gth_derivative` | explicit host integral boundary | H1 matrix, Ha/Bohr | GDF/RSDF matrix FD |
| $-\operatorname{Tr}(D^{E,\mathrm{target}}S^{A_x})$, Eq. `eq:target-overlap-pulay` | target overlap/Pulay term | `grad/pulay.py::contract_target_overlap` | `grad/pulay.py::contract_target_overlap` | $(N_{\mathrm{atom}},3)$, Ha/Bohr | component and full-energy FD |

## FFTDF

| Theory equation | 계산량 | CPU owner | GPU owner | Shape / unit | Validation |
| --- | --- | --- | --- | --- | --- |
| $\rho(\mathbf G)$, Eq. `eq:fftdf-density-fourier` | AO/pair density on reciprocal grid | `df/fft/grid.py` | `df/fft/grid.py` | complex grid array | CPU/GPU value parity |
| $dE_H/dR_{A_x}$, Eq. `eq:fftdf-hartree-derivative` | full Coulomb matrix derivative | `df/fft/jk_derivative.py::fft_jk_derivative` | `df/fft/jk_derivative.py::fft_jk_derivative` | $(N_{\mathrm{atom}},N_k,3,n_{AO},n_{AO})$, Ha/Bohr | Gamma/k-point matrix FD |
| $dE_x^{\mathrm{HF}}/dR_{A_x}$, Eqs. `eq:fftdf-exchange`, `eq:fftdf-ewald-exchange-derivative` | full exchange matrix derivative | `df/fft/jk_derivative.py::fft_jk_derivative` | `df/fft/jk_derivative.py::fft_jk_derivative` | same as above | Gamma/k-point matrix FD |
| $dE_{\mathrm{DC}}/dR_{A_x}$, Eq. `eq:final-gradient-assembly` | FFTDF total gradient | `grad/fftdf.py::fftdf_cpu_gradient` | `grad/fftdf.py::fftdf_gpu_gradient` | $(N_{\mathrm{atom}},3)$, Ha/Bohr | H2/H2O, SZV/DZVP, Gamma/k-point FD |

## GDF

| Theory equation | 계산량 | CPU owner | GPU owner | Shape / unit | Validation |
| --- | --- | --- | --- | --- | --- |
| $B_{P,\mu\nu}$, Eq. `eq:gdf-three-center` | three-center source | `df/gdf/source.py::three_center_source` | native GPU4PySCF CUDA RSDF builder; reference: `df/gdf/response.py` | compressed CDERI source | central J/K and energy parity |
| $M_{PQ}$, Eq. `eq:gdf-two-center` | two-center Coulomb metric | `df/gdf/metric.py` | native GPU4PySCF metric kernel; reference: `df/gdf/response.py` | $(n_{aux},n_{aux})$ conceptually | metric derivative and energy FD |
| $C=L^{-1}B$, Eq. `eq:gdf-cderi` | central CDERI | `df/gdf/cholesky.py` | `df/gdf/gdf.py::GDF.build` | compressed native storage or canonical reference blocks | J/K and energy parity |
| $dC=L^{-1}(dB-dL\,C)$, Eq. `eq:gdf-cderi-derivative` | source + metric auxiliary response | `df/gdf/cholesky.py::differentiate_cderi` | native source-contracted CUDA kernel; reference: `df/gdf/cholesky.py::CDERIFactorization.differentiate` | atom scalar derivative or reference tensor | direct CDERI/reference FD and full-energy FD |
| $dJ/dR$, $dK/dR$, Eqs. `eq:gdf-coulomb-aux-derivative`, `eq:gdf-j-derivative` | GDF J/K energy derivative | `df/gdf/jk_derivative.py::gdf_jk_derivative` | `df/gdf/contracted_derivative.py::{coulomb,exchange,restricted_hf}_energy_derivative` | $(N_{\mathrm{atom}},3)$, Ha/Bohr in native mode | Gamma/k-point algebra and full-energy FD |
| $Q(D+T_Z)-Q(D-T_Z)$ | bilinear HF two-electron Z response | full $dJ/dR,dK/dR$ contraction | `df/gdf/contracted_derivative.py::restricted_hf_bilinear_derivative` (현재 two-pass polarization) | $(N_{\mathrm{atom}},3)$, Ha/Bohr | full-tensor reference parity; signed one-pass kernel은 optimization track |
| $dE_{\mathrm{DC}}/dR_{A_x}$, Eq. `eq:final-gradient-assembly` | GDF total gradient | `grad/gdf.py::gdf_cpu_gradient` | `grad/gdf.py::gdf_gpu_gradient` | $(N_{\mathrm{atom}},3)$, Ha/Bohr | full-energy FD and CPU/GPU parity |

## MGGA: SCAN 및 r2SCAN

| Theory equation | 계산량 | CPU owner | GPU owner | Shape / unit | Validation |
| --- | --- | --- | --- | --- | --- |
| $\tau(\mathbf r)$, Eq. `eq:mgga-tau-ao` | kinetic-energy density | `dft/fixed_density.py` target NumInt | `dft/fixed_density.py` target KNumInt | real-space grid scalar | target energy parity |
| $V^{\tau}_{\mu\nu}$, Eq. `eq:mgga-vtau-matrix` | tau-dependent target Fock | PySCF NumInt route | GPU4PySCF KNumInt route | $(N_k,n_{AO},n_{AO})$, Ha | target-Fock parity |
| $dE_{xc}/dR_{A_x}$, Eqs. `eq:mgga-xc-derivative`--`eq:mgga-tau-derivative` | MGGA XC AO-row derivative | `grad/target.py::_target_mgga_xc_row_derivative` | `grad/target.py::target_xc_row_derivative` | atom gradient, Ha/Bohr | FFTDF/GDF/RSDF SCAN/r2SCAN Cartesian FD |
| $g^{\mathrm{MGGA}}_{ai\mathbf{k}}$, Eq. `eq:mgga-target-vo` | MGGA response RHS | `grad/orbitals.py::target_fock_vo_blocks` | same path | VO blocks, Ha | Z residual/full FD |

## RSDF

| Theory equation | 계산량 | CPU owner | GPU owner | Shape / unit | Validation |
| --- | --- | --- | --- | --- | --- |
| $(\omega_s,\mathcal M_s;\omega_m,\mathcal M_m)$, Eq. `eq:rsdf-independent-parameters` | independent source/metric settings | `df/rsdf/parameters.py::parameters_from_provider` | CPU provider metadata uploaded by `df/rsdf/rsdf.py` | immutable metadata | public provider build contract |
| $dB^{\mathrm{RS}}/dR$, Eq. `eq:rsdf-source-derivative` | screened three-center source response | `df/rsdf/source.py::rsdf_source_derivatives` | `df/rsdf/source_adjoint.py` + `source_native.py` direct contraction | source adjoint to $(N_{\mathrm{atom}},3)$ scalar | direct source FD, Gamma/k-point adjoint parity |
| $dM^{\mathrm{RS}}/dR$, Eq. `eq:rsdf-metric-derivative` | screened metric response | `df/rsdf/metric.py::rsdf_metric_derivatives` | exact host SR `ip1/ip2` + `df/rsdf/metric_adjoint.py`/`metric_native.py` reciprocal direct contraction | metric adjoint to $(N_{\mathrm{atom}},3)$ scalar | full $dM$ witness, Gamma/k-point parity |
| $dC^{\mathrm{RS}}/dR$, Eq. `eq:rsdf-cderi-derivative` | RSDF source + metric auxiliary response | `df/rsdf/response.py::build_rsdf_response_tensors` | `df/rsdf/response.py::build_rsdf_response_tensors` | atom-resolved CDERI derivative | direct CDERI FD/parity |
| $dE_J/dR$ and $z^T B_{J/K}^R$, Eq. `eq:rsdf-cderi-derivative` | production adjoint-contracted response | direct CPU gradient contraction | `df/rsdf/contracted_response.py::build_rsdf_contracted_response`; host opt-out memory policy: `component_stream.py` | two $(N_{\mathrm{atom}},3)$ scalars, Ha/Bohr | four ownership modes, full-tensor witness, batch-1/OOM, Gamma/k-point parity |
| $dE_{\mathrm{DC}}^{\mathrm{RSDF}}/dR$, Eq. `eq:rsdf-total-gradient` | RSDF total gradient | `grad/rsdf.py::rsdf_cpu_gradient` | `grad/rsdf.py::rsdf_gpu_gradient` | $(N_{\mathrm{atom}},3)$, Ha/Bohr | PBE/SCAN/r2SCAN, H2/H2O, SZV/DZVP, Gamma/k-point FD |

## Global conventions

- Equal-weight k-point sums use $N_k^{-1}\sum_{\mathbf k}$; see Eq.
  `eq:trace-convention`.
- Public return value is the gradient $dE/d\mathbf R$ in Ha/Bohr, not the force
  $-dE/d\mathbf R$.
- CPU/GPU parity uses Eqs. `eq:cpu-gpu-energy-error` and
  `eq:cpu-gpu-gradient-error`.
- Numerical derivative validation uses Eqs. `eq:central-fd`, `eq:fd-max-error`, and
  `eq:translation-error`.
