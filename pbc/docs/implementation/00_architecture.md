# Architecture

## 1. 두 distribution의 관계

`pyscf-pbc-dcdft`는 restricted periodic DC-DFT의 CPU reference package다. PySCF의
`Cell`, RHF/KRHF, RKS/KRKS와 periodic DF object를 사용한다.
`gpu4pyscf-pbc-dcdft`는 같은 public calculation contract를 유지하면서 FFT/grid,
J/K, CDERI response, XC, AX/Z-vector와 final contraction을 CuPy/GPU4PySCF route로
교체한다.

두 package는 다음 물리적 의미를 공유한다.

$$
E_{\mathrm{DC-DFA}}(\mathbf R)
=E_{\mathrm{DFA}}[D_{\mathrm{HF}}(\mathbf R);\mathbf R].
$$

GPU package는 CPU package의 public object를 변환할 수 있지만, production energy 또는
gradient를 CPU implementation에 통째로 위임하지 않는다. GPU4PySCF에 periodic
derivative integral primitive가 없으면 그 primitive만 explicit host boundary로
선언한다.

## 2. Object lifecycle

`DCRKS`는 Gamma-only driver이고 `DCKRKS`는 sampled-k-point driver다. 두 object의
`kernel()`은 다음 순서를 따른다.

```text
Cell + k-points + DF provider
            |
            v
RHF/KRHF source SCF -----> D_HF, C_HF, epsilon_HF
            |                            |
            |                            +---- HF Hessian / AX
            v                            |
non-SCF target RKS/KRKS                  +---- adjoint Z-vector
            |                            |
            +---- E_DFA[D_HF]            |
            +---- F_target[D_HF] VO -----+
                                         |
explicit hcore/J/XC/Pulay derivatives ---+
                                         v
                           dE_DC-DFA / dR
```

Target RKS/KRKS는 orbital optimization을 실행하지 않는다. HF density를 외부 density로
받아 target energy, effective potential과 target Fock만 평가한다. Gradient object는
energy kernel이 보존한 source HF와 target KS object를 사용하므로 서로 다른 mesh,
k-point 또는 DF provider를 몰래 다시 만들지 않는다.

## 3. Public provider 선택

기본 provider는 FFTDF다.

```python
mf = dft.DCKRKS(cell, kpts=kpts, xc="PBE")
mf.fft_density_fit(mesh=cell.mesh)
mf.density_fit(auxbasis="gth-szv")      # GDF
mf.rs_density_fit(auxbasis="gth-szv")   # RSDF
```

마지막 modifier가 `with_df`를 교체한다. Source HF SCF, target DFA energy/Fock,
CPHF action과 derivative route는 모두 이 provider를 따라야 한다. 작은 Gamma
calculation에서 PySCF가 in-core ERI를 자동 선택하도록 두면 energy와 derivative가 서로
다른 Hamiltonian을 미분할 수 있으므로 package reference builder가 provider ownership을
고정한다.

## 4. Module ownership

| Module | 책임 |
| --- | --- |
| `scf/reference.py` | RHF/KRHF 생성, DF ownership과 convergence contract |
| `dft/base.py` | public driver state와 PySCF-style modifiers |
| `dft/fixed_density.py` | $E_{\mathrm{DFA}}[D_{\mathrm{HF}}]$와 component metadata |
| `grad/context.py` | Gamma를 $N_k=1$로 정규화한 immutable tensor view |
| `grad/orbitals.py` | occupied/virtual partition과 target-Fock VO source |
| `grad/response.py` | restricted HF Hessian action |
| `grad/zvector.py` | real-linear adjoint solve와 diagnostics |
| `grad/one_electron.py`, `pseudo/` | kinetic, overlap, local/nonlocal GTH H1 |
| `grad/target.py`, `grad/pulay.py` | target J/XC와 target overlap/Pulay |
| `df/fft/` | reciprocal-grid FFTDF value와 full J/K derivative |
| `df/gdf/` | GDF source, metric, CDERI 및 auxiliary response |
| `df/rsdf/` | independent source/metric RSDF parameter와 screened response |
| `grad/fftdf.py`, `grad/gdf.py`, `grad/rsdf.py` | backend별 final assembly |
| `lib/` | k-point normalization, result dataclass와 units |

FFTDF, GDF와 RSDF를 한 함수의 긴 조건문으로 합치지 않는다. 공통 response algebra는
`grad/`에 두고 representation-specific derivative는 각 `df` subpackage가 소유한다.

## 5. Canonical tensor convention

내부 tensor는 Gamma에서도 k-axis를 유지한다.

| Quantity | Shape | Dtype | Unit |
| --- | --- | --- | --- |
| k-points | `(nkpts, 3)` | float64 | Bohr$^{-1}$ |
| AO density/Fock | `(nkpts, nao, nao)` | real/complex | dimensionless / Ha |
| MO coefficient | `(nkpts, nao, nmo)` | real/complex | dimensionless |
| VO response block | k-point별 `(nvir, nocc)` | real/complex | context dependent |
| full H1 matrix | `(natm, 3, nkpts, nao, nao)` | real/complex | Ha/Bohr |
| nuclear gradient | `(natm, 3)` | float64 | Ha/Bohr |

Equal-weight mesh는 $N_k^{-1}\sum_{\mathbf k}$를 사용한다. Full derivative matrix와
AO-row derivative를 구분하며, row derivative를 full matrix처럼 contraction해 factor
2를 중복 적용하지 않는다.

## 6. CPU/GPU memory boundary

GDF/RSDF의 periodic two-/three-center integral과 GTH derivative primitive는 현재
PySCF host integral engine이 생성한다. GPU package는 이 경계를 metadata와 timing에
기록한 뒤 다음 자료를 device에 올린다.

- central CDERI와 atom-resolved source/metric primitive
- density와 MO block
- FFT/grid AO value와 derivative
- Z-vector operator input과 final contraction tensor

`device="gpu"`는 모든 metadata 생성이 GPU라는 뜻이 아니다. Capability가
`True`인 numerical route가 CPU DC-DFT implementation으로 silent fallback하지 않는다는
뜻이다.

## 7. Failure policy

- HF가 수렴하지 않으면 target energy를 평가하지 않는다.
- unsupported XC, electronic state 또는 DF provider는 명시적으로 실패한다.
- central CDERI parity가 깨지면 그 derivative를 사용하지 않는다.
- source와 target의 cell, basis, pseudo, k-point 또는 DF provider를 섞지 않는다.
- validation FD와 molecule-specific correction을 production source에 넣지 않는다.

## 8. Active/deferred boundary

Active public API는 fixed-cell nuclear energy/gradient, PBE/SCAN/r2SCAN,
FFTDF/GDF/RSDF, CPU/GPU, Gamma/k-point까지다. Target-only multigrid, stress/cell
derivative, larger-angular-momentum 확장과 open-shell/UHF는 archive에 분리되어 있으며
active import graph에 포함되지 않는다.
