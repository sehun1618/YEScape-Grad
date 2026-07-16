# RSDF CPU/GPU restricted DC-DFT analytic gradient

## 1. Scope and public API

RSDF는 Coulomb kernel을 real-space short-range와 reciprocal-space long-range로
분리하여 Gaussian density fitting을 계산한다. 이 package의 RSDF route는 restricted
RHF/KRHF source density와 PBE, SCAN, r2SCAN target에 대해 Gamma 및 sampled k-point
energy와 fixed-cell nuclear analytic gradient를 지원한다.

CPU에서는 PySCF-style modifier를 사용한다.

```python
mf = dft.DCKRKS(cell, kpts=kpts, xc="PBE")
mf.rs_density_fit(auxbasis="gth-szv")
energy = mf.kernel()
gradient = mf.nuc_grad_method().kernel()  # dE/dR, Ha/Bohr
```

GPU package도 같은 API와 계산 의미를 유지한다. GPU route는 CPU DC-DFT energy나
gradient routine으로 silent fallback하지 않는다. 다만 GPU4PySCF에 없는 periodic
two-/three-center integral generation은 capability report에 기록되는 host boundary이다.

## 2. GDF와 RSDF의 공통점과 차이

한 k-point pair `(ki, kj)`와 momentum transfer `q = kj - ki`에 대해 auxiliary
source와 metric을

$$
B^{ij}_{\mu\nu,P}=(\mu k_i,\nu k_j|Pq),
\qquad
M_{PQ}(q)=(Pq|Qq)
$$

로 둔다. Cholesky factor와 canonical CDERI는

$$
M(q)=L(q)L(q)^\dagger,
\qquad
C^{ij}=L(q)^{-1}B^{ij}
$$

이다. 이 algebra는 GDF와 RSDF에 공통이다. 차이는 `B`와 `M`을 생성하는 numerical
contract에 있다.

Ordinary project GDF는 공통 range-separation contract를 project에서 명시적으로
재구성한다. Public PySCF RSDF는 source와 metric에 서로 독립적인 parameter를
허용한다.

$$
B = B_{\rm SR}(\omega_B,\epsilon_R)
  + B_{\rm LR}(\omega_B,\mathcal G_B)
  + \delta_{q0}B_{vbar},
$$

$$
M = M_{\rm SR}(\omega_M,\epsilon_M)
  + M_{\rm LR}(\omega_M,\mathcal G_M)
  + \delta_{q0}M_{G=0}.
$$

따라서 일반적으로

$$
\omega_B \ne \omega_M,
\qquad
\mathcal G_B \ne \mathcal G_M.
$$

이를 같은 값으로 가정하면 continuum limit에서는 비슷해 보여도 실제 SCF가 사용한
finite RSDF CDERI의 derivative가 아니다. H2O k-point test에서 stored CDERI와 ordinary
GDF convention의 차이는 약 $7.2\times10^{-6}$이었지만, exact RSDF source/metric contract는
$4.3\times10^{-9}$ 이하로 복원했다.

## 3. Resolved parameter contract

`df/rsdf/parameters.py::parameters_from_provider`는 central RSDF build 이후 다음
parameter를 immutable object로 고정한다.

| Field | Meaning |
| --- | --- |
| `source_omega` | three-center SR/LR split parameter |
| `source_mesh` | long-range source reciprocal mesh |
| `metric_omega` | two-center metric SR/LR split parameter |
| `metric_mesh` | long-range metric reciprocal mesh |
| `precision_r` | real-space three-center screening threshold |
| `precision_j2c` | two-center metric screening threshold |
| `use_bvk`, `bvk_kmesh` | optional Born-von Karman image grouping |

이 값들은 user가 임의로 추정하지 않고 PySCF `RSGDF.build()`가 실제로 결정한 값을
읽는다. `response_settings()`와 gradient metadata가 source/metric parameter를 따로
기록하므로 재현성 audit에서 둘을 확인할 수 있다.

## 4. Three-center source response

`df/rsdf/source.py`가 central `B`와 nuclear derivative를 소유한다. 한 atom `A`와
Cartesian axis `x`에 대해

$$
B^{A_x}=B_{\rm SR}^{A_x}+B_{\rm LR}^{A_x}
          +\delta_{q0}B_{vbar}^{A_x}.
$$

source에는 bra AO center, ket AO center, auxiliary center의 세 nuclear ownership이
있다. short-range `int3c2e_ip1/ip2`는 electronic-coordinate derivative이므로
atom-centered Gaussian 이동에는 반대 부호를 적용한다. non-Gamma에서는 complex
phase와 `(ki, kj)` orientation을 보존한다.

PySCF RSDF의 central short-range source는 `precision_R`과 optional BvK grouping을
사용한다. BvK는 lattice images의 평가 순서를 바꾸지만 물리적 center derivative 식은
바꾸지 않는다. 구현은 central-value parity guard로 해당 평가 convention이 저장된
CDERI와 일치함을 항상 확인한다.

Source derivative shape는

```text
(natm, 3, nao**2, naux)
```

이고 canonical CDERI orientation으로 변환한 뒤에는

```text
(natm, 3, naux, nao, nao)
```

이다.

## 5. Exact RSDF metric response

`df/rsdf/metric.py`는 public RSDF가 factorize한 metric 자체를 미분한다.

$$
M^{A_x}=M_{\rm SR}^{A_x}+M_{\rm LR}^{A_x}
         +\delta_{q0}M_{G=0}^{A_x}.
$$

가장 중요한 구현점은 short-range value와 derivative가 같은 image list와 screening
bounds를 사용해야 한다는 것이다. Value는 PySCF
`rsdf_helper.intor_j2c`가 `PBCsr2c_k_drv`로 계산한다. Project derivative는 같은
`_get_2c2e_Rcut`, `_get_atom_Rcuts_2c`, `_get_Lsmin` 결과를 사용하여
`int2c2e_ip1/ip2` driver를 호출한다. 따라서 다른 lattice truncation을 가진
continuum surrogate가 아니다.

Auxiliary atom `A`에 속하는 row/column block을 각각 선택하면

$$
(M_{\rm SR}^{A_x})_{PQ}
=-\delta_{P\in A}(M_{ip1,x})_{PQ}
 -\delta_{Q\in A}(M_{ip2,x})_{PQ}.
$$

metric derivative shape는 `(natm, 3, naux, naux)`이다. H2 validation에서 analytic
metric/factor response와 storage finite difference의 maximum difference는
$4.3\times10^{-10}$ 이하이다.

## 6. Cholesky and auxiliary-basis response

Central CDERI $C=L^{-1}B$를 미분하면

$$
C^{A_x}=L^{-1}B^{A_x}-L^{-1}L^{A_x}C.
$$

첫 항은 source response, 둘째 항은 metric response이다. Auxiliary basis가 nuclear
center와 함께 움직이므로 둘째 항은 선택 사항이 아니다. `df/gdf/cholesky.py`의
검증된 lower-Cholesky gauge를 RSDF도 재사용한다.

$$
W=L^{-1}M^{A_x}L^{-\dagger},
\qquad
L^{A_x}=L\Phi(W),
$$

여기서 `Phi`는 strict lower triangle을 보존하고 real diagonal의 절반을 취한다.
`RSDFResponseTensors`는 `source_dcderi`와 `metric_dcderi`를 따로 보존하여 auxiliary
response를 component-wise audit할 수 있게 한다.

PySCF HDF5에 저장된 central CDERI는 SCF energy의 authoritative value이다. Package는
독립적으로 `B`, `M`, $L^{-1}B$를 재구성하고 모든 canonical k-pair에서

$$
\max |C_{\rm reconstructed}-C_{\rm storage}| < 10^{-7}
$$

를 만족할 때만 derivative를 허용한다. 이 guard는 omega, mesh, auxiliary basis,
pair orientation, Cholesky gauge mismatch를 production gradient에 섞이지 않게 한다.

## 7. J/K, target DFA, and Z-vector assembly

$dC$가 준비된 뒤 RSDF는 GDF의 backend-independent post-CDERI algebra를 재사용한다.

1. $dJ/dR$과 $dK/dR$에 product rule을 적용한다.
2. target PBE/SCAN/r2SCAN explicit Hartree/XC derivative를 조립한다.
3. target-Fock overlap/Pulay term을 더한다.
4. HF orbital Hessian action과 adjoint Z-vector를 계산한다.
5. `z^T B_HF^R` response contribution을 더한다.
6. GTH one-electron derivative와 Ewald nuclear gradient를 더한다.

최종 식은

$$
E_{\rm DC}^{A_x}
=E_{\rm explicit,target}^{A_x}
+z^T B_{\rm HF}^{A_x}
$$

이며 반환값은 force가 아니라 $dE/dR$이다.

PySCF의 low-rank molecular-orbital K-build는 rank-one real transition density에서
빈 imaginary factor를 처리하지 못할 수 있다. RSDF response는 GDF와 마찬가지로
transition-density AX build에서 exact density-matrix J/K route를 강제하여 이 upstream
특수-case를 피한다. 이는 numerical approximation이 아니라 algebraically identical한
full-density contraction이다.

## 8. CPU and GPU ownership

### CPU

- central RSDF build/CDERI: PySCF `RSGDF`
- source/metric derivative primitives: `pyscf_pbc_dcdft.df.rsdf`
- Cholesky, dCDERI, J/K derivative: project NumPy/SciPy route
- target DFA, AX/Z-vector, final assembly: project CPU route

### GPU

Explicit host boundaries:

- `Cell`/`AuxCell`, lattice-image and integral optimizer metadata
- central RSDF two-/three-center integral and CDERI setup
- derivative source/metric and GTH H1 primitive generation
- Ewald nuclear gradient and final NumPy reporting

Device-owned operations:

- canonical CDERI storage after one-time upload
- source HF and target DFA J/K contractions
- metric Cholesky and `dL`/$dC$ response algebra
- full $dJ/dR$, $dK/dR$
- target XC contraction, HF AX/Z-vector, final electronic assembly

`silent_cpu_fallback`은 `False`이다. Host integral generation은 숨겨진 CPU DC-DFT
fallback이 아니라 GPU4PySCF에 해당 periodic derivative primitive가 없기 때문에
공개 metadata에 명시한 setup boundary이다.

## 9. Validation evidence

CPU full Cartesian central-FD gate:

| Case | max analytic-FD (Ha/Bohr) |
| --- | ---: |
| H2 Gamma SZV | $6.922\times10^{-8}$ |
| H2 k211 DZVP | $4.893\times10^{-8}$ |
| distorted H2O Gamma SZV | $1.476\times10^{-7}$ |
| distorted H2O k211 SZV | $3.989\times10^{-7}$ |

GPU full Cartesian central-FD 및 CPU parity gate:

| Case | GPU analytic-FD | CPU-GPU energy | CPU-GPU gradient |
| --- | ---: | ---: | ---: |
| H2 Gamma SZV | $6.931\times10^{-8}$ | $3.790\times10^{-11}$ Ha | $2.079\times10^{-12}$ Ha/Bohr |
| H2 k211 DZVP | $5.098\times10^{-8}$ | $6.951\times10^{-12}$ Ha | $4.044\times10^{-10}$ Ha/Bohr |
| distorted H2O Gamma SZV | $1.481\times10^{-7}$ | $5.130\times10^{-12}$ Ha | $5.031\times10^{-11}$ Ha/Bohr |
| distorted H2O k211 SZV | $3.119\times10^{-7}$ | $1.361\times10^{-12}$ Ha | $3.662\times10^{-10}$ Ha/Bohr |

Direct evidence:

```text
pbc/cpu/validation/results/2026-07-14_rsdf_cpu.json
pbc/gpu/validation/results/2026-07-14_rsdf_gpu.json
pbc/gpu/validation/results/2026-07-14_mgga_gpu.json
```

PBE 전용 RSDF gate에 더해 SCAN/r2SCAN과 결합한 4개 RSDF full Cartesian case도
통과했다. 이 결합 matrix의 maximum CPU analytic-FD 및 GPU analytic-FD error는 각각
$7.395\times10^{-7}$ 및 $7.396\times10^{-7}$ Ha/Bohr이다.

Regression gates는 CPU `39/39`, GPU `17/17`를 통과했다. Production RSDF response는
matrix finite difference를 호출하지 않는다.

## 10. Current limitations and failure policy

- fixed three-dimensional lattice의 nuclear gradient만 지원한다.
- stress/cell derivative는 아직 지원하지 않는다.
- restricted closed-shell RHF/KRHF source만 지원한다.
- `j2c_eig_always=True` eigen-metric factorization은 아직 analytic response가 없으므로
  `NotImplementedError`를 발생시킨다.
- grid/cutoff/auxiliary-basis convergence는 physical accuracy 문제이며 analytic-FD gate와
  별도로 확인해야 한다.
- 지원하지 않는 경로는 다른 backend로 silent fallback하지 않는다.
