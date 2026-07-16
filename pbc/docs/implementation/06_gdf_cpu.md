# GDF CPU restricted DC-PBE analytic gradient

## 1. 목적과 public route

이 문서는 `pyscf_pbc_dcdft.df.GDF`를 source HF와 target PBE에 함께 사용했을 때

$$
E_{\mathrm{DC-PBE}}(\mathbf R)
=E_{\mathrm{PBE}}[D_{\mathrm{HF}}(\mathbf R);\mathbf R]
$$

의 nuclear analytic gradient를 어떻게 계산하는지 설명한다. 사용자는 PySCF와 같은
방식으로 GDF를 선택한다.

```python
mf = dft.DCKRKS(cell, kpts=kpts, xc="PBE")
mf.density_fit(auxbasis="gth-szv")
energy = mf.kernel()
gradient = mf.nuc_grad_method().kernel()
```

반환되는 `gradient[A,x]`는 force가 아니라
$\partial E/\partial R_{Ax}$이며 단위는 Ha/Bohr이다. 현재 지원 범위는
restricted RHF/KRHF source, pure PBE target, GTH pseudopotential, Gamma 및 일반
equal-weight k-point mesh이다.

## 2. GDF factorization

한 k-point pair $(k_i,k_j)$와 momentum transfer
$q=k_j-k_i$에 대해 AO pair density를 auxiliary Gaussian basis로 fitting한다.
unorthogonalized three-center source와 two-center Coulomb metric을 각각

$$
B_{\mu\nu,P}^{ij}=(\mu k_i,\nu k_j|Pq),
\qquad
M_{PQ}(q)=(Pq|Qq)
$$

로 둔다. PySCF GDF가 저장하는 `CDERI`는 metric Cholesky factor

$$
M(q)=L(q)L(q)^\dagger
$$

를 사용하여 orthonormalize한

$$
C_{P,\mu\nu}^{ij}
=\sum_Q[L(q)^{-1}]_{PQ}B_{\mu\nu,Q}^{ij}
$$

이다. code shape는 다음과 같다.

| Quantity | Shape | Source owner |
| --- | --- | --- |
| `B` | `(nao**2, naux)` | `df/gdf/source.py` |
| `M` | `(naux, naux)` | `df/gdf/metric.py` |
| `C` | `(naux, nao, nao)` | `df/gdf/cholesky.py` |
| $dC$ | `(natm,3,naux,nao,nao)` | `df/gdf/pair_response.py` |

central `C`는 새로 계산한 값으로 SCF 결과를 바꾸지 않는다. 실제 HF가 사용한 PySCF
HDF5 CDERI를 `storage.py`에서 읽고, package가 독립 재구성한 `C`와 비교한다. 이
`value_parity_max_abs` guard가 $\omega$, mesh, pair orientation 또는 metric gauge가
다른 derivative를 잘못 붙이는 것을 막는다.

## 3. Range-separated periodic source와 metric

PySCF CPU GDF는 단순한 molecular 3-center integral이 아니라 periodic RSGDF builder를
사용한다. Coulomb kernel은 short-range와 long-range로 분리된다.

$$
\frac1r
=\frac{\operatorname{erfc}(\omega r)}r
+\frac{\operatorname{erf}(\omega r)}r.
$$

따라서 source와 metric도

$$
B=B^{\mathrm{SR}}+B^{\mathrm{LR}}+B^{vbar},
$$

$$
M=M^{\mathrm{SR}}+M^{\mathrm{LR}}+\delta_{q0}M^{G=0}
$$

로 구성한다.

- `source_short_range.py`: periodic 3-center real-space integral
- `source_long_range.py`: Fourier AO/auxiliary contraction
- `vbar.py`: zero-transfer charge correction
- `metric.py`: SR, LR, $G=0$ metric와 각 auxiliary-center derivative
- `pair_response.py::rsgdf_parameters`: PySCF `_RSGDFBuilder`와 같은 $\omega$, source
  mesh 및 별도의 metric mesh 결정

metric mesh는 source mesh와 같다고 가정하면 안 된다. auxiliary Gaussian은 primary
AO보다 sharp할 수 있고 $M^{-1}$ solve가 작은 metric error를 증폭하므로 PySCF도
auxiliary basis에서 더 엄격한 cutoff를 추정한다.

## 4. Auxiliary-basis response

nuclear displacement에서 primary AO center뿐 아니라 auxiliary Gaussian center도 함께
움직인다. 따라서 `B`만 미분하고 metric을 고정하면 fitted energy의 derivative가
아니다. 정확한 식은

$$
C=L^{-1}B,
$$

$$
C^{A_x}
=L^{-1}B^{A_x}-L^{-1}L^{A_x}C.
$$

첫 항은 source response, 둘째 항은 metric/auxiliary response이다.
`cholesky.py::differentiate_cderi`는 두 항을 `source`와 `metric`으로 따로 보존한다.

Cholesky derivative는

$$
M^{A_x}=L^{A_x}L^\dagger+L(L^{A_x})^\dagger
$$

에서 얻는다. reduced matrix

$$
W=L^{-1}M^{A_x}L^{-\dagger}
$$

를 만든 뒤 lower-Cholesky gauge에서 strict lower triangle은 그대로, real diagonal은
절반만 취한다. 이를 $\Phi(W)$라 하면

$$
L^{A_x}=L\Phi(W).
$$

metric response를 생략하면 작은 H2에서는 우연히 오차가 작을 수 있으나, 여러
auxiliary center가 섞이는 H2O나 larger system에서 큰 force error로 나타난다.

## 5. Nuclear-center ownership

### 5.1 Three-center source

$B^{A_x}$에는 세 center가 있다.

1. bra primary AO $\mu$가 atom A에 속할 때의 center derivative
2. ket primary AO $\nu$가 atom A에 속할 때의 center derivative
3. auxiliary function $P$가 atom A에 속할 때의 center derivative

short-range integral의 `ip1`/`ip2` 결과는 electron-coordinate derivative이므로 nuclear
translation에는 반대 부호가 붙는다. non-Gamma pair에서는 complex conjugation과 pair
orientation을 유지해야 한다.

### 5.2 Exact Gamma pair

PySCF RSGDF helper는 exact Gamma pair 하나만 전달되면 AO pair를 packed `s2`로
최적화한다. value에는 충분하지만 nuclear derivative에서는 $\mu$ center와
$\nu$ center ownership을 다시 나눌 수 없다. `source_short_range.py`는 exact Gamma
block과 사용하지 않는 non-Gamma diagonal block을 함께 요청하여 general-k `s1`
driver를 선택한다. 첫 block은 여전히 정확한 Gamma integral이며 small-k approximation이
아니다. 이 방식으로 `(nao,nao,naux)` ownership을 보존한다.

## 6. GDF J/K derivative

closed-shell density는 occupation factor 2를 이미 포함한다. Coulomb auxiliary density는

$$
j_P=\sum_k\sum_{\mu\nu}C^k_{P,\mu\nu}D^k_{\nu\mu}
$$

이고

$$
J^k_{\mu\nu}=\frac1{N_k}\sum_Pj_PC^k_{P,\mu\nu}.
$$

따라서 fixed-density derivative는 output factor와 auxiliary density를 모두 미분한다.

$$
(J^k)^{A_x}=\frac1{N_k}\sum_P
\left[j_P(C^k_P)^{A_x}+j_P^{A_x}C^k_P\right].
$$

exchange는

$$
K^{k_i}_{\mu\lambda}
=\frac1{N_k}\sum_{k_jP\nu\sigma}
C^{ij}_{P,\mu\nu}D^{k_j}_{\nu\sigma}
C^{ji}_{P,\sigma\lambda}
$$

이므로 두 CDERI factor에 각각 product rule을 적용한다. 구현은
`df/gdf/jk_derivative.py::gdf_jk_derivative`이며 결과 shape는
`(natm,nkpts,3,nao,nao)`이다. reverse pair는
`storage.py::derivative_pair_block`에서 Hermitian conjugate로 복원한다.

PySCF `exxdiv='ewald'`가 더하는

$$
K_{G=0}^k=mS_kD_kS_k
$$

도 overlap 양쪽을 미분한다. fixed lattice에서는 Madelung scalar $m$ 자체의 nuclear
derivative는 없다.

## 7. GDF-consistent GTH one-electron derivative

GDF energy의 local GTH potential은 uniform FFT representation이 아니라
`_RSNucBuilder`와 `_IntPPBuilder`를 사용한다. continuum limit에서 같은 operator라도
finite representation이 다른 value와 derivative를 섞으면 analytic-FD residual이 남는다.

- `local_part1_short_range.py`: `_RSNucBuilder` SR integral, diffuse restoration,
  overlap $G=0$ term
- `local_part1_long_range.py`: LR Fourier AO 및 pseudo-center derivative
- `local_part2.py`: `_IntPPBuilder` 3-center analytic derivative
- `pseudo/nonlocal_derivative.py`: exact GTH projector half-integral product rule
- `grad/gdf_one_electron.py`: kinetic, local part1/part2, nonlocal 및 overlap assembly

nonlocal potential은 angular channel별로

$$
V_{nl}=\sum_l I_l^\dagger h_l I_l
$$

이며

$$
V_{nl}^{A_x}=\sum_l
\left[(I_l^{A_x})^\dagger h_lI_l
+I_l^\dagger h_lI_l^{A_x}\right].
$$

PySCF `fake_cell_vnl`과 analytic `int1e_*_ip2` projector를 사용하므로 O의 p/d
channel과 이후 더 높은 angular momentum도 같은 normalization을 따른다.

## 8. DC-PBE gradient assembly

GDF-specific derivative가 준비된 뒤 DC nonvariational 구조는 FFTDF와 같다.

$$
E^{A_x}_{DC}
=E^{A_x}_{1e}
+E^{A_x}_{J,\mathrm{target}}
+E^{A_x}_{xc}
+E^{A_x}_{S,\mathrm{target}}
+E^{A_x}_{nuc}
+z^TB^{A_x}_{HF}.
$$

`grad/gdf.py::gdf_cpu_gradient` 호출 순서는 다음과 같다.

1. HF MO partition, target-Fock `vo` RHS, adjoint Z-vector
2. GDF-consistent one-electron 및 overlap derivative
3. 모든 canonical k-pair의 `C`와 $dC$
4. target $dJ$, source `d(J-K/2)`
5. uniform-grid PBE XC derivative와 target-Fock Pulay
6. Ewald nuclear gradient
7. Z-vector와 HF nuclear-response source contraction

`GDFGradientResult.metadata()`는 $\omega$, source/metric mesh, CDERI parity, Z residual,
condition number 및 단계별 wall time을 보존한다.

## 9. Uniform XC grid와 translation gate

현재 target PBE는 fixed uniform grid를 사용한다. 이 선택은 PySCF periodic gradient의
AO-row derivative와 정확히 일치하므로 analytic gradient는 같은 finite-grid energy의 FD와
일치한다. 그러나 finite mesh energy 자체에는 egg-box error가 있어

$$
\left\|\sum_A\mathbf g_A\right\|\ne0
$$

일 수 있다. distorted H2O에서 non-XC term의 atom sum은 numerical zero이고 net error는
target XC가 소유한다. mesh를 $17^3$에서 $225^3$으로 높이면 XC net norm은 약
$1.01\times10^{-1}$에서 $5.55\times10^{-7}$ Ha/Bohr로 감소한다. 중간 값은 aliasing 때문에 단조롭지 않다.

periodic `BeckeGrids`를 사용하면 grid가 atom과 함께 움직이므로 별도 weight/coordinate
response가 필요하다. 현재 upstream PySCF KRKS도 `grid_response=True`를 지원하지 않으므로
이 route에서는 Becke energy와 fixed-grid analytic formula를 섞지 않는다.

## 10. 검증과 실행

Unit test:

```bash
PYTHONPATH=src python -m unittest tests.test_gdf_algebra \
  tests.test_gdf_gradient -v
```

모든 Cartesian analytic-FD gate:

```bash
PYTHONPATH=src python validation/validate_gdf_cpu.py --full \
  --output validation/results/gdf_cpu.json
```

translation/cutoff gate:

```bash
PYTHONPATH=src python validation/validate_xc_grid_translation.py \
  --output validation/results/gdf_xc_grid_translation.json
```

사용 예제는 `examples/04_gdf_cpu_energy_gradient.py`이다. validation의 FD는 구현 일부가
아니며 production gradient에는 displaced geometry가 전혀 사용되지 않는다.
