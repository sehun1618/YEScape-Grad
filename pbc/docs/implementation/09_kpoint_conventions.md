# k-point와 complex tensor convention

## 좌표와 normalization

Public `kpts`는 Cartesian reciprocal coordinate in Bohr$^{-1}$다.
`lib/kpoints.py::normalize_kpts`는 Gamma 입력도 `(1, 3)`으로 바꾼다. Equal-weight mesh는

$$
\sum_{\mathbf k}w_{\mathbf k}=1,
\qquad w_{\mathbf k}=\frac{1}{N_k}
$$

를 사용한다.

## Canonical shape

| Quantity | Canonical shape |
| --- | --- |
| k-points | `(nkpts, 3)` |
| AO density/Fock/overlap | `(nkpts, nao, nao)` |
| MO coefficient | k-point별 `(nao, nmo)` |
| VO block | k-point별 `(nvir_k, nocc_k)` |
| Atom derivative matrix | `(natm, 3, nkpts, nao, nao)` |

Gamma-only public API가 `(nao, nao)`를 반환하더라도 backend boundary에서 k-axis를
복원한다. Singleton axis를 임의 `squeeze()`하지 않는다.

## Complex conjugation

Density와 Fock은 각 $k$에서 Hermitian이다. Canonical DF pair $(i,j)$의 reverse pair는

$$
C_{P,\nu\mu}^{ji}=C_{P,\mu\nu}^{ij*}
$$

로 복원한다. Derivative도 같은 relation을 따른다. `transpose`만 사용하고 `conj()`를
빠뜨리는 오류는 Gamma에서는 보이지 않을 수 있다.

## Momentum transfer

GDF/RSDF metric block과 FFT exchange kernel은

$$
\mathbf q=\mathbf k_j-\mathbf k_i
$$

로 분류한다. $q=0$은 Gamma-only가 아니라 모든 diagonal pair를 포함한다. Reciprocal
lattice vector만큼 다른 $q$도 같은 momentum class로 canonicalize한다.

## Real-linear Z-vector

Complex VO response $U$가 만드는 density에는 $U$와 $U^*$가 함께 나타난다. Solver는

$$
u_R=(\operatorname{Re}U,\operatorname{Im}U)^T
$$

를 사용하고 real-linear AX를 제공한다. Complex matrix에 일반 complex-linear Krylov
solver를 그대로 적용하지 않는다.

## 반드시 확인할 factor

- Energy, explicit derivative, AX, Z contraction의 $1/N_k$.
- Restricted density occupation factor $2$.
- Target Fock VO RHS의 real/imag packing 순서.
- Canonical/reverse pair의 AO index order.
- `exxdiv='ewald'`가 적용되는 $q=0$ pair.

## Tests

- `tests/test_kpoints.py`
- `tests/test_zvector_algebra.py`
- FFTDF/GDF/RSDF Gamma 및 non-Gamma gradient tests
- CPU/GPU parity tests
- Gamma object와 explicit one-k-point object의 energy/gradient equality
