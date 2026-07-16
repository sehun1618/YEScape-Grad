# GTH pseudopotential derivative

## 책임

`pseudo/`와 `grad/one_electron.py`는 Eq. `gth-hamiltonian`의 kinetic, local GTH,
nonlocal GTH matrix와 overlap의 fixed-cell nuclear derivative를 제공한다. Canonical
output은 full AO matrix derivative이며 단위는 Ha/Bohr다.

## Local과 nonlocal 분리

One-electron derivative는

$$
h^{Ax}=T^{Ax}+V_{\mathrm{loc}}^{Ax}+V_{\mathrm{nl}}^{Ax}
$$

로 나눈다. Local part에는 GTH potential structure factor와 AO center response가,
nonlocal part에는 projector-center 및 AO/projector overlap response가 포함된다.

## Public/internal owner

| Backend | Local/kinetic/overlap | Nonlocal GTH |
| --- | --- | --- |
| CPU FFTDF | `grad/one_electron.py` | `pseudo/fft_nonlocal_derivative.py` |
| CPU GDF/RSDF | `grad/gdf_one_electron.py` bridge | `pseudo/nonlocal_derivative.py` |
| GPU FFTDF | `grad/one_electron.py` | `pseudo/nonlocal_derivative.py` 또는 FFT provider |
| GPU GDF/RSDF | `grad/gdf_one_electron.py` host primitive bridge | GPU contraction + explicit host boundary |

## Shape와 convention

- Full hcore derivative: `(natm, 3, nkpts, nao, nao)`
- AO-row derivative: `(3, nkpts, nao_atom, nao)`
- Full overlap derivative: `(natm, 3, nkpts, nao, nao)`
- Gamma matrix도 singleton k-axis를 내부적으로 유지한다.
- Reverse AO-center term은 complex conjugate transpose를 사용한다.

AO-row derivative $X_A^x$에서 full matrix를 만들 때

$$
X^{Ax}=X_A^x+X_A^{x\dagger}+X_{\mathrm{operator},A}^x
$$

를 따른다. Operator-center term이 이미 row/column translation identity에 포함됐는지
provider별로 문서화해야 하며 중복 합산하지 않는다.

## $G=0$와 Ewald

Local GTH reciprocal representation의 $G=0$ constant, ionic charge와 Ewald nuclear
repulsion은 하나의 convention이다. `cell.energy_nuc()` 및 PySCF gradient routine과 같은
GTH ionic charge를 사용한다. $G=0$을 무조건 0으로 두는 것과 neutral convention을
적용하는 것은 같은 말이 아니다.

## CPU/GPU boundary

GPU4PySCF가 제공하지 않는 small radial/projector table과 periodic integral optimizer
setup은 host에서 수행할 수 있다. 그러나 projector contraction과 최종 electronic
gradient는 지원된 GPU route에서 CuPy로 수행하며 CPU package의 gradient를 호출하지
않는다.

## 검증 순서

1. Kinetic/overlap matrix derivative를 직접 AO integral FD와 비교한다.
2. Local GTH matrix derivative를 atom/axis별로 비교한다.
3. Nonlocal projector derivative를 angular channel별로 비교한다.
4. Full hcore derivative와 one-electron energy contraction을 비교한다.
5. Gamma와 sampled k-point conjugation 및 translation sum을 검사한다.
6. FFTDF와 GDF/RSDF hcore value가 같은 pseudopotential convention인지 확인한다.

Basis와 pseudopotential parameter를 source와 target 사이에서 바꾸면 같은 energy surface의
gradient가 아니므로 허용하지 않는다.
