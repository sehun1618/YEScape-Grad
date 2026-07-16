# Fixed-density DC energy flow

## 1. 계산하는 energy

Source RHF/KRHF가 수렴해 만든 spin-summed density $D^{\mathrm{HF}}$에서 target
functional을 평가한다.

$$
E_{\mathrm{DC-DFA}}
=E_{1e}[D^{\mathrm{HF}}]
+E_H[D^{\mathrm{HF}}]
+E_{\mathrm{xc}}^{\mathrm{DFA}}[n_{\mathrm{HF}}]
+E_{\mathrm{NN}}.
$$

PBE, SCAN, r2SCAN 모두 같은 flow를 사용한다. 차이는 target XC quadrature가 GGA
variable까지만 쓰는지, $\tau$ channel까지 쓰는지에 있다. Target RKS/KRKS SCF는
실행하지 않으므로 이 energy는 target orbital variation에 대해 stationary하지 않다.

## 2. Public call flow

1. `DCRKS` 또는 `DCKRKS`가 source RHF/KRHF object를 만든다.
2. 사용자가 고른 `with_df` provider를 source HF에 명시적으로 연결한다.
3. `reference.kernel()`이 HF를 수렴시키고 `make_rdm1()`으로 AO density를 만든다.
4. `build_target_ks()`가 같은 cell, k-points, XC와 DF provider의 RKS/KRKS를 만든다.
5. Target SCF 없이 `get_hcore()`와 `get_veff(cell, dm_hf)`를 한 번 평가한다.
6. `energy_tot(dm=dm_hf, h1e=hcore, vhf=veff)`로 total energy를 얻는다.
7. Energy component와 source/target/provider metadata를 immutable result에 보존한다.

Implementation owner:

- public lifecycle: `dft/base.py::DCRestrictedBase.kernel`
- target construction: `dft/fixed_density.py::build_target_ks`
- component evaluation: `dft/fixed_density.py::evaluate_energy`

## 3. k-point normalization

Sampled k-point에서 one-electron component는

$$
E_{1e}
=\frac{1}{N_k}\sum_{\mathbf k}
\operatorname{Tr}\!\left[D_{\mathbf k}^{\mathrm{HF}}h_{\mathbf k}\right].
$$

반면 PySCF가 반환하는 `veff.ecoul`과 `veff.exc`에는 k-point weighting이 이미
포함된다. 이 두 scalar에 $1/N_k$를 다시 곱하면 안 된다. Gamma는 같은 식의
$N_k=1$ case로 취급한다.

## 4. Inputs와 outputs

| Quantity | Shape | Dtype | Device | Unit |
| --- | --- | --- | --- | --- |
| k-points | `(nkpts, 3)` | float64 | host metadata | Bohr$^{-1}$ |
| HF density | Gamma `(nao,nao)` 또는 k-point `(nkpts,nao,nao)` | real/complex | CPU/GPU route | dimensionless |
| hcore | density와 같은 matrix shape | real/complex | provider route | Ha |
| target Veff | density와 같은 matrix shape | real/complex | NumInt + DF route | Ha |
| total/reference energy | scalar | float64 | report host | Ha |

`DCEnergyComponents`는 one-electron, Coulomb, XC, nuclear, total을 보존한다.
`DCEnergyResult`는 target energy, source HF energy, target XC, k-point count와
provider metadata를 보존한다. Complex scalar의 imaginary residual이 tolerance를 넘으면
실수로 강제 변환하지 않고 오류를 발생시킨다.

## 5. DF ownership

Source HF와 target Hartree term은 같은 FFTDF/GDF/RSDF object를 공유한다. RSDF는
provider build에서 결정된 source $\omega$/mesh와 metric $\omega$/mesh를 metadata로
별도 보존한다. GDF/RSDF에서 target object가 새 CDERI convention을 독립적으로 만들면
energy는 비슷해 보여도 derivative response와 다른 representation이 될 수 있다.

작은 Gamma cell의 source HF도 `with_df`를 우회하는 in-core ERI path를 사용하지
않도록 reference builder가 J/K ownership을 고정한다.

## 6. MGGA energy path

SCAN/r2SCAN target은 density $n$, gradient invariant $\sigma$와 kinetic-energy
density $\tau$를 평가한다.

$$
\tau(\mathbf r)
=\frac{1}{2N_k}\sum_{\mathbf k\mu\nu}
D_{\mu\nu}^{\mathbf k}
\nabla\phi_{\mu\mathbf k}^{*}(\mathbf r)
\cdot\nabla\phi_{\nu\mathbf k}(\mathbf r).
$$

Energy flow 자체는 PBE와 같지만, target Fock과 이후 gradient에는 $v_\tau$ AO
contraction이 추가된다. CPU/GPU가 같은 libxc label, mesh와 $\tau$ convention을
사용해야 한다.

## 7. Common errors

- HF energy와 target energy를 더해 이중 계산함
- target RKS/KRKS SCF를 실행하여 DC energy surface를 바꿈
- `ecoul`/`exc`에 $1/N_k$를 두 번 적용함
- source와 target에 서로 다른 mesh 또는 DF provider를 사용함
- complex residual을 검사하지 않고 실수부만 취함
- SCAN/r2SCAN target을 GGA density layout으로 평가함

## 8. Tests와 evidence

- `tests/test_fixed_density_energy.py`: component reconstruction과 k-point weighting
- `tests/test_import_isolation.py`: legacy runtime dependency 금지
- FFTDF/GDF/RSDF validation JSON: full-energy FD에 사용된 central energy
- CPU/GPU validation: energy component와 final scalar parity

Energy flow는 gradient 구현보다 먼저 검증한다. Analytic gradient가 FD와 일치해도
central energy의 provider 또는 component convention이 잘못됐다면 올바른 DC-DFT
derivative라고 할 수 없다.
