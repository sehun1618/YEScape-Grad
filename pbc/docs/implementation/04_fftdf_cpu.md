# FFTDF CPU restricted DC-PBE analytic gradient

## 1. 이 route가 계산하는 양

`pyscf_pbc_dcdft.grad.fftdf.fftdf_cpu_gradient`는 restricted HF density에서
평가한 PBE energy,

$$
E_{\mathrm{DC-PBE}}(\mathbf R)
=E_{\mathrm{PBE}}[D_{\mathrm{HF}}(\mathbf R);\mathbf R],
$$

의 nuclear derivative를 계산한다. 반환값은 force가 아니라 gradient
$\partial E/\partial R_{Ax}$이며 단위는 Ha/Bohr이다. HF orbital은 각
geometry에서 다시 최적화되지만 PBE orbital SCF는 수행하지 않는다.

PBE energy는 HF orbital에 대해 stationary하지 않으므로 ordinary PBE gradient나
ordinary HF gradient를 그대로 사용할 수 없다. 전체 derivative는 다음 두 부분으로
나뉜다.

$$
E^{A_x}
=E^{A_x}_{\mathrm{explicit}}
+E^{A_x}_{\mathrm{response}}.
$$

- `explicit`: HF MO coefficient를 고정한 target PBE energy의 derivative와
  target-Fock overlap/Pulay term
- `response`: nuclear perturbation에 대한 HF orbital response를 Z-vector로
  contraction한 항

코드에서는 `GradientContributions.explicit`, `response`, `total`로 보존한다.

## 2. Gamma와 k-point의 공통 tensor convention

내부 구현은 Gamma도 $N_k=1$로 표현한다. 주요 shape는 다음과 같다.

| Quantity | Shape | Meaning |
| --- | --- | --- |
| `kpts` | `(nkpts, 3)` | reciprocal Bohr |
| `density` | `(nkpts, nao, nao)` | spin-summed HF AO density |
| `hcore_by_atom` | `(natm, 3, nkpts, nao, nao)` | full one-electron derivative |
| `overlap_rows` | `(nkpts, 3, nao, nao)` | PySCF AO-row overlap derivative |
| `source_jk` | `(natm, nkpts, 3, nao, nao)` | full `d(J-K/2)` matrix |
| `Z_k` | `(nvir_k, nocc_k)` | k-resolved Z-vector block |
| final gradient | `(natm, 3)` | Ha/Bohr |

equal-weight Monkhorst-Pack mesh에서는 모든 k-sum에 $1/N_k$가 들어간다. 이
factor는 각 low-level kernel에 흩어 놓지 않고 scalar contraction 시점에 명시한다.

## 3. source HF backend ownership

작은 Gamma cell에서 ordinary PySCF `RHF.get_jk()`는 메모리가 충분하면
`with_df=FFTDF`가 있어도 in-core ERI를 선택할 수 있다. 이 경우 HF energy와 density는
in-core Ewald-exchange model로 계산되지만 gradient code가 FFTDF matrix를 미분하게
되어 서로 다른 approximate Hamiltonian을 비교하게 된다.

`scf/reference.py::DensityFittedRHF`는 Gamma data layout을 유지하면서 J/K를 항상
명시된 `with_df` provider에 위임한다. 따라서 다음 세 곳이 같은 mesh와 Coulomb
convention을 사용한다.

1. HF SCF와 HF density
2. target PBE energy의 Hartree term
3. source HF CPHF operator와 nuclear response source

이 경계는 단순 performance option이 아니라 analytic-vs-FD consistency 조건이다.

## 4. explicit gradient

### 4.1 One-electron term

kinetic plus local GTH derivative는 PySCF
`pyscf.pbc.grad.krhf.hcore_generator`를 사용한다. nonlocal GTH는 Z-vector source에서
matrix 자체가 필요하므로 `pseudo/fft_nonlocal_derivative.py`에서 finite reciprocal-mesh
projector를 직접 미분한다. GDF energy가 사용하는 exact projector integral의 derivative는
별도 owner인 `pseudo/nonlocal_derivative.py`에 있다. 두 표현은 continuum limit에서는
같지만 finite mesh에서는 교환할 수 없다.

$$
E^{A_x}_{1e}
=\frac{1}{N_k}\sum_k
\operatorname{Tr}\!\left[D_k h_k^{A_x}\right].
$$

`h_k^{A_x}`는 이미 bra와 ket, local projector center, nonlocal projector center,
AO-center translation을 포함한 full derivative이다. 따라서 이 contraction에는
추가 factor `2`를 붙이지 않는다.

### 4.2 Target Hartree 및 XC term

PySCF `get_j_e1`과 periodic `krks.get_vxc`는 derivative가 AO bra row에 작용한
matrix를 반환한다. atom A가 소유한 AO row만 contraction하고 conjugate ket 쪽을
더하기 위해 factor `2`를 적용한다.

$$
E^{A_x}_{J+xc}
=\frac{2}{N_k}\sum_k
\sum_{\mu\in A,\nu}
\left(V^{x}_{J+xc,k}\right)_{\mu\nu}
D_{k,\nu\mu}.
$$

Gamma에서도 `KNumInt`와 one-element k-axis를 사용한다. 이는 Gamma `NumInt`
energy와 같은 quadrature를 사용하면서 shape branch를 제거한다.

### 4.3 Target-Fock overlap/Pulay term

target energy-weighted density는 HF orbital energy가 아니라 target PBE Fock의
occupied block으로 정의한다.

$$
D^{E,\mathrm{target}}_k
=2C_{o,k}
\left(C_{o,k}^{\dagger}F^{\mathrm{PBE}}_kC_{o,k}\right)
C_{o,k}^{\dagger}.
$$

따라서 overlap term은

$$
E^{A_x}_{S,\mathrm{target}}
=-\frac{2}{N_k}\sum_k
\sum_{\mu\in A,\nu}
S^x_{k,\mu\nu}D^{E,\mathrm{target}}_{k,\nu\mu}
$$

이다. ordinary HF의 `make_rdm1e(mo_energy, ...)`로 대체하면 DC-DFT
nonvariational term의 일부가 사라진다.

### 4.4 Nuclear Ewald term

fixed lattice에서 PySCF `krhf.grad_nuc`의 Ewald nuclear gradient를 사용한다.
stress/cell derivative는 이 route에 포함되지 않는다.

## 5. Z-vector response

target PBE Fock의 virtual--occupied block

$$
g_{ai,k}=\left(C_{v,k}^{\dagger}F^{\mathrm{PBE}}_kC_{o,k}\right)_{ai}
$$

이 nonvariational orbital gradient이다. HF orbital Hessian의 real-linear 표현을
$A_R$라 하면 adjoint equation은

$$
A_R^{T}z_R=-g_R
$$

이다. complex k-point block은 real/imaginary part를 모두 보존해 real-linear matrix로
변환한다. 작은-system correctness route는 dense solve를 사용하며, 후속 large-system
route는 같은 operator contract를 matrix-free solver로 교체할 수 있다.

nuclear perturbation source는

$$
B^{A_x}_{ai,k}
=H^{A_x}_{ai,k}
-S^{A_x}_{ai,k}\epsilon_{i,k}
-\frac12\left[(A+B)^{\mathrm{HF}}S^{A_x}_{oo,k}\right]_{ai}
$$

이고 response correction은

$$
E^{A_x}_{\mathrm{response}}
=\frac{4}{N_k}\operatorname{Re}
\sum_k\langle Z_k|B^{A_x}_k\rangle.
$$

factor `4`는 restricted occupation factor와 Hermitian `vo + ov` rotation channel을
합친 결과다.

## 6. FFTDF J/K full matrix derivative

ordinary variational gradient는 PySCF `get_j_e1/get_k_e1` AO-row derivative를 density와
바로 contraction할 수 있다. DC response는 derivative matrix를 먼저 `vo` block으로
project해야 하므로 AO가 나타나는 모든 위치를 product rule로 미분한다.

- J: output bra/ket AO, density를 만드는 bra/ket AO, Hartree potential response
- K: target-k bra/ket AO, source-k bra/ket AO, pair-density potential response

구현 위치는 다음과 같다.

- grid primitives: `df/fft/grid.py`
- full J/K derivatives: `df/fft/jk_derivative.py`
- FFTDF finite-mesh GTH nonlocal derivative: `pseudo/fft_nonlocal_derivative.py`

GDF의 exact-integral GTH derivative를 이 route에 재사용하면 FFTDF finite-mesh energy와
derivative가 서로 다른 Hamiltonian을 나타내게 된다. 실제 distorted H2O gate에서 이
잘못된 공유는 $10^{-2}$ Ha/Bohr 수준의 analytic-FD 회귀를 만들었으므로 backend-specific
ownership을 regression test로 고정한다.

현재 exchange implementation은 수식이 드러나는 direct correctness kernel이다.
작은 validation system에는 적합하지만 large production system에서는 blocked 또는
GPU kernel로 대체해야 한다.

## 7. cutoff, mesh 및 translation invariance

`cell.ke_cutoff`에서 PySCF가 만든 `cell.mesh`를 energy와 gradient가 함께 사용한다.
별도의 gradient mesh를 만들지 않는다. 낮은 cutoff에서 원자 전체를 평행이동했을 때
finite FFT grid aliasing 때문에 energy 자체가 완전히 invariant하지 않을 수 있다.
이 경우 analytic gradient의 atom sum도 0이 아니며, 다음 두 값을 구분해야 한다.

1. analytic gradient와 **같은 finite-grid energy**의 FD 차이
2. cutoff 증가에 따른 net gradient 및 energy convergence

첫 번째가 작고 두 번째가 cutoff와 함께 감소하면 analytic implementation은 현재
discretized model을 올바르게 미분하고 있는 것이다. net force만 보고 derivative term
누락으로 단정하면 안 된다.

## 8. 검증

빠른 unit gate:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

전체 completion matrix:

```bash
PYTHONPATH=src python validation/validate_fftdf_cpu.py --full \
  --output validation/results/fftdf_cpu.json
```

full validation은 H2/H2O, Gamma/non-Gamma, GTH-SZV/GTH-DZVP를 포함하며 전체
Cartesian central FD와 비교한다. 현재 promotion threshold는 maximum absolute error
$10^{-5}$ Ha/Bohr이다.
