# GDF GPU restricted DC-DFT analytic gradient

## 1. 목적과 완료된 public contract

GPU GDF route는 CPU GDF와 같은 restricted DC-DFT functional

$$
E_{\mathrm{DC-PBE}}(\mathbf R)
=E_{\mathrm{PBE}}[D_{\mathrm{HF}}(\mathbf R);\mathbf R]
$$

을 계산한다. source density는 RHF/KRHF에서 얻고, 그 density를 고정한 채 target
PBE energy를 평가한다. nuclear gradient에는 target energy의 explicit derivative와
source HF orbital response를 모두 포함한다.

사용자는 PySCF의 `with_df` convention으로 GDF를 선택한다.

```python
from gpu4pyscf_pbc_dcdft import df, dft

mf = dft.DCKRKS(cell, kpts=kpts, xc="PBE")
mf.with_df = df.GDF(cell, kpts=kpts, auxbasis="gth-szv")
mf.conv_tol = 1.0e-11
mf.conv_tol_grad = 1.0e-9
energy = mf.kernel()
gradient = mf.nuc_grad_method().kernel()
```

지원 범위는 다음과 같다.

- restricted closed-shell RHF/KRHF source
- PBE, SCAN 및 r2SCAN target DFA
- GTH pseudopotential
- Gamma 및 equal-weight sampled k-point mesh
- GDF source/metric/auxiliary response를 포함한 analytic nuclear gradient
- 반환 단위 Ha/Bohr; 반환값은 force가 아니라 $\partial E/\partial R$

`gpu4pyscf_pbc_dcdft.capability()`와 `mf.with_df.capability()`는 energy 및 analytic
gradient 지원 여부, silent fallback 여부, host boundary를 machine-readable dict로
반환한다.

## 2. Native와 reference setup mode

GPU GDF는 목적이 다른 두 setup mode를 제공한다.

### 2.1 Native mode: production default

`setup_mode="native"`는 GPU4PySCF의 CUDA RSDF builder를 GDF factor generator로
사용한다. 여기서 RSDF는 user-visible backend가 바뀐다는 뜻이 아니라 GDF Coulomb
integral을 short-/long-range로 나누어 계산하는 내부 acceleration algorithm이다.

Device가 직접 소유하는 큰 작업은 다음과 같다.

1. short-range 3-center integral과 long-range Fourier contribution
2. auxiliary metric factorization과 central CDERI 생성
3. source RHF/KRHF 및 target Hartree의 J/K contraction
4. GDF `int2c2e_ip1`, `int3c2e_ip1` source-contracted derivative
5. DC-DFT Z-vector의 bilinear two-electron response
6. HF AX, target XC와 final electronic assembly

### 2.2 Reference mode: equation-level witness

`setup_mode="reference"`는 PySCF CPU GDF가 만든 canonical CDERI를 device에 올리고,
$dC/dR$, $dJ/dR$, $dK/dR$ 전체 tensor를 명시적으로 구성한다. 과거 completion evidence,
CPU-identical factor gauge와 개별 source/metric term을 조사할 때 사용한다. production
benchmark에는 사용하지 않는다.

### 2.3 남아 있는 host boundary

- PySCF `Cell`/`AuxCell` metadata와 작은 optimizer bookkeeping
- GPU4PySCF CDERI의 mapped-host storage
- GTH local-part-2 value setup과 analytic H1 primitive
- sampled-k metric derivative에서 upstream이 호출하는 host `pbc_intor`
- Ewald ion-ion nuclear gradient와 마지막 NumPy report

따라서 `device="gpu"`는 host 호출이 0이라는 뜻이 아니라 central GDF와 반복적인
response contraction이 GPU 소유이며 CPU DC-DFT routine으로 silent fallback하지 않는다는
뜻이다.

## 3. Central CDERI representation

`df/gdf/gdf.py::GDF.build`는 native mode에서
`gpu4pyscf.pbc.df.GDF.build`를 호출한다. GPU4PySCF builder는 compressed CDERI를
GPU에서 만들고 mapped host storage와 device block iterator를 함께 사용한다. reference
mode에서는 CPU PySCF의 canonical unique-k-pair block을 CuPy로 upload한다.

`df/gdf/build_signature.py::GDFBuildSignature`는 central factor의 cache ownership을
명시한다. Key에는 geometry, lattice, k-point, basis, pseudo, mesh, omega, auxbasis,
linear-dependence threshold와 setup mode가 들어간다. 같은 key에서는 CDERI를 재사용하고,
in-place geometry 변경처럼 provider object는 그대로인데 physical input만 바뀐 경우에는
stale factor를 자동 폐기하고 다시 build한다. 이 동작은 실제 native CDERI object의
identity와 signature를 함께 검사하는 regression test로 고정돼 있다.

두 mode는 finite factor gauge가 다를 수 있지만 observable J/K와 energy가 promotion
threshold 안에서 일치해야 한다. 실제 native H2 gate의 CPU-GPU energy 차이는
$5.26\times10^{-10}$ Ha이다.

$$
C^{ij}_{P\mu\nu}
=\sum_Q [L(q)^{-1}]_{PQ}B^{ij}_{Q\mu\nu},
\qquad q=k_j-k_i,
$$

$$
M(q)=L(q)L(q)^\dagger.
$$

reference-mode device storage의 주요 shape는 다음과 같다.

| Quantity | Shape | Dtype | Owner |
| --- | --- | --- | --- |
| `C[(ki,kj)]` | `(naux,nao,nao)` | complex128 | CuPy |
| density | `(nset,nk,nao,nao)` | real/complex | CuPy |
| J/K | density와 같은 shape | real/complex | CuPy |
| `dC[(ki,kj)]` | `(natm,3,naux,nao,nao)` | complex128 | CuPy |

Gamma에서 real density가 들어오면 최종 J/K의 numerical imaginary noise를 제거해 real
array를 반환한다. 일반 k-point에서는 phase를 보존하기 위해 complex128을 유지한다.

## 4. Device J/K value contraction

closed-shell density matrix는 occupation factor 2를 이미 포함한다. auxiliary Coulomb
density와 Coulomb matrix는

$$
j_P=\sum_{k\mu\nu}C^k_{P\mu\nu}D^k_{\nu\mu},
$$

$$
J^k_{\mu\nu}=\frac1{N_k}\sum_Pj_PC^k_{P\mu\nu}
$$

로 계산한다. reference mode의 `df/gdf/gdf.py::_coulomb_matrices`에 있는 두
`einsum`은 각각 이 두 식에 대응한다. native mode에서는 GPU4PySCF의
compressed-CDERI block iterator와 CUDA contraction이 같은 식을 blockwise로 평가한다.
$1/N_k$는 equal k-point weight이다.

exchange는 모든 incoming/outgoing k-pair를 합한다.

$$
K^{k_i}_{\mu\lambda}
=\frac1{N_k}\sum_{k_jP\nu\sigma}
C^{ij}_{P\mu\nu}D^{k_j}_{\nu\sigma}
C^{ji}_{P\sigma\lambda}.
$$

reference mode에서는 `_exchange_matrices`가 이 contraction을 수행한다. native mode에서는
upstream GPU4PySCF GDF exchange kernel이 담당한다. `exxdiv='ewald'`이면 upstream
GPU4PySCF의 Ewald $G=0$ value correction을 같은 device density에 적용한다.

## 5. Metric response와 reference-mode dCDERI

nuclear displacement $R_{Ax}$에서 source integral과 auxiliary Coulomb metric이 모두
변한다.

$$
C=L^{-1}B,
$$

$$
C^{A_x}=L^{-1}B^{A_x}-L^{-1}L^{A_x}C.
$$

첫 항은 primary/auxiliary center가 움직이는 3-center source response이고 둘째 항은
2-center metric, 즉 auxiliary-basis response이다. 이 둘 중 하나라도 생략하면 fitted
Coulomb/exchange energy를 미분한 값이 아니다.

Reference mode에서 `df/gdf/response.py::build_gdf_response_tensors`는 host에서
$B,M,B^{A_x},M^{A_x}$ primitive를 만든다. 이후
`df/gdf/cholesky.py::CDERIFactorization`이 CuPy에서 다음을 수행한다.

1. $M=LL^\dagger$ Cholesky factorization
2. triangular solve로 $C=L^{-1}B$ 계산
3. $W=L^{-1}M^{A_x}L^{-\dagger}$ 계산
4. lower-Cholesky gauge $L^{A_x}=L\Phi(W)$ 구성
5. source와 metric response를 합쳐 $C^{A_x}$ 계산

complex source와 real Gamma metric을 섞을 때 dtype을 complex128로 승격한다. 그렇지
않으면 imaginary k-phase가 암묵적으로 버려지고 non-Gamma derivative가 깨진다.
`value_parity_max_abs`는 device에서 재구성한 central C와 uploaded witness C의 차이를
기록하여 metric gauge/mesh mismatch를 즉시 검출한다.

## 6. Native source-contracted derivative와 reference full tensor

Native mode는 full derivative AO tensor를 저장하지 않는다. GPU4PySCF의 periodic GDF
gradient kernel이 $dB$, $dM$ 및 long-range Fourier response를 density와 즉시
contraction하여 원자별 scalar derivative를 반환한다. Package는 이를 다음 API로 노출한다.

```python
provider.get_j_e1(dm)
provider.get_k_e1(dm, exxdiv="ewald")
provider.get_hf_e1(dm, exxdiv="ewald")
```

### 6.1 Fixed CUDA task pool과 bounded batching

GPU4PySCF 1.7.4의 `PBCsr_ejk_int3c2e_ip1` kernel은 worker마다 고정된
`POOL_SIZE=16384` task record를 사용한다. 한 short-range derivative work group의
실제 task 수는 대략

$$
N_{\mathrm{task}}
=N_{\mathrm{AO\ shell\ pair}}
 N_{\mathrm{aux\ shell}}
 N_{\mathrm{BvK\ cell}}
$$

이다. Small H2/H2O에서는 이 제한이 드러나지 않지만 water20 local PP part 1은 한
호출에서 제한을 넘었고, out-of-bounds task write가 실행마다 다른 gradient를 만들었다.
`local_part1_derivative.py::_bounded_short_range_batches`는 AO shell-pair와 fake-nucleus
auxiliary-shell offset을 함께 분할하고 다음 invariant를 실행 전에 검사한다.

$$
\max_b N_{ij}^{(b)}
\max_c N_{P}^{(c)}
N_{\mathrm{BvK\ cell}}
\leq \mathrm{POOL\_SIZE}.
$$

이 분할은 integral screening, density contraction 또는 물리식을 바꾸지 않는다. 같은
analytic contraction을 여러 safe CUDA launch로 나눌 뿐이다. Source HF J/K derivative의
upstream wrapper도 `contracted_derivative.py::_bounded_sr_derivative_batches`에서 같은
제약을 적용한다.

DC-DFT Z-vector에는 reference density $D$와 Hermitian transition density $T_Z$ 사이의
bilinear derivative가 필요하다. Quadratic RHF two-electron derivative를 $Q(D)$라 하면

$$
R^{Z}_{2e}=Q(D+T_Z)-Q(D-T_Z)
$$

이다. Distorted H2O에서 이 값은 reference full-$dJ/dR$, full-$dK/dR$ contraction과
$2.8\times10^{-10}$ Ha/Bohr 이내에서 일치했다. 구현 owner는
`df/gdf/contracted_derivative.py::restricted_hf_bilinear_derivative`다.

Reference/rollback mode인 `gdf_bilinear_mode="two_pass"`는 위 polarization identity를
그대로 두 번 평가한다. Production default인 `"one_pass"`는 Gamma와 sampled-k에 각각
전용 signed source를 만들고, 두 state의 SR/LR three-center, two-center metric,
auxiliary-center 및 Ewald source를 합친 뒤 각 nuclear derivative kernel을 한 번만
순회한다. 각 state의 zero-order factor는 그대로 보존하므로 식을 근사하지 않으며,
full derivative tensor도 만들지 않는다. Upstream의 `n_dm=2` UHF path는 이 restricted
cross bilinear와 식이 다르므로 사용하지 않는다. 세부 equation과 k-point phase convention은
`20_gdf_signed_two_density.md`에 정리했다.

J derivative는 auxiliary density와 output CDERI 양쪽 product rule을 포함한다.

$$
(J^k)^{A_x}=\frac1{N_k}\sum_P
\left[j_P(C^k_P)^{A_x}+j_P^{A_x}C^k_P\right],
$$

$$
j_P^{A_x}=\sum_{k\mu\nu}
(C^k_{P\mu\nu})^{A_x}D^k_{\nu\mu}.
$$

K derivative도 두 CDERI factor를 각각 미분한다.

$$
(K^{k_i})^{A_x}=\frac1{N_k}\sum_{k_jP\nu\sigma}
\left[
(C^{ij}_{P\mu\nu})^{A_x}D^{k_j}_{\nu\sigma}C^{ji}_{P\sigma\lambda}
+C^{ij}_{P\mu\nu}D^{k_j}_{\nu\sigma}(C^{ji}_{P\sigma\lambda})^{A_x}
\right].
$$

Reference 구현은 `df/gdf/jk_derivative.py::gdf_jk_derivative`이며 output shape는
`(natm,nkpts,3,nao,nao)`이다. reverse k-pair block은 Hermitian conjugation으로
복원한다. Ewald exchange correction $mSDS$의 nuclear derivative는 overlap 두 쪽의
product rule을 적용한다. fixed lattice에서 Madelung scalar $m$은 nuclear coordinate에
대해 상수이다.

## 7. Target PBE와 Gamma complex AX adapter

target Fock은 source HF density에서

$$
F_{\mathrm{PBE}}[D_{\mathrm{HF}}]
=h+J[D_{\mathrm{HF}}]+V_{xc}^{\mathrm{PBE}}[D_{\mathrm{HF}}]
$$

로 만들고, source HF Fock과의 occupied-virtual mismatch가 Z-vector RHS가 된다.
PBE XC density, potential, kernel과 AO-grid contractions는 GPU4PySCF `KNumInt` route를
사용한다.

Gamma GDF의 optimized GPU response routine은 real float64 density만 받지만, orbital
response에서 transition density는 complex Hermitian일 수 있다. 선형성으로

$$
G[D_R+iD_I]=G[D_R]+iG[D_I]
$$

를 사용하여 real symmetric와 imaginary antisymmetric channel을 분리한다.
`grad/response.py` adapter가 이 decomposition을 수행하고 imaginary channel에는
`hermi=0`을 전달한다. 이 처리가 없으면 Gamma Z-vector에서 dtype error 또는 imaginary
response 유실이 발생한다.

## 8. One-electron, Pulay, XC 및 전체 assembly

Native GDF의 one-electron derivative는 항별 소유권이 다르다.

- kinetic과 overlap은 GPU matrix primitive를 사용하고 nonlocal GTH projector derivative는
  GPU density-contracted `vppnl_nuc_grad`를 사용한다.
- local GTH part 1은 short-range 3-center와 long-range Fourier 항으로 나누고,
  `df/gdf/local_part1_derivative.py`가 CUDA kernel에서 density와 직접 contraction한다.
- local GTH part 2만 CPU analytic matrix primitive를 만든 뒤 device로 upload한다.

Production route는 이 항들을 모든 원자에 대한 거대한
$(3N_{atom},N_k,N_{AO},N_{AO})$ tensor로 조립하지 않는다. Kinetic AO-row primitive와
local-part-2 integral context를 보존하고, memory-derived Cartesian batch마다 완전한
$H^{A_x}$를 만들어 즉시 $D_{HF}$ 및 adjoint source와 contraction한 뒤 버린다. 따라서
peak memory는 전체 원자 수가 아니라 선택된 batch 크기에 비례한다.

Energy 평가에서 이미 만든 target Veff와 Fock은 immutable target workspace에 보관한다.
Geometry, lattice, k-point, mesh, basis, pseudo, precision, range-separation parameter,
target XC 또는 source MO object가 바뀌면 signature가 달라져 재사용하지 않는다. 동일한
energy object에서 바로 gradient를 계산할 때만 target numerical integration과 Coulomb
Fock build를 생략한다.

특히 local part 1은 전체
$(N_{atom},3,N_k,N_{AO},N_{AO})$ H1 tensor를 만들지 않는다. Explicit gradient에는
$D_{HF}$를, Z-response에는 transition density $T_Z$를 contraction한다. restricted
Z-vector convention 때문에 후자는 $2\,\mathrm{Tr}[T_Z V_{loc1}^{A_x}]$로 더한다.
이 route는 CPU package의 reciprocal-space `_ft_ip_ovlp`를 호출하지 않는다.
`setup_mode="reference"`만 equation-level witness로 기존 full host H1 builder를 유지한다.

최종 gradient는

$$
E^{A_x}_{DC}
=E^{A_x}_{1e}
+E^{A_x}_{J,\mathrm{target}}
+E^{A_x}_{xc}
+E^{A_x}_{S,\mathrm{target}}
+E^{A_x}_{nuc}
+z^TB^{A_x}_{HF}
$$

이다. `grad/gdf.py::gdf_gpu_gradient`의 native production 순서는 다음과 같다.

1. source MO partition과 target-Fock occupied-virtual RHS
2. GPU HF AX와 `auto` dense-reference/matrix-free MINRES adjoint Z-vector
3. GPU kinetic/nonlocal H1, host local-part-2 H1 및 native local-part-1 contraction
4. native `get_j_e1(D)`로 target Hartree derivative 계산
5. GPU target XC와 target-Fock Pulay assembly
6. matrix H1/overlap과 native local-part-1을 Z-vector와 contraction
7. overlap-induced nuclear CPHF source를 memory-bounded density batch로 계산
8. signed two-density one-pass kernel로 HF two-electron nuclear response 계산
9. Ewald nuclear gradient를 더하고 NumPy result 생성

`gdf_bilinear_mode="one_pass_fused"`에서는 4와 8을 별도로 실행하지 않는다. Pure-J
quadratic state $(D;w_J=1,w_K=0)$와 signed HF states
$(D+T_Z;+1,+1)$, $(D-T_Z;-1,-1)$의 source를 먼저 합쳐 SR/LR/metric/auxiliary/Ewald
derivative traversal을 한 번 수행한다. 이때 계산되는 것은 정확한 합

$$
G_{J}[D] + Q_{HF}(D+T_Z)-Q_{HF}(D-T_Z)
$$

이며 근사는 없다. 다만 한 output channel에서 두 항을 역으로 분리할 수 없으므로 이
합은 `GradientContributions.coupled`에 저장한다. 성분별 분석에는 기본 `one_pass`를,
최종 gradient timing에는 명시적 `one_pass_fused`를 사용할 수 있다.

또한 nuclear CPHF source의 HF response는 self-adjoint identity

$$
\langle Z,G[M^{A_x}]\rangle=\langle G[Z],M^{A_x}\rangle
$$

를 사용한다. 따라서 모든 Cartesian source에 대해 $G[M^{A_x}]$를 반복 계산하지 않고
Hermitian Z transition density에 $G$를 한 번 적용한 뒤, streamed $H^{A_x}$와
$S^{A_x}$ batch를 직접 contraction한다.

`GDFGradientResult.timings_s`는 native mode에서
`gdf_native_target_j_derivative`, `z_nuclear_device_contraction`,
`z_nuclear_cphf_source`, `gdf_native_bilinear_z_response`,
`gdf_native_contracted_pseudo_explicit`, `gdf_native_contracted_pseudo_z_response`,
`one_electron_host_integrals`를 분리한다. `z_nuclear_response_batch_size`는 한 번의
provider J/K traversal에 묶인 Cartesian source 수를 기록한다.
`z_nuclear_response_batch_plan`은 memory-derived preflight를,
`z_nuclear_response_batch_execution`은 OOM rebatch 횟수, 완료 component 수와 sampled
device/CuPy-pool high-water를 기록한다. 여기서 native mode의
`one_electron_host_integrals`는 local part 2 생성 시간이다. reference mode에서는
`gdf_host_primitive_integrals`, `gdf_metric_cderi_device_response`,
`gdf_jk_device_contraction`을 기록한다. 따라서 setup mode를 확인한 뒤 integral setup,
target Hartree derivative, Z-response 중 어느 부분이 병목인지 판단할 수 있다.

Water20 r2SCAN/Gamma/SZV, 400 Ha maintenance profile에서 matrix-free 전환 뒤
analytic-gradient wall time은 148.66 s에서 52.60 s로 감소했다. Nuclear CPHF source를
180-component batch로 묶은 뒤에는 49.43 s였다. One-pass production route의 독립 3회
median은 35.0482 s이고 bilinear response median은 15.2402 s다. 즉 batched two-pass의
49.4298/29.7504 s 대비 각각 29.1%/48.8% 감소했다. Gradient CV는 0.353%, run 간 최대
drift는 $4.45\times10^{-14}$ Ha/Bohr다.

`SRInt3c2eOpt` 객체만 plus/minus call 사이에 재사용한 실험은 water8에서 4.8686 s에서
4.8671 s로 사실상 변화가 없어 revert했다. 반면 signed source를 kernel 경계까지 내린
one-pass는 실제 derivative traversal을 절반으로 줄였기 때문에 유지했다.

Bounded-memory gate에서는 H2 Gamma와 $2\times1\times1$ sampled-k에서 automatic batch와
`z_nuclear_batch_max_components=1`을 비교해 final-gradient gap이 $10^{-9}$ Ha/Bohr
아래임을 확인했다. Water20에서는 default 180-component batch와 maximum 30-component
강제 run을 비교했다. 강제 run은 6 batch에서 180/180 component를 완료했고 gradient
maximum gap은 $3.78\times10^{-14}$ Ha/Bohr였다. Synthetic OOM test는 첫 range를
실패시킨 뒤 batch를 절반으로 줄여 모든 component를 순서대로 정확히 한 번 commit하는지
검증한다.

## 9. Gamma와 sampled k-point 차이

Gamma에서는 `nkpts=1`, 모든 central value는 실수로 축약할 수 있다. sampled k-point에서는
다음을 추가로 지켜야 한다.

- CDERI key는 ordered pair `(ki,kj)`와 momentum transfer $q=k_j-k_i$를 보존한다.
- reverse pair는 단순 transpose가 아니라 complex Hermitian conjugate이다.
- Coulomb은 diagonal `(k,k)` block, exchange는 모든 compatible pair를 사용한다.
- density/J/K/response matrix는 complex128 phase를 유지한다.
- 모든 Brillouin-zone sum에 equal weight $1/N_k$를 한 번만 적용한다.

`examples/03_gdf_gpu_kpoint.py`와 validation의 H2/H2O `(2,1,1)` case가 이 경로를 직접
실행한다.

## 10. 검증 결과와 실행 방법

Native production evidence는
`pbc/gpu/validation/results/2026-07-15_gdf_gpu_native.json`이다.

2026-07-16 optimization release gate는 public benchmark API로 다시 실행했다.

| Case | max analytic-FD | CPU-GPU energy | CPU-GPU gradient |
| --- | ---: | ---: | ---: |
| H2 Gamma PBE, native GPU | $2.411\times10^{-9}$ | $2.335\times10^{-11}$ | $4.832\times10^{-11}$ |
| H2 $2\times1\times1$ PBE, native GPU | $2.416\times10^{-9}$ | - | - |
| H2O Gamma PBE, native GPU | $9.176\times10^{-8}$ | $4.531\times10^{-11}$ | $7.966\times10^{-11}$ |
| water20 Gamma r2SCAN, selected component | $4.618\times10^{-9}$ | - | - |

Raw reports는
`benchmark/results/gdf_optimization_release_gate_2026-07-16/`에 있다. 모든 값은
optimization goal의 FD, energy 및 gradient gate를 통과했다.

| Case | max analytic-FD | CPU-GPU energy | CPU-GPU gradient |
| --- | ---: | ---: | ---: |
| H2 Gamma SZV, native | $6.949\times10^{-8}$ | $5.260\times10^{-10}$ | $3.094\times10^{-10}$ |
| distorted H2O Gamma SZV, native central check | - | $1.337\times10^{-9}$ | $3.439\times10^{-10}$ |
| water20 Gamma SZV, r2SCAN native | $1.215\times10^{-8}$ | - | - |

H2 native full Cartesian FD와 distorted-H2O CPU/GPU central parity가 모두 promotion gate를
통과했다. Gamma 및 sampled-k native unit gradient gate와
`get_j_e1/get_k_e1/get_hf_e1` algebra gate도 통과한다.
Water20 row는 수정 전 계산에서 이미 저장된 $10^{-4}$ Bohr full-energy central FD 180개
성분을 수정 후 analytic gradient와 비교한 값이다. RMS error는
$3.152\times10^{-9}$ Ha/Bohr이고, analytic gradient wall time은 $151.157$ s였다.

다음 표는 full derivative tensor를 명시적으로 만드는 `setup_mode="reference"`의
2026-07-14 equation-level evidence다.

| Case | max analytic-FD | CPU-GPU energy | CPU-GPU gradient |
| --- | ---: | ---: | ---: |
| H2 Gamma SZV | $6.962\times10^{-8}$ | $3.790\times10^{-11}$ | $6.266\times10^{-10}$ |
| H2 k211 DZVP | $5.104\times10^{-8}$ | $6.907\times10^{-12}$ | $4.389\times10^{-10}$ |
| distorted H2O Gamma SZV | $1.590\times10^{-7}$ | $5.137\times10^{-12}$ | $1.489\times10^{-10}$ |
| distorted H2O k211 SZV | $3.117\times10^{-7}$ | $1.322\times10^{-12}$ | $1.260\times10^{-10}$ |

모든 값은 각각 $10^{-5}$ Ha/Bohr, $10^{-8}$ Ha, $10^{-7}$ Ha/Bohr gate보다 작다.
full Cartesian FD는 validation에서만 사용되며 production gradient에는 displaced geometry가
없다.

```bash
cd pbc/gpu
export PYTHONPATH=../cpu/src:src
python validation/validate_gdf_gpu.py --full \
  --output validation/results/gdf_gpu.json
```

GPU unit suite:

```bash
python -m unittest discover -s tests -v
```

public examples:

```bash
python examples/02_gdf_gpu_energy_gradient.py
python examples/03_gdf_gpu_kpoint.py
```

## 11. 흔한 오류와 debug 순서

1. **Large system에서만 FD error 증가**: metric/auxiliary response omission 또는
   source/metric mesh mismatch와 함께 `PBCsr_ejk_int3c2e_ip1`의 shell-pair ×
   auxiliary-shell × BvK-cell batch product가 `POOL_SIZE`를 넘지 않는지 검사한다. 실행마다
   analytic gradient가 달라지면 task-pool overflow를 먼저 의심한다.
2. **Gamma는 맞고 k-point만 실패**: pair orientation, conjugation, $1/N_k$, complex dtype을
   검사한다.
3. **analytic-FD는 맞고 CPU-GPU energy만 실패**: CPU/GPU가 다른 central CDERI 또는
   hcore representation을 사용했는지 확인한다.
4. **Z-vector residual 증가**: source SCF의 `conv_tol_grad`, Gamma complex transition-density
   adapter, AX density shape를 확인한다.
5. **net translation이 mesh에 민감**: GDF metric보다 uniform target-XC grid의 egg-box
   error를 먼저 분리한다.
6. **GPU라고 했는데 CPU time이 큼**: 먼저 `gdf_setup_mode`를 확인한다. `reference`이면
   CPU canonical CDERI와 derivative primitive generation이 의도된 동작이다. `native`이면
   local part 2에 해당하는 `one_electron_host_integrals`, mapped-host CDERI bookkeeping,
   sampled-k metric의 upstream host `pbc_intor`가 남은 boundary다. Native route에서
   `_ft_ip_ovlp`가 나타나면 새 source가 아닌 이전 process/wheel을 사용 중인지 먼저
   확인한다. 어느 경우에도 metadata의 `silent_cpu_fallback`과 `host_boundaries`를 함께
   확인한다.
