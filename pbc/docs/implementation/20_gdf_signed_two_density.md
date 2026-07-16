# GDF signed two-density derivative optimization boundary

## 1. 목적

Native GPU GDF DC-DFT gradient의 현재 지배 병목은 reference density $D$와 Z-vector
transition density $T$ 사이의 restricted-HF two-electron nuclear derivative다. 현재
production 구현은 quadratic derivative $Q(X)$를 이용한 exact polarization identity

$$
R_{2e}^{Z}=Q(D+T)-Q(D-T)
$$

를 사용한다. 이 식은 정확하지만 periodic three-center source, Coulomb metric,
auxiliary-center, short-range 및 long-range derivative traversal을 두 번 수행한다.

이 문서는 one-pass signed two-density kernel이 만족해야 하는 equation과 source ownership을
고정한다. 이 설계는 `Goal_GDF_Optimization.md`의 G2에 속하며 public physics scope를
확장하지 않는다.

## 2. Full-matrix reference equation

Nuclear component $R_{A_x}$에서 density에 선형인 restricted-HF matrix derivative를

$$
L^{A_x}[X]
=\frac{\partial}{\partial R_{A_x}}
\left(J[X]-\frac12K[X]\right)
$$

로 정의한다. Fixed-density quadratic derivative는

$$
Q^{A_x}(X)
=\frac{1}{2N_k}\sum_{\mathbf k}
\operatorname{Re}\operatorname{Tr}
\left[X_{\mathbf k}L_{\mathbf k}^{A_x}[X]\right]
$$

이다. $L$의 density linearity를 사용하면 polarization difference를 cancellation 없이
다음과 같이 쓸 수 있다.

$$
R_{2e}^{Z,A_x}
=\frac{1}{N_k}\sum_{\mathbf k}
\operatorname{Re}\left\{
\operatorname{Tr}\left[D_{\mathbf k}L_{\mathbf k}^{A_x}[T]\right]
+\operatorname{Tr}\left[T_{\mathbf k}L_{\mathbf k}^{A_x}[D]\right]
\right\}.
$$

`df/gdf/jk_derivative.py`의 다음 helper가 이 식의 equation-level witness다.

- `contract_restricted_hf_energy_derivative`
- `contract_restricted_hf_bilinear_energy_derivative`

이 helper는 full $dC/dR$ reference tensor를 소비하므로 production route가 아니다.
Gamma와 sampled-k synthetic CDERI/dCDERI에서 polarization identity와 $10^{-9}$
Ha/Bohr 이내로 일치하는 unit test가 one-pass 구현의 첫 parity gate다.

## 3. Coulomb cross term

Whitened CDERI를 $C_{P\mu\nu}^{\mathbf k}$라 하고 auxiliary density를

$$
u_P[X]=\sum_{\mathbf k\mu\nu}
C_{P\mu\nu}^{\mathbf k}X_{\nu\mu}^{\mathbf k}
$$

로 정의한다. Bilinear Coulomb derivative에는 $u[D]u[T]$의 두 orientation이 필요하다.

$$
M_{PQ}^{J,DT}
=u_P[D]u_Q[T]+u_P[T]u_Q[D].
$$

Three-center source derivative contraction도 같은 cross structure를 갖는다.

$$
X_{P\mu\nu}^{J,DT}
=D_{\mu\nu}u_P[T]+T_{\mu\nu}u_P[D].
$$

따라서 기존 UHF처럼 spin density를 단순 합산한 뒤 $J[D_\alpha+D_\beta]$를 계산하면
self term과 cross term이 섞이며 원하는 signed result를 직접 얻을 수 없다.

## 4. Exchange cross term

Exchange는 density를 low-rank factor로 분해한 occupied block을 사용한다. Symbolically,

$$
M_{PQ}^{K,DT}
=-\frac12\left[
O_P[D]O_Q[T]+O_P[T]O_Q[D]
\right]
$$

이며 three-center derivative source에도 $D$와 $T$를 서로 바꾼 두 orientation이 모두
필요하다. Molecular GPU4PySCF의
`gpu4pyscf/df/grad/tdrhf.py::_jk_energies_per_atom`은 `dm_pairs`를 받아 이 구조를
구현한다. 하지만 해당 함수는 molecular integral optimizer와 non-periodic AO layout을
사용하므로 PBC GDF에 직접 호출할 수 없다.

## 5. PBC에서 추가로 소유해야 하는 항

PBC one-pass 구현은 molecular pair contraction 외에 다음 항을 모두 포함해야 한다.

1. `SRInt3c2eOpt`의 BvK image 및 sorted-cell AO layout
2. `PBCsr_ejk_int3c2e_ip1`에 전달할 cross-density compressed source
3. reciprocal-space LR three-center zeroth-order block
4. LR AO-pair derivative `PBC_ft_aopair_ek_ip1`용 cross source
5. two-center metric derivative의 $D/T$ cross auxiliary matrix
6. auxiliary-center derivative와 k-point conjugate pair normalization
7. `exxdiv="ewald"` overlap correction의 cross derivative

Gamma 구현 owner의 upstream reference는
`gpu4pyscf/pbc/df/grad/rhf.py::_jk_energy_per_atom`, sampled-k owner는
`gpu4pyscf/pbc/df/grad/krhf.py::_jk_energy_per_atom`이다. 두 함수는 quadratic density만
받으므로 public wrapper 변경만으로는 충분하지 않다.

## 6. 왜 UHF `n_dm=2`가 대체가 아닌가

PBC UHF derivative route는

$$
J[D_\alpha+D_\beta]-K[D_\alpha]-K[D_\beta]
$$

형태로 spin channel을 합산한다. 예를 들어
$D_\alpha=D+T$, $D_\beta=D-T$를 넣으면 exchange cross term은 두 channel 사이에서
상쇄되고 Coulomb은 $2D$의 self term이 된다. $D_\alpha=D$, $D_\beta=T$를 넣으면
Coulomb cross는 포함되지만 exchange cross는 없다. 현재 kernel은 각 channel의 signed
cross contribution을 별도로 반환하지 않으므로 한 번의 UHF call로 원하는 식을 복원할
수 없다.

## 7. 구현 순서와 promotion gate

One-pass route는 다음 순서로만 진행한다.

1. **Gamma algebra prototype**: full-matrix reference와 Coulomb/exchange/metric 항별 parity
2. **Gamma native integration**: 기존 two-pass와 H2/H2O full-gradient parity
3. **Sampled-k extension**: ordered k-pair, conjugation 및 $1/N_k$ normalization parity
4. **Performance gate**: water8에서 bilinear wall time의 측정 가능한 감소
5. **Limited water20 gate**: 필요할 때만 한 번 실행하며 water64는 실행하지 않음

다음 조건을 모두 만족하기 전에는 production dispatcher를 one-pass로 바꾸지 않는다.

- existing two-pass와 maximum difference $\le 10^{-9}$ Ha/Bohr
- full gradient CPU/GPU difference $\le 10^{-7}$ Ha/Bohr
- H2/H2O analytic-FD difference $\le 10^{-5}$ Ha/Bohr
- Gamma와 sampled-k 모두 지원
- full $dC/dR$ tensor materialization 없음
- bounded batch/task-pool policy 유지

## 8. 구현 전 판정과 reference prototype

Reference equation과 test contract를 먼저 고정한 뒤 별도 pair-contracted
implementation을 작성했다. 아래 8.1--8.2는 구현 전 algebra/API 판단을 보존하는
기록이며, 현재 production 판정은 9--11절을 따른다.

### 8.1 Gamma direct-source algebra prototype

`df/gdf/signed_density.py`는 real Hermitian Gamma density $D,T$와 central CDERI
$C_{Pij}$로부터 derivative kernel에 직접 전달할 source를 구성한다. Coulomb auxiliary
density를

$$
u_P[X]=\sum_{ij}C_{Pij}X_{ji}
$$

로 두면 source는

$$
S^J_{Pij}=2\left(D_{ji}u_P[T]+T_{ji}u_P[D]\right),
$$

$$
S^{K,\mathrm{RHF}}_{Pij}
=-\left(D C_P T+T C_P D\right)_{ji}
$$

이다. 두 번째 식에는 restricted $-K/2$ coefficient와 product-rule의 두 orientation이
이미 포함된다. 따라서 한 nuclear component의 central-factor response는

$$
R^{DT}_{A_x}=\sum_{Pij}
\frac{\partial C_{Pij}}{\partial R_{A_x}}
\left(S^J_{Pij}+S^{K,\mathrm{RHF}}_{Pij}\right)
$$

로 계산할 수 있다.

Random real symmetric Gamma CDERI/dCDERI에서 이 direct source와 기존 full-matrix
bilinear reference의 maximum difference는 $9.09\times10^{-13}$ Ha/Bohr였고
$10^{-9}$ gate를 통과했다. 이 prototype은 $O(N_{aux}N_{AO}^2)$ source만 만들고
full nuclear $dC/dR$는 production에서 만들지 않는 설계다.

현재 이 source는 test/reference boundary이며 production dispatcher에는 연결하지 않았다.
Native GPU4PySCF GDF의 SR/LR/metric decomposition에서 사용하는 central object가 단순
whitened CDERI 하나가 아니므로, source를 기존 SR kernel에 바로 넘기는 것만으로 metric,
auxiliary-center와 LR response가 자동으로 포함되지는 않는다.

### 8.2 CUDA API source audit

현재 PBC short-range derivative kernel
`PBCsr_ejk_int3c2e_ip1`의 C API는 한 개의 contracted `dm`과 한 개의
`density_auxvec`만 받는다. Molecular GPU4PySCF의
`df/grad/tdrhf.py::_jk_energies_per_atom`은 반대로 `dm_pairs`와 `n_dm` axis를 갖고,
각 pair의 direct cross source를 한 traversal에서 처리한다. 따라서 molecular wrapper를
PBC object에 연결하거나 PBC RHF helper를 두 density로 두 번 호출하는 것만으로는
one-pass가 되지 않는다.

PBC에서 가능한 최소 native 설계는 Python/CuPy에서 $D/T$의 direct cross-contracted
short-range source를 만들고 기존 SR CUDA traversal을 한 번 호출하는 것이다. 그러나
그것만으로 전체 식이 완성되지는 않는다. 같은 call에서 사용하는 long-range
three-center response, two-center metric derivative, auxiliary-center derivative 및 Ewald
overlap correction도 동일한 $D/T$ cross algebra로 바뀌어야 한다. 이 항 중 하나라도
quadratic $D\pm T$ route에 남으면 전체 traversal은 여전히 두 번 필요하거나 물리식이
불완전해진다.

그러므로 다음 source patch는 public dispatcher 변경이 아니라 다음 두 층으로 나눈다.

1. 현재 Gamma direct-source algebra를 SR/LR/metric/auxiliary/Ewald별 native
   intermediate로 분해하는 private implementation
2. 항별 parity가 확보된 뒤 기존 `PBCsr_ejk_int3c2e_ip1`의 pair-aware sibling kernel 또는
   equivalent compressed-source entry point

이 audit 단계에서는 `restricted_hf_bilinear_derivative`의 exact two-pass
$Q(D+T)-Q(D-T)$를 authoritative baseline으로 유지했다.

## 9. Native one-pass implementation

Production implementation은 다음 두 private module로 분리한다.

- `df/gdf/signed_gamma_native.py`: real Gamma AO layout
- `df/gdf/signed_kpoint_native.py`: sampled-k momentum-transfer layout

두 module은 $X_+=D+T$, $X_-=D-T$를 각각 factorize한다. Zero-order occupied-pair
factor는 상태별로 유지하지만 nuclear derivative 앞에서 source를

$$
S^{(1)} = S^{(1)}[X_+] - S^{(1)}[X_-]
$$

로 합친다. 따라서 다음 derivative traversal은 각각 한 번만 실행된다.

1. short-range three-center AO/auxiliary derivative
2. reciprocal-space long-range AO-pair derivative
3. two-center auxiliary metric derivative
4. long-range auxiliary-center derivative
5. Ewald overlap correction

Gamma는 original AO density와 sorted AO density를 구분한다. Ewald overlap은 original AO
order를 사용해야 하며, 초기 prototype에서 sorted density를 사용했을 때 H2O Ewald
maximum gap이 $5.26\times10^{-2}$ Ha/Bohr였다. `original_density` ownership을 분리한 뒤
gap은 $10^{-9}$ gate 아래로 내려갔다.

Sampled-k route는 GPU4PySCF KRHF의 `double_translation_indices`,
`kk_adapted_iter`, BvK phase 및 conjugate momentum pair를 그대로 따른다. Coulomb과
exchange source에는 각각 signed state coefficient와 $1/N_k^2$를 적용하고, conjugate가
서로 다른 momentum-transfer block은 한 쪽을 계산한 뒤 factor 2를 적용한다. Ewald 항은
최종적으로 $1/N_k$ normalization을 사용한다.

두 route 모두 full nuclear $dC/dR$, $dJ/dR$, $dK/dR$ tensor를 만들지 않는다. SR task
pool은 AO shell-pair, auxiliary shell 및 BvK cell의 곱이 GPU4PySCF `POOL_SIZE`를 넘지
않도록 두 batching axis를 명시적으로 제한한다.

## 10. Correctness와 performance evidence

다음 gate를 통과했다.

- H2/H2O Gamma one-pass 대 two-pass matrix parity
- H2/H2O $2\times1\times1$ sampled-k one-pass 대 two-pass matrix parity
- H2 Gamma/sampled-k full-gradient one-pass 대 two-pass parity
- default one-pass H2 Gamma/sampled-k CPU-GPU full-gradient parity
- GDF package regression 27/27

모든 one-pass/two-pass matrix 및 full-gradient 비교 기준은
$10^{-9}$ Ha/Bohr다. Fixed water8 r2SCAN/GDF/Gamma/SZV-GTH, 400 Ha에서 independent
process 3회 결과는 다음과 같다.

| 항목 | Two-pass median | One-pass median | Reduction |
| --- | ---: | ---: | ---: |
| Analytic gradient | 9.5656 s | 7.0863 s | 25.9% |
| Bilinear response | 4.8493 s | 2.4702 s | 49.1% |
| Z nuclear total | 5.1764 s | 2.8100 s | 45.7% |

One-pass run 간 maximum gradient drift는 $1.19\times10^{-14}$ Ha/Bohr이고 two-pass
baseline 대비 maximum gap은 $8.95\times10^{-14}$ Ha/Bohr다. Energy는 동일하며 target J
derivative timing은 1.4925 s 대 1.5021 s로 변하지 않았다. 따라서 관측된 감소는 target
J나 energy가 아니라 signed bilinear derivative traversal을 절반으로 줄인 결과다.

## 11. Production policy

`RestrictedDCDFTGradient.gdf_bilinear_mode`의 기본값은 `"one_pass"`다. 이 mode는 Gamma와
sampled-k를 자동 dispatch한다. Exact polarization route는 다음처럼 명시적으로 선택할 수
있으며 reference/rollback gate로 계속 유지한다.

```python
gradient = mf.nuc_grad_method()
gradient.gdf_bilinear_mode = "two_pass"
value = gradient.kernel()
```

기존 water8 evidence 재현을 위한 `"one_pass_gamma"` alias는 Gamma에서만 허용한다.
Reference GDF setup, FFTDF 및 RSDF에는 one-pass mode를 적용하지 않으며 unsupported 조합은
silent fallback 없이 preflight에서 실패한다. G2는 이 evidence로 `complete`다.
Water64는 실행하지 않았다.

### 11.1 Target-J와 signed response의 fused mode

`"one_pass_fused"`는 signed bilinear traversal에 pure Coulomb state를 하나 더 넣는다.
각 quadratic state $s$는 독립적인 Coulomb/exchange weight
$(w_J^s,w_K^s)$를 가지며 source는 다음처럼 합쳐진다.

$$
\Gamma_{metric}
=\sum_s\left[
w_J^s u_su_s^\dagger
-\frac{w_K^s}{2}\operatorname{Tr}_{ij}
\left(B^s_{Pij}B^s_{Qji}\right)
\right].
$$

Fused route의 세 state는

$$
(D;1,0),\qquad (D+T_Z;1,1),\qquad (D-T_Z;-1,-1)
$$

이다. 같은 weight 분리를 SR three-center, LR Fourier, auxiliary-center와 Ewald exchange
source에도 적용한다. 따라서 derivative integral을 평가한 뒤 source별 결과를 더하는
것이 아니라, source를 먼저 더하고 각 native derivative kernel을 한 번만 호출한다.

이 route는 최종 합만 반환한다. 단일-output CUDA ABI에서는 target-J와 Z-bilinear를
사후에 유일하게 복원할 수 없으므로 두 성분을 별도로 보고해야 하는 계산은 기존
`"one_pass"`를 사용한다. Fused 값은 `GradientContributions.coupled`에 저장되어
explicit/response 성분에 임의로 섞이지 않는다.
