# FFTDF GPU 구현

## 1. 목적과 완료 범위

이 문서는 `gpu4pyscf-pbc-dcdft`가 restricted periodic DC-PBE energy와
nuclear analytic gradient를 FFTDF backend에서 어떻게 GPU로 계산하는지 설명한다.
지원 범위는 다음과 같다.

- source state: closed-shell RHF 또는 KRHF
- target energy: frozen HF density에서 평가한 PBE
- boundary: 3D periodic cell
- core: GTH pseudopotential
- sampling: Gamma-point와 일반 k-point mesh
- numerical backend: GPU4PySCF FFTDF와 project-local CuPy derivative kernel
- output unit: energy는 Hartree, gradient는 Hartree/Bohr

`capability()["fftdf_analytic_gradient"] == True`는 단순히 GPU object를 만든다는
뜻이 아니다. HF SCF, target PBE, response operator, Z-vector, explicit electronic
gradient가 GPU에서 계산되고, 문서화된 작은 host boundary 외에는 CPU numerical
kernel로 조용히 우회하지 않는다는 dispatch contract이다.

## 2. 전체 계산 흐름

Public driver는 `dft.DCRKS`와 `dft.DCKRKS`이다. 전자는 Gamma-point, 후자는
sampled k-point를 소유한다.

```text
DCRKS/DCKRKS.kernel()
  -> scf.reference.build_reference()
  -> GPU RHF/KRHF + GPU FFTDF SCF
  -> dft.fixed_density.evaluate_energy()
  -> E_PBE[D_HF], F_PBE[D_HF], V_eff^PBE[D_HF]

mf.nuc_grad_method().kernel()
  -> grad.context.build_gradient_context()
  -> grad.response.build_target_orbital_gradient()
  -> grad.zvector.solve_adjoint_zvector()
  -> grad.one_electron + pseudo.nonlocal_derivative
  -> df.fft.jk_derivative
  -> grad.target + grad.pulay
  -> grad.nuclear_response
  -> total dE_DC-PBE/dR
```

CPU package와 GPU package는 같은 물리적 decomposition을 사용하지만, GPU package는
CPU gradient object를 감싸지 않는다. GPU source SCF와 target object를 새로 만들고
CuPy array를 통하여 독립적으로 계산한다.

## 3. GPU HF source density

`scf/reference.py::build_reference`는 Gamma-point에서는 GPU RHF를, 일반 k-point에서는
GPU KRHF를 만든다. `with_df`에는 `gpu4pyscf_pbc_dcdft.df.FFTDF`가 소유한 실제
GPU4PySCF FFTDF object가 연결된다.

RHF density matrix는 closed-shell occupation을 포함하여

$$
D_{\mu\nu}^{\mathbf{k}}
= \sum_i^{\mathrm{occ}} 2 C_{\mu i}^{\mathbf{k}}
  C_{\nu i}^{\mathbf{k}*}
$$

로 정의된다. k-point 평균은 energy와 response contraction에서 일관되게
$1/N_k$를 포함한다. 작은 Gamma cell에서 PySCF가 in-core ERI로 자동 전환하는 것과
달리, 이 route는 선택된 FFTDF provider를 명시적으로 소유한다.

## 4. Frozen-density target PBE

`dft/fixed_density.py::evaluate_energy`는 HF density를 고정한 채 PBE energy,
target Fock 및 target effective potential을 계산한다.

$$
E_{\mathrm{DC-PBE}}[D^{\mathrm{HF}}]
= \operatorname{Tr}[D^{\mathrm{HF}} h]
+ E_J[D^{\mathrm{HF}}]
+ E_{\mathrm{xc}}^{\mathrm{PBE}}[n_{\mathrm{HF}}]
+ E_{\mathrm{nuc}}.
$$

PBE orbital로 재최적화하지 않기 때문에 이 energy는 HF orbital variation에 대해
stationary하지 않다. 따라서 ordinary RKS gradient만 계산해서는 안 되며 HF CPHF
response 또는 동등한 adjoint Z-vector correction이 필요하다.

GPU target object에는 ordinary `NumInt/KNumInt`를 명시적으로 연결한다. Energy,
target Fock과 explicit XC derivative가 같은 quadrature와 mesh를 사용하도록 provider
ownership을 고정한다.

## 5. Real-linear HF Hessian과 Z-vector

Complex k-point orbital rotation을 $x=x_R+i x_I$로 쓰면 HF orbital Hessian은
복소 선형이 아니라 real-linear operator이다. 구현은 독립적인 실수 변수로 펼친
matrix를 GPU에서 구성한다.

$$
\mathcal{A}_{\mathrm{R}}
\begin{pmatrix}x_R\\x_I\end{pmatrix}
=
\begin{pmatrix}
\Re \mathcal{A}[x]\\
\Im \mathcal{A}[x]
\end{pmatrix}.
$$

Target orbital gradient를 $g_R$라 하면 adjoint equation은

$$
\mathcal{A}_{\mathrm{R}}^{T} z_R = -g_R
$$

이다. `grad/response.py`는 GPU J/K response로 Hessian action을 만들고,
`grad/zvector.py`는 `cupy.linalg.solve`로 adjoint equation을 푼다. 이때 source HF
kernel은

$$
G^{\mathrm{HF}}[D] = J[D] - \frac{1}{2}K[D]
$$

이며 closed-shell occupation factor와 k-point normalization은 CPU implementation과
동일하게 유지한다.

## 6. Nuclear response RHS

핵 변위 $R_A^x$에 대한 occupied-virtual response source는 개략적으로

$$
B_{ai}^{A x}
= (H_1^{A x})_{ai}
- (S_1^{A x})_{ai}\epsilon_i
- \frac{1}{2}(\mathcal{A}[S_1^{A x}])_{ai}
$$

이다. 실제 구현은 Gamma와 k-point 모두 `(A,k,x,a,i)` convention을 사용한다.
Z-vector correction은

$$
E_{\mathrm{resp}}^{A x}
= \frac{4}{N_k}\Re\sum_{\mathbf{k}ai}
Z_{ai}^{\mathbf{k}*}B_{ai}^{A x,\mathbf{k}}
$$

로 contraction된다. `4`에는 closed-shell occupation과 real-part symmetrization이
포함되며, 임의의 fitting coefficient가 아니다.

## 7. FFTDF J/K nuclear derivative

GPU4PySCF 1.7.4의 periodic FFTDF에는 energy용 `get_jk`가 있지만, DC response가
필요로 하는 full AO matrix 형태의 `get_k_e1`/`get_k_e1_kpts`가 구현되어 있지 않다.
따라서 `df/fft/jk_derivative.py`가 project-local CuPy kernel을 제공한다.

출력 shape은 J와 K 모두

```text
(natm, nkpts, 3, nao, nao)
```

이고 dtype은 일반 k-point에서 `complex128`이다. 이는 ordinary variational gradient가
사용하는 atom-contracted scalar가 아니라, Z-vector와 arbitrary transition density에
다시 contraction할 수 있는 full derivative matrix이다.

Reciprocal-space Coulomb kernel을 `v(G)`라 하면 J의 대표적인 구조는

$$
J_{\mu\nu}^{\mathbf{k}}
= \frac{1}{\Omega}\sum_{\mathbf{G}}
\rho(\mathbf{G})v(\mathbf{G})
\rho_{\mu\nu}^{\mathbf{k}}(-\mathbf{G})
$$

이다. Nuclear derivative는 AO pair Fourier transform과 density Fourier component의
center derivative를 모두 포함한다. Exchange는 k-point pair와 momentum transfer를
직접 순회하는 correctness-first kernel이다. 향후 block/symmetry optimization을 할 수
있지만 수식을 바꾸지는 않는다.

$G=0$은 GPU4PySCF/PySCF FFTDF의 Coulomb 및 Ewald convention을 그대로 공유한다.
별도의 경험적 constant나 후처리 correction을 넣지 않는다.

## 8. GTH pseudopotential derivative

One-electron derivative는 kinetic, local GTH, nonlocal GTH를 포함한다.
`pseudo/nonlocal_derivative.py`는 nonlocal projector의 다음 세 의존성을 구분한다.

1. projector center phase derivative
2. Fourier AO center derivative
3. radial/projector coefficient contraction

작은 GTH radial/projector polynomial table은 host에서 준비하지만, Fourier AO,
projector 및 derivative matrix contraction은 GPU에서 수행한다. 이는 큰 numerical
workload를 CPU로 위임하는 fallback이 아니라 metadata-sized setup boundary이다.

## 9. Target XC와 Pulay term

PBE XC derivative는 GPU numerical integration grid에서 AO row derivative를 평가한다.
Target Fock은 HF orbital이 만족하는 stationary Fock과 다르므로 overlap derivative와
energy-weighted density만 ordinary HF 방식으로 대체할 수 없다. 구현은 target-Fock
Pulay contribution과 Z-vector correction을 별도로 보존한다.

주의할 오류는 다음과 같다.

- gradient $dE/dR$와 force $-dE/dR$의 부호 혼동
- AO derivative가 bra 또는 ket 어느 center에 작용하는지의 부호 혼동
- k-point sum의 $1/N_k$ 누락
- complex conjugation과 `Re`의 중복 적용
- closed-shell occupation factor `2`의 중복 또는 누락

## 10. Host/device 경계

현재 허용된 host boundary는 다음뿐이다.

- PySCF `Cell`, lattice, AO slice 및 mesh metadata
- 작은 GTH radial/projector setup table
- PySCF Ewald ion-ion gradient
- 최종 NumPy gradient 및 metadata reporting

HF SCF, target PBE, FFT J/K, full J/K derivative, nonlocal GTH contraction,
AX/Z-vector와 final electronic assembly는 GPU route이다. 이 계약은
`gpu4pyscf_pbc_dcdft.capability()`에서 machine-readable하게 확인할 수 있다.

## 11. Timing metadata

GPU kernel은 asynchronous이므로 각 공개 timing 경계에서 default stream을
synchronize한다. Energy object는 다음 항목을 기록한다.

- `hf_scf`
- `target_pbe_energy`
- `energy_total`

Gradient result는 다음 항목을 기록한다.

- `target_fock_and_rhs`
- `hf_ax_and_zvector`
- `one_electron_and_pseudopotential`
- `full_jk_nuclear_derivative`
- `target_coulomb_xc`
- `pulay_and_nuclear`
- `nuclear_response_contraction`
- `final_host_reporting`
- `gradient_total`

따라서 host launch 시간만 측정하는 실수를 피하고 실제 GPU 완료 시간을 비교할 수 있다.

## 12. 검증과 근거

Unit test:

- `tests/test_package_boundary.py`: capability와 no-silent-fallback contract
- `tests/test_fftdf_gpu.py`: Gamma/non-Gamma CPU-GPU parity

Full validation:

- `validation/validate_fftdf_gpu.py`
- `validation/results/2026-07-13_fftdf_gpu.json`

Full gate는 H2 및 distorted H2O, GTH-SZV/DZVP, Gamma 및 `(2,1,1)` k-mesh를
포함한다. Acceptance threshold는 다음과 같다.

- GPU analytic gradient 대 GPU displaced-energy central FD: $10^{-5}$ Ha/Bohr
- CPU-GPU energy: $10^{-8}$ Ha
- CPU-GPU gradient: $10^{-7}$ Ha/Bohr

모든 case가 이 threshold를 통과했다. CPU와 GPU SCF가 같은 stationary point에 충분히
수렴하도록 parity gate에서는 공통 `conv_tol=1e-11`을 사용한다.

## 13. 사용 예제

- `pbc/gpu/examples/00_fftdf_gpu_energy_gradient.py`
- `pbc/gpu/examples/01_fftdf_gpu_kpoint.py`
- `pbc/gpu/examples/04_cpu_gpu_parity.py`
- `pbc/gpu/examples/05_gpu_timing_breakdown.py`
- `pbc/gpu/examples/06_device_and_fallback_report.py`

GDF example은 아직 제공하지 않는다. 미완성 route를 실행 가능한 것처럼 보이게 하지
않는 것이 package contract의 일부이며, Stage 4/5 completion gate 뒤에 추가한다.
