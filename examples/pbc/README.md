# PBC examples

PBC 예제는 PySCF 예제와 같이 각 case의 객체 생성부터 `kernel()`,
`nuc_grad_method()`와 출력까지를 직선형 코드로 보여준다. 모든 조합을 선택하는 CLI
driver는 사용하지 않는다. 같은 수학을 공유하는 변형은 한 파일에 `Case 1`, `Case 2`로
묶고, CPU/GPU 또는 GDF/RSDF처럼 객체와 파라미터 체계가 실질적으로 달라지는 경우만
파일을 분리한다.

| 순서 | Example | 포함된 case |
|---|---|---|
| 1 | `01-cpu-rhf_dft.py` | CPU FFTDF, GDF, RSDF PBE@RHF energy와 gradient |
| 2 | `02-gpu-rhf_dft.py` | GPU FFTDF, GDF, RSDF energy와 gradient |
| 3 | `03-multigrid.py` | provider-J + MG-XC와 target MG-J+XC 선택법 |
| 4 | `04-gpu-scanner.py` | Scanner 생성, energy/gradient curve, selected FD |
| 5 | `05-gpu-geometric_opt.py` | Gradient scanner와 geomeTRIC 연결 |
| 6 | `06-numerical_controls.py` | Cutoff convergence와 CPHF convergence |
| 7 | `07-gdf_options.py` | GPU GDF production 설정과 numerical oracle |
| 8 | `08-rsdf_options.py` | GPU RSDF source/metric 설정과 storage policy |
| 9 | `09-cpu-kpts-rhf_dft.py` | CPU full-BZ FFTDF/GDF/RSDF PBE@KRHF energy/gradient와 one-Gamma limit |
| 10 | `10-cpu-kpts-zvec.py` | CPU sampled-k dense/matrix-free real-linear Z-vector 진단 |
| 11 | `11-cpu-kpts-xc.py` | 같은 KRHF density의 sampled-k LDA/GGA/R2SCAN energy/gradient와 CPU MG 경계 |
| 12 | `12-gpu-kpts-rhf_dft.py` | GPU full-BZ FFTDF/GDF/RSDF semilocal-DFA@KRHF와 selectable MG Hartree |
| 17 | `17-gdf-mixed-source.py` | Real-Gamma GDF source HF의 guarded mixed J/K와 native-FP64 audit/fallback |
| 18 | `18-gdf-precision-profiles.py` | GDF `fp64`, moderate `mixed`, aggressive experimental `fp32_build` 전체-workflow profile |

Repository root에서 파일을 그대로 실행한다.

```bash
PYTHONPATH=src conda run --no-capture-output -n gpu \
  python examples/pbc/01-cpu-rhf_dft.py

PYTHONPATH=src conda run --no-capture-output -n gpu \
  python examples/pbc/02-gpu-rhf_dft.py

PYTHONPATH=src conda run --no-capture-output -n gpu \
  python examples/pbc/03-multigrid.py

PYTHONPATH=src conda run --no-capture-output -n gpu \
  python examples/pbc/04-gpu-scanner.py

PYTHONPATH=src conda run --no-capture-output -n gpu \
  python examples/pbc/05-gpu-geometric_opt.py

PYTHONPATH=src conda run --no-capture-output -n gpu \
  python examples/pbc/06-numerical_controls.py

PYTHONPATH=src conda run --no-capture-output -n gpu \
  python examples/pbc/07-gdf_options.py

PYTHONPATH=src conda run --no-capture-output -n gpu \
  python examples/pbc/08-rsdf_options.py

PYTHONPATH=src conda run --no-capture-output -n gpu \
  python examples/pbc/09-cpu-kpts-rhf_dft.py

PYTHONPATH=src conda run --no-capture-output -n gpu \
  python examples/pbc/10-cpu-kpts-zvec.py

PYTHONPATH=src conda run --no-capture-output -n gpu \
  python examples/pbc/11-cpu-kpts-xc.py

PYTHONPATH=src conda run --no-capture-output -n gpu \
  python examples/pbc/12-gpu-kpts-rhf_dft.py

PYTHONPATH=src conda run --no-capture-output -n gpu \
  python examples/pbc/17-gdf-mixed-source.py

PYTHONPATH=src conda run --no-capture-output -n gpu \
  python examples/pbc/18-gdf-precision-profiles.py
```

## Case를 바꾸는 방법

PySCF 예제처럼 바꾸어야 할 줄을 case 바로 옆에 주석으로 제시한다. 예를 들어
multigrid 예제는 FFTDF를 기본으로 사용한다. GDF 또는 RSDF로 바꾸는 것은 다음 한 줄의
교체다.

```python
mf = CPURHFDFT(cell, xc='PBE').fft_density_fit()
# mf = CPURHFDFT(cell, xc='PBE').density_fit(auxbasis='gth-szv')
# mf = CPURHFDFT(cell, xc='PBE').rs_density_fit(auxbasis='gth-szv')
```

Target functional도 객체를 만드는 줄에서 직접 바꾼다.

```python
mf = RHFDFT(cell, xc='PBE')
# mf = RHFDFT(cell, xc='R2SCAN')
```

이 방식은 실행 option의 의미를 별도로 추적하지 않고, 실제로 어떤 PySCF/DC-Forge 객체가
만들어지는지 소스에서 바로 확인하게 한다.

## 파라미터를 읽는 기준

| 종류 | 대표값 | 해석 |
|---|---|---|
| 정확도/표현 | `cell.ke_cutoff`, `cell.precision`, `cell.mesh` | 유한 수치 표현을 바꾸므로 energy와 gradient를 함께 수렴시킨다. |
| SCF/response 수렴 | `conv_tol`, `cphf_conv_tol`, `cphf_residual_tol` | 고정된 표현에서 반복해의 정확도를 정한다. |
| 동등한 알고리즘 경로 | `df_cphf_response_mode`, `df_cphf_response_cache`, `gdf_response_mode`, `hcore_method`, Gamma conjugate reduction | 같은 수학을 다른 contraction, storage 또는 oracle로 계산한다. `direct_vo`와 bounded `auto` cache가 GDF CPHF 기본값이고 `recompute` cache 및 `ao_oracle`이 storage/full-AO oracle이다. |
| source-SCF 정밀도 정책 | `source_precision_policy='fp64'`, `source_mixed_precision(...)` | 기본은 native FP64다. 명시적 mixed 경로는 FP32 factor J/K에서 시작해 native FP64로 단조 전환하며, 전환마다 DIIS와 incremental Fock을 비운다. |
| response 정밀도 정책 | `precision_policy='fp64'`, `'mixed'`, `'mixed_lowmem'` | 두 mixed 정책은 real-Gamma GDF direct-VO Z action만 FP32 GEMM/FP64 block sum으로 계산하고 독립 FP64 physical residual로 검사한다. |
| 전체 GDF 정밀도 profile | `gdf_precision_profile='fp64'`, `'mixed'`, `'fp32_build'` | `fp64`가 기본 reference다. `mixed`는 FP64로 CDERI를 만든 뒤 FP32로 저장하고 mixed SCF/Z/derivative를 묶는다. `fp32_build`는 CDERI 변환까지 FP32/complex64로 낮추는 aggressive experimental 경로다. |
| storage policy | CDERI storage, reciprocal block size와 memory fraction | Intermediate 크기와 batching을 조절한다. |
| 자동 결정값 | RSDF source/metric 설정과 local-GTH part-1 plan | Cell과 precision에 따라 backend가 resolve한다. |

RSDF의 source `(omega, mesh_compact)`, metric `(omega_j2c, mesh_j2c)`와 local-GTH
part-1 plan은 서로 다른 work split이다. `07-gdf_options.py`와
`08-rsdf_options.py`는 이를 일반 single-point 예제에서 분리해 보여준다.

GPU GDF/RSDF의 `cderi_storage='mapped_host'`는 압축 factor를 CUDA mapped
pinned host memory에 유지하는 기본값이고, `'hdf5'`는 같은 factor를 디스크에 저장해
auxiliary block으로 읽는다. `cderi_preflight_max_bytes`는 auxbasis를 자동 축소하지 않고
명시적 한도를 넘으면 build 전에 실패한다. `release_cderi_after_response=True`는 Z-vector와
마지막 overlap-response action 이후 factor를 해제하므로, 이후 같은 객체에서 새 SCF를
시작하면 CDERI를 다시 만든다.

## GDF precision profile

전체 workflow에서 검증된 precision 조합을 한 번에 선택하려면
`18-gdf-precision-profiles.py`의 `set_gdf_precision_profile()`을 사용한다.

`mixed`는 native FP64 primitive/metric/CDERI transformation 뒤의 완성된 factor만 FP32로
저장하고, persistent-FP32 source SCF, mixed direct-VO Z-vector와 FP32 MO
derivative-source contraction을 함께 선택한다. 마지막 source tier는 현재 FP32 factor
block을 device FP64로 승격해 J/K를 다시 계산하고, Z-vector는 별도의 FP64 physical
residual로 검사한다. 이 검사는 이미 FP32로 양자화된 Hamiltonian 안에서의 수렴 검사이며
원래 CDERI bit를 복원하지 않는다. 대표 production 구조 일부에서 `fp64` energy와 force를
비교한 뒤 사용하는 moderate approximation이다.

`fp32_build`는 CDERI metric transformation과 reciprocal contraction의
FP32/complex64 계산까지 더한다. 변환 과정의 반올림 오차가 누적될 수 있고,
SCF/CPHF residual이 작아도 `fp64` energy/force와의 bias는 클 수 있다. 따라서 이 mode는
aggressive experimental로 분류되며, 실제 basis/auxbasis/cutoff와 구조 분포를 대표하는
subset에서 `fp64` energy와 gradient RMSD/max error를 별도로 통과시킨 뒤에만 MLIP label
생성에 사용해야 한다. Reference를 복구하려면 CDERI를 FP64로 새로 build해야 한다.

어느 profile도 numerical threshold, auxbasis 또는 mesh를 자동으로 느슨하게 만들지 않는다.
현재 accelerated profile은 single-GPU 3D positive-metric real-Gamma GDF energy와 nuclear
gradient까지만 지원한다. Stress, sampled-k, RSDF 및 provider-factor checkpoint는
fail-closed하며 `fp64` rebuild가 필요하다.

RSDF의 기본 `response_mode='sequential_packed'`는 unique Gamma AO pair로 target-J와
source-Fock을 차례로 처리한다. `'dense'`는 값 비교용 oracle이다. GPU real-Gamma
closed-shell GDF는 `gdf_adjoint_backend='auto'`에서 exact MO-direct response를 사용해
`Boo=C.T@B@C`, `Bou=C.T@B@U`만 metric solve하고 persistent packed `B/W/source`를
만들지 않는다. 명시적 `'packed'`는 값 비교용 oracle과 unsupported-representation
fallback으로 남아 있다. 이 storage 선택들은 수식이나 DF auxiliary space를 바꾸지
않는다. 반면 작은 auxbasis를 선택하는 것은 DF 근사를 바꾸므로 별도의
energy/gradient 검증이 필요하다.

## 범위와 검증

모든 gradient는 fixed-cell `g=dE/dR`이고 force는 `F=-g`이며 단위는 `Ha/Bohr`다.
`09-cpu-kpts-rhf_dft.py`는 full-BZ sampled-k FFTDF K1/K3, complex-pair GDF K5와
independent source/metric RSDF K6의 public call을 함께 보여준다.
`10-cpu-kpts-zvec.py`는 small-system dense oracle K2와 matrix-free GMRES K4를 나란히
비교한다. `11-cpu-kpts-xc.py`는 같은 converged KRHF density에서 LDA, GGA와
k-resolved-tau R2SCAN을 계산한다. CPU sampled-k multigrid는 PySCF 2.14 pair-MG의
Gamma-only 제한 때문에 즉시 실패하며, uniform-grid 결과로 조용히 대체하지 않는다.
`12-gpu-kpts-rhf_dft.py`는 동일한 explicit full-BZ 계약을 GPU FFTDF/GDF/RSDF와
GDF-source + target MG-J+XC에 적용하고, native/host response 경계를 출력한다.
Multigrid R2SCAN은 k-resolved tau를 포함한다. Sampled-k scanner,
stress/cell derivative, UHF/ROHF, PBC hybrid target, R2SCAN 이외 MGGA와 NLC/VV10는
현재 sampled-k 예제 범위가 아니다.

GPU sampled-k의 complex overlap projection, real-linear Z-vector, provider별 host
boundary, K9 storage 최적화와 Water8 release gate는
`docs/learning/24_gpu_pbc_sampled_k_gradient.md`에 term 순서대로 정리했다. 저장된 두
Water8 run이 현재 source/test tree와 같은지 다시 검사하려면 다음 CPU-only auditor를
실행한다.

```bash
conda run --no-capture-output -n gpu \
  python tools/validation/summarize_gpu_pbc_kpts_water8.py
```

Water2 `2x2x1`에서 production matrix-free Z, selected relaxed-source FD, response
workspace, stage timing와 peak RSS를 한 artifact로 다시 만들려면 다음을 실행한다.

```bash
conda run --no-capture-output -n gpu \
  python tools/validation/pbc_kpts_water2_matrix_free.py
```

두 validator는 SCF/CPHF tolerances와 provider/storage 설정을 JSON에 직접 기록한다.
Water8 auditor는 repeat와 FD artifact의 이 설정들이 서로 같지 않으면 numerical gate를
계산하지 않고 실패한다.

`04-gpu-scanner.py`는 analytic/FD identity가 SCF energy noise에 가려지지 않도록 source
SCF에 `conv_tol=1e-12`를 사용한다. Cutoff와 CPHF의 차이는
`06-numerical_controls.py`에서 각각 별도 case로 확인한다.

GDF Z-vector의 `df_cphf_response_cache='auto'`는 action workspace와 device reserve를
남긴 뒤 compressed CDERI와 occupied-transformed factor를 함께 보존할 수 있는지 판단한다.
명시적 `occupied`, `cderi`, `all`은 진단/재현용이고, `recompute`는 같은 수식의 storage
oracle이다. Cache는 한 Z solve가 끝나기 전에 `finally`에서 해제된다.

일반 사용자는 cache mode를 지정할 필요가 없다. GPU GDF gradient 객체의 기본값이 이미
`auto`이므로 다음처럼 계산하면 된다.

```python
mf = RHFDFT(cell, xc='R2SCAN').density_fit(auxbasis='weigend')
mf = mf.multigrid_numint(with_j=False).run()  # GDF J + multigrid XC
mf_grad = mf.nuc_grad_method().run()

print(mf_grad.df_cphf_response_cache)  # requested: auto
print(mf_grad.z_response_profile['resolved_cache_mode'])
```

96-GiB GPU의 Water64/GTH-DZVP/Weigend/XC-only-MG 측정에서는 `auto`가 37.80 GiB의
`all` cache를 선택했다. `recompute`, `occupied`, `cderi`, `all`, `auto`의 전체 gradient
시간은 각각 433.51, 368.83, 428.26, 360.49, 359.93초였고, cache별 gradient 최대 차이는
`5.84e-14 Ha/Bohr` 이하였다. 이 결과에서 raw CDERI만 유지한 `cderi`보다
`A_P=L_P C_occ`와 `H_P=A_P^T C_occ`를 유지한 `occupied`가 더 큰 효과를 냈다.
전체 peak VRAM은 cache가 Z-vector 뒤에 해제되고 후속 MO-direct GDF response가 지배하여
모든 mode에서 약 69.78 GiB였다. 이 수치는 장치와 계에 종속되므로 `all`을 고정된
production 기본값으로 해석하면 안 된다.

Scanner의 `z_warm_start=True`는 이전 raw MO amplitude를 그대로 쓰지 않는다. 서로 다른
geometry의 AO cross overlap으로 occupied/virtual subspace를 각각 투영하고, 현재 physical
residual이 diagonal guess보다 작을 때만 사용한다. 이 기능은 반복 횟수를 줄일 수 있으나
추가 residual action 때문에 기본값은 `False`다. 수식, profile field와 Water8/20/64
결과는 `docs/learning/27_gpu_pbc_gdf_optimization.md`에 정리했다.

현재 CuPy 14.0.1에서 upstream의 12.8-GB reduction 문제에 도달할 수 있는 대형 job은
Python 시작 전에 `CUPY_ACCELERATORS=cutensor,cub`을 설정한다. `unset`/CUB-only는
fresh child에서 CUDA illegal address가 재현됐고 `cutensor,cub`은 통과했다. 이는 GDF
수식 option이 아니라 CUDA process-safety 조건이며, 정확한 isolated reproducer와 결과는
learning chapter 27에 있다.

Scanner reset ownership, all-3N finite difference, CPU/GPU parity, peak VRAM, allocator
cache와 native endpoint benchmark는 공개 사용 예제가 아니라 `tools/tests`와
`tools/validation`이 담당한다.
