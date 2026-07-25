# PBC DC-DFT Stress Benchmarks

이 directory는 기존 PBC gradient benchmark와 같은 system/basis/backend 입력으로
analytic stress 또는 gradient+stress를 실행한다.

```text
benchmark/
  run_stress_queue.py   queue 실행/skip/summary
  stress_benchmark.py  한 case 계산
  show_stress_progress.py
  structures/          H2, Water8/20/64, water-ethanol
  user/queues/         사용자가 실행하는 queue
  user/results/        사용자 결과
  development/         구현용 small-system gate
```

## 실행

```bash
cd stress/benchmark
conda run -n gpu python run_stress_queue.py \
  --queue user/queues/current_water20_sz.json
```

완료된 case의 input/source fingerprint가 같으면 자동으로 skip한다. 같은 계산을 다시
실행할 때만 `--overwrite`를 쓴다.

긴 stress/FD 계산은 default `checkpoint_resume: true`로 case의 `checkpoints/`를
사용한다. Source-HF density/MO state, 선택적 gradient와 Z-vector block, analytic stress,
각 strained-cell FD energy와 density를 완료 즉시 checksum이 붙은 NPY/JSON으로 원자적으로
저장한다. 중단 후 같은 queue 명령을 `--overwrite` 없이 다시 실행하면 완료된 stage와
strain geometry를 재사용한다. 설정, XYZ 또는 source가 바뀌어 fingerprint가 달라지거나
`--overwrite`를 지정한 경우에만 checkpoint를 폐기한다.

PySCF/GPU4PySCF 객체 자체는 pickle하지 않는다. Energy까지만 완료된 재시작에서는 저장된
density로 source SCF를 warm-start하고, stress까지 완료됐다면 numerical stress state로
report/MLIP 출력을 직접 재구성한다.

Stress의 2c/3c strain integral에는 package 전용 C extension이 필요하다. Runner는 현재
Python ABI에 맞는 `_sr3c_strain`이 source tree에 없으면 첫 case 실행 전에 자동으로
`stress/cpu/setup.py build_ext --inplace`를 수행한다. 별도의 수동 설치 명령은 필요 없다.
기존 binary가 있더라도 `setup.py`/`c_ext/*.c`가 더 새롭거나 현재 fused Fourier ABI
symbol이 없으면 stale binary로 판단해 다시 빌드한다.

```bash
conda run -n gpu python run_stress_queue.py \
  --queue user/queues/current_water20_sz.json --only rsdf_stress_only

conda run -n gpu python run_stress_queue.py \
  --queue user/queues/current_water20_sz.json --overwrite
```

현재 진행 상황은 다음처럼 확인한다.

```bash
python show_stress_progress.py user/results/current_water20_sz
```

## 계산 mode

- `stress_only`: source HF/target energy 이후 analytic stress만 계산한다. Nuclear
  gradient route를 호출하지 않는다.
- `gradient_and_stress`: analytic gradient와 stress를 모두 계산한다. Gradient에서 푼
  adjoint Z-vector를 stress에 재사용하며 MLIP JSON/extxyz도 저장한다.

한 job에서 둘 다 실행하려면 다음처럼 쓴다.

```json
{
  "device": "gpu",
  "df_backends": ["gdf", "rsdf"],
  "calculation_modes": ["stress_only", "gradient_and_stress"]
}
```

지원 범위는 CPU/GPU, GDF/RSDF, Gamma/sampled-k, PBE/r2SCAN과 uniform XC다.
현재 stress가 지원하지 않는 FFTDF, multigrid, D2C/dispersion, open-shell 입력은 다른
route로 바꾸지 않고 오류로 종료한다.

## Stress FD

작은 correctness job에는 선택적인 central strain FD를 사용할 수 있다.

```json
"run_stress_fd": true,
"stress_fd_components": ["xx", "xy"],
"stress_fd_step": 0.00025,
"stress_fd_tolerance_ha_per_bohr3": 1e-5
```

FD는 각 strained cell에서 같은 device/backend의 source HF와 target energy를 다시
계산한다. 다만 중앙 cell은 이미 계산한 source state를 재사용하며, 각 `+/-` strained
cell은 완료 즉시 별도 checkpoint로 저장된다. FD 시간은 analytic stress 시간과 별도로
기록된다.

각 case에는 `report.json`, `progress.json`, `stdout.txt`, `stderr.txt`, NPY arrays가
생긴다. Combined mode는 `mlip_frame.json`과 `mlip_frame.xyz`도 생성한다.

## Host reciprocal memory

GDF/RSDF stress의 3-center source와 metric strain은 G-vector block을 최종 응답에
즉시 수축한다. 기본 block은 약 8 GiB의 host working-set 추정값으로 자동 결정된다.
PP와 DF를 함께 명시적으로 조정하는 기존 option은 다음과 같다.

```json
"stress_host_reciprocal_block_size": 4096
```

DF source/metric만 제한하고 PP 경로는 그대로 유지하려면 다음 option을 사용한다.

```json
"stress_df_host_reciprocal_block_size": 4096
```

두 값이 모두 있으면 PP는 `stress_host_reciprocal_block_size`, DF는
`stress_df_host_reciprocal_block_size`를 사용한다. DF 전용 값이 없으면 기존 동작과
호환되도록 global 값을 상속한다. Memory/OOM 진단에는 서로 다른 항을 동시에 바꾸지 않는
DF 전용 option을 권장한다.

큰 값은 Fourier integral 호출 횟수를 줄이지만 host RAM peak를 높인다. 작은 값은 RAM을
줄이지만 block 호출 overhead가 늘어난다. `report.json`의
`stress.metadata.df_parameters.stress_reciprocal_block_size`와
`memory.host_peak_rss_bytes`에서 실제 선택값과 전체 process peak를 확인한다. 요청값은
`stress.metadata.host_reciprocal_block_size_requested`와
`stress.metadata.df_host_reciprocal_block_size_requested`에 분리해 기록된다.

Water8 exact direct-contracted 측정에서 DF block 4096은 GDF/RSDF peak RSS를 각각
52.8%/33.8% 줄였지만 wall time을 19.4%/15.3% 늘렸다. 따라서 현재 speed default는 자동
block이고 4096은 memory/OOM opt-in이다. Block 8192는 이 시스템에서 더 느렸으므로
일반적인 절충 기본값으로 간주하지 않는다.

`stress_df_diagonal_cache_bytes`는 이전 component-stream queue와의 입력 호환을 위해
남아 있다. 현재 single-pass Voigt DF response는 diagonal $dC$를 재사용해야 하는 중복
순회 자체를 제거했으므로 이 값과 관계없이 별도 cache를 만들지 않는다.

진행 상태는 다음 analytic-stress substage를 기록한다.

- `stress_target_fock_and_zvector`
- `stress_host_one_electron_pp`
- `stress_df_host_primitives`
- `stress_df_device_response`
- `stress_xc_lagrangian_assembly`

2026-07-22 최적화의 정확성, current-load timing, 메모리 판정은
`../docs/S7_Optimization_Evaluation_2026-07-22.md`와
`development/results/stress_evaluation_2026-07-22/summary.json`에 정리되어 있다.
Water8/400 Ha 확장성 평가는 `../docs/S7_Water8_Evaluation_2026-07-22.md`와
`development/results/water8_optimization_evaluation_2026-07-22/summary.json`에 있다.

## Experimental GPU contracted response

큰 uniform mesh의 PP/DF strain matrix 생성을 피하는 development option은 다음과 같다.

```json
"stress_pp_response_backend": "gpu4pyscf_native_contracted",
"stress_df_response_backend": "gpu_fixed_density_fd_contracted",
"stress_df_contracted_fd_step": 1e-5
```

PP grid working set을 강제로 더 제한할 때만
`stress_pp_grid_block_size`를 양의 point 수로 지정한다. 지정하지 않으면 AO 수,
k-point 수와 density 수로부터 GPU4PySCF 방식의 memory block을 자동 선택한다.

두 option은 기본값이 아니다. Source HF, target DFA와 Z-vector는 중앙 state에서 한 번만
계산하며, PP/DF에 필요한 최종 scalar contraction만 낮은 수준의 centered cell
displacement로 미분한다. 따라서 report의
`stress.metadata.low_level_numerical_components`에 해당 성분이 기록되고
`stress.metadata.production`은 `false`가 된다.

H2 parity queue는
`development/queues/h2_contracted_df_validation_2026-07-22.json`, Water8 성능/FD queue는
`development/queues/water8_gpu_pp_df_optimization_2026-07-22.json`이다. 결과와 적용
범위는 `../docs/S7_GPU_Contracted_Response_2026-07-22.md`를 따른다.

DF derivative는 analytic으로 유지하면서 J/K AO response를 만들지 않는 development
route는 다음과 같다.

```json
"stress_df_response_backend": "host_analytic_direct_contracted"
```

각 canonical k-pair를 source/metric build 직후 six-direction factor response와 두 density
scalar에 수축하고 폐기한다. `stress.metadata.df_memory_estimate`에서 retained pair
derivative, central factor와 AO response가 모두 0인지 확인할 수 있다. H2 gate queue는
`development/queues/optimization_o4_h2_direct_df.json`이다.

이 direct route는 기본적으로 exact `auto` AO-pair layout을 사용한다. 한 reciprocal
image가 256 MiB 미만이면 기존 assembled contraction, 그 이상이면 center/moment-first
factorized contraction을 선택한다. 이는 성능용 행렬곱 재결합이며 mesh나 integral
정확도를 바꾸지 않는다. 명시적 factorized gate는
`development/queues/optimization_o4_h2_factorized_source.json`과
`development/queues/optimization_o4_water8_factorized_source.json`이다.

GPU projection의 bounded-transfer 검증은
`development/queues/optimization_o4_h2_factorized_device_bounded.json`과
`development/queues/optimization_o4_water8_factorized_device_bounded.json`을 사용한다.
이 경로는 exact opt-in이며 큰 sampled-k와 Water20/64 gate 전에는 자동 기본값이 아니다.

Reverse-source GPU FT gate는 다음 development queue가 소유한다.

- `optimization_o4_h2_gpu_ft_source_adjoint.json`
- `optimization_o4_water8_gamma_gpu_ft_source_adjoint.json`
- `optimization_o4_water8_k211_gpu_ft_source_adjoint.json`
- `optimization_o4_water8_gdf_gamma_gpu_ft_source_adjoint.json`
- `optimization_o4_water8_gdf_k211_gpu_ft_source_adjoint.json`
- `optimization_o4_water8_gamma_gpu_ft_repeats.json`
- `optimization_o4_water8_gdf_gamma_gpu_ft_repeats.json`
- `optimization_o4_h2_dz_gpu_ft_source_adjoint.json`

The current GPU user queues select both exact optimized response owners:

```json
"stress_pp_response_backend":
  "gpu4pyscf_native_contracted_analytic_nonlocal_device",
"stress_df_response_backend": "host_analytic_source_adjoint_gpu_ft"
```

This is an explicit queue policy, not a hidden change to the Python API
defaults. Water20/64 runs remain user-owned scaling evidence.

PP의 CUDA stream interval과 Python wall을 함께 진단하려면
`"stress_detailed_profile": true`를 사용한다. 성능 median에는 profiler가 꺼진 run만
사용한다.

Nonlocal GTH PP를 cell displacement나 host AO derivative matrix 없이 analytic하게
직접 수축하는 development option은 다음과 같다.

```json
"stress_pp_response_backend":
  "gpu4pyscf_native_contracted_analytic_nonlocal_device"
```

큰 FT-AO first-moment와 density contraction은 GPU가 소유하고, 작은 projector
radial/angular template만 host에서 만든다. 실제 경계는
`stress.metadata.pp_response_metadata.work_profile.nonlocal.host_boundaries`에서 확인한다.
Water1/Water8 gate queue는 각각
`development/queues/optimization_o3_water1_device_analytic_nonlocal_pp.json`과
`development/queues/optimization_o3_water8_device_analytic_nonlocal_pp.json`이다.
