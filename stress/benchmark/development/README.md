# Development Stress Benchmarks

`benchmark_fd_stress.py` evaluates the independent CPU Gamma/GDF central-FD
oracle. It records every strained energy, mesh, volume, stress convention, and
wall time in JSON.

```bash
python benchmark_fd_stress.py structure.xyz \
  --cubic-cell-bohr 6 --mesh 15 15 15 --output fd-stress.json
```

`benchmark_semi_analytic_stress.py` retains its historical filename but now
measures the no-extra-HF analytic PBE/Gamma/GDF route. Add `--with-full-fd`
only for a small correctness gate:

```bash
python benchmark_semi_analytic_stress.py structure.xyz \
  --cubic-cell-bohr 6 --mesh 15 15 15 \
  --with-full-fd --output semi-analytic-stress.json
```

The JSON metadata must report an empty `low_level_numerical_components` list.
FD timing is separate and is never included in analytic stress wall time.

Queue runner의 end-to-end smoke와 selected stress FD는 다음으로 실행한다.

```bash
cd stress/benchmark
conda run -n gpu python run_stress_queue.py \
  --queue development/queues/smoke_h2.json
```

이 결과는 `development/results/`에만 저장하며 user scaling result와 섞지 않는다.

Water20 host-memory repair는 대형 회귀라 자동 test에서 실행하지 않는다. 사용자가 다음
queue로 peak RSS와 component timing을 확인한다.

```bash
cd stress/benchmark
conda run -n gpu python run_stress_queue.py \
  --queue development/queues/water20_gdf_host_memory_gate.json
```

Stress optimization O0는 correctness와 timing을 분리한다.

```bash
# H2 exact/contracted, Gamma/k211, GDF/RSDF, six-component FD
conda run -n gpu python run_stress_queue.py \
  --queue development/queues/optimization_o0_h2_correctness.json

# Water8 selected xx/xy FD. 실행 시간이 길 수 있다.
conda run -n gpu python run_stress_queue.py \
  --queue development/queues/optimization_o0_water8_correctness.json

# Idle GPU에서만 실행하는 explicit three-repeat timing queue
conda run -n gpu python run_stress_queue.py \
  --queue development/queues/optimization_o0_water8_timing.json
```

Timing queue는 `checkpoint_resume=false`다. 반복 job마다 source state를 새로 만들며,
checkpoint reuse가 섞인 결과를 median으로 사용하지 않는다.

Device-contracted analytic nonlocal PP correctness/profile gate는 다음과 같다.

```bash
conda run -n gpu python run_stress_queue.py \
  --queue development/queues/optimization_o3_water1_device_analytic_nonlocal_pp.json

conda run -n gpu python run_stress_queue.py \
  --queue development/queues/optimization_o3_water8_device_analytic_nonlocal_pp.json
```

첫 queue는 GDF/RSDF six-component FD, 두 번째는 selected `xx/xy` FD와
Water8 wall/RSS/VRAM profile을 소유한다.

O4 AO-pair source profile과 factorized gate는 다음으로 실행한다.

```bash
conda run -n gpu python run_stress_queue.py \
  --queue development/queues/optimization_o4_water8_forward_source_profile.json

conda run -n gpu python run_stress_queue.py \
  --queue development/queues/optimization_o4_h2_factorized_source.json

conda run -n gpu python run_stress_queue.py \
  --queue development/queues/optimization_o4_water8_factorized_source.json

conda run -n gpu python run_stress_queue.py \
  --queue development/queues/optimization_o4_h2_factorized_device_bounded.json

conda run -n gpu python run_stress_queue.py \
  --queue development/queues/optimization_o4_water8_factorized_device_bounded.json

conda run -n gpu python run_stress_queue.py \
  --queue development/queues/optimization_o4_h2_gpu_ft_source_adjoint.json

conda run -n gpu python run_stress_queue.py \
  --queue development/queues/optimization_o4_water8_gamma_gpu_ft_source_adjoint.json

conda run -n gpu python run_stress_queue.py \
  --queue development/queues/optimization_o4_water8_k211_gpu_ft_source_adjoint.json

conda run -n gpu python run_stress_queue.py \
  --queue development/queues/optimization_o4_water8_gdf_gamma_gpu_ft_source_adjoint.json

conda run -n gpu python run_stress_queue.py \
  --queue development/queues/optimization_o4_water8_gdf_k211_gpu_ft_source_adjoint.json

conda run -n gpu python run_stress_queue.py \
  --queue development/queues/optimization_o4_h2_dz_gpu_ft_source_adjoint.json
```

`optimization_o4_water8_source_adjoint_blocks.json`과
`optimization_o4_water8_device_gemm.json`은 실패한 성능 가설의 재현용이다. 정확하지만
production speed route로 사용하지 않는다.
