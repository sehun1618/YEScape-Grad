# User Stress Benchmarks

현재 queue는 기존 PBC gradient benchmark와 system/numerical setting을 맞춘다.

- `current_h2_matrix.json`: CPU/GPU, GDF/RSDF, sampled-k 전체 API smoke
- `current_water8_correctness.json`: selected stress FD와 combined output
- `current_water20_sz.json`, `current_water20_sz_pbe.json`, `current_water20_dz.json`
- `current_water20_sz_stress_scaling.json`: production 승격용 최소 GDF/RSDF stress-only gate
- `current_water64_sz.json`, `current_water64_dz.json`
- `current_water_ethanol_sz.json`

Water64 queue는 사용자가 직접 실행하는 scaling input이다. Development 자동 gate에서
실행하지 않는다. 결과는 반드시 `user/results/` 아래에 저장된다.

GPU-only current queue는 검증된 exact optimized owner를 명시적으로 선택한다.

```json
"stress_pp_response_backend":
  "gpu4pyscf_native_contracted_analytic_nonlocal_device",
"stress_df_response_backend": "host_analytic_source_adjoint_gpu_ft"
```

이는 queue 정책이며 package Python API의 보수적인 기본값을 몰래 바꾸지 않는다.
CPU/GPU가 섞인 `current_h2_matrix.json`은 각 device의 기본 경로를 검증하기 위해 이
GPU-only override를 공통 defaults에 넣지 않는다.

모든 current queue는 `checkpoint_resume: true`를 명시한다. 중단된 계산은 같은 명령으로
재실행하며, 완료된 source/gradient/stress/strain-FD geometry를 버리고 처음부터 계산할
때만 `--overwrite`를 사용한다.

최소 scaling gate는 `stress/benchmark`에서 다음과 같이 실행한다.

```bash
conda run -n gpu python run_stress_queue.py \
  --queue user/queues/current_water20_sz_stress_scaling.json
```

완료 후 같은 위치에서 release gate를 검사한다.

```bash
python validate_scaling_gate.py \
  user/results/current_water20_sz_stress_scaling
```
