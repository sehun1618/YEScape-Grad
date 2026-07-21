# User Benchmarks

이 directory는 사용자가 직접 실행하는 안정 benchmark만 보관한다.

## 운영 규칙

- queue는 `queues/`, 새 결과는 `results/`에만 둔다.
- Codex는 사용자가 명시적으로 요청하지 않는 한 이 queue를 수정하거나 실행하지 않는다.
- 구현 중인 route, parameter sweep, profiling 및 실패 재현은 넣지 않는다.
- production 기본값만 사용한다. 실험적 opt-in은 `../development/`에 둔다.
- development queue를 이곳으로 옮길 때는 energy/gradient parity, 필요한 FD,
  memory gate와 반복 timing을 통과한 뒤 별도 이름으로 복사한다.

현재 queue는 commit `37f8f5d` 이후 production profile에 맞춰 다음을 명시한다.

- GDF: `setup_mode="native"`, `gdf_bilinear_mode="optimized_fused"`
- RSDF: `central_setup="exact_gpu_source"`, compressed CDERI mapped-host storage
- derivative omega scale: GDF/RSDF 모두 `1.0`
- Z-vector: matrix-free, VRAM peak tracking 활성화

## 실행

Benchmark root에서 실행한다.

```bash
python run_benchmark_queue.py --queue user/queues/current_correctness.json
python run_benchmark_queue.py --queue user/queues/current_water20_sz.json
python run_benchmark_queue.py --queue user/queues/current_water64_sz.json
```

각 queue의 `output_root`는 `user/results/` 밖을 가리킬 수 없으며 runner가 이를
검사한다. 기존 `../results/`의 과거 결과는 경로 보존을 위해 이동하지 않는다.

## 주요 queue

- `current_correctness.json`: Water8 analytic/selected-FD 확인
- `current_water20_sz.json`, `current_water20_dz.json`: Water20 production 비교
- `current_water64_sz.json`: Water64/SZ production 네 경로 비교
- `current_water64_dz.json`: Water64/DZ에서 완료된 GDF 경로만 기본 활성화한다.
  RSDF 두 경로는 현재 OOM 이력 때문에 `enabled: false`다.
- `current_water64_sz_uniform_vs_multigrid.json`: 동일 mesh의 적분 경로 비교
- `current_water_ethanol_sz_uniform_vs_multigrid.json`: 혼합 원소계 비교
- `current_water_ethanol_sz_d2c.json`: 같은 혼합 원소계의 r2SCAN+D2C-D4
  GDF/RSDF uniform/multigrid-XC 비교
