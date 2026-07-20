# Current Benchmark Queues

이 directory에는 현재 source tree에서 새로 실행할 queue만 둔다. 과거 queue와 그 결과는
`../backup/` 아래에 함께 보존한다.

## Queue 목록

- `current_correctness.json`: Water8/SZ, GDF와 RSDF의 uniform 및 multigrid-XC route를
  atom 0의 z 성분 central FD와 비교한다.
- `current_water64_sz.json`: Water64/SZ 네 route의 timing 및 VRAM 비교. FD는 끈다.
- `current_water64_sz_uniform_vs_multigrid.json`: Water64/SZ에서 source와 uniform XC
  mesh를 명시적 `240^3`으로 고정하고, GDF/RSDF 각각의 uniform과 multigrid-XC
  gradient를 같은 최고 mesh에서 비교한다. FD는 끈다.
- `current_water_ethanol_sz_uniform_vs_multigrid.json`: Water–ethanol/SZ에서 source,
  uniform XC 및 multigrid-XC 최고 mesh를 모두 명시적 `240^3`으로 고정하고 GDF/RSDF
  네 경로의 gradient, timing 및 memory를 비교한다. FD는 끈다.
- `current_gdf_rsdf_optimization_baseline.json`: GDF/RSDF 최적화 전 Water8과 Water20
  production 경로를 각각 독립 process 3회로 반복한다. r2SCAN, multigrid-XC 및 현재
  exact response route를 고정하고 median timing, spread, checksum과 memory를 남긴다.
- `current_gdf_reverse_adjoint_gate.json`: 기존 signed-state GDF와 Gamma
  reverse-adjoint를 Water8/Water20에서 각각 3회 비교한다.
- `current_water64_prepost_optimized_omega1.json`: Water64/SZ, mesh `240^3`,
  omega scale 1.0에서 현재 GDF/RSDF production 경로를 실행한다. 결과는
  `a112d2c` pre-optimization worktree의 paired baseline과 비교한다.
- `current_water20_sz.json`, `current_water20_dz.json`: Water20의 현재 SZ/DZ
  production 경로를 비교한다.
- `current_water20_sz_pbe.json`: Water20/SZ의 PBE 비교 입력이다.
- `current_water64_dz.json`: Water64/DZ memory-safe 재검증. 확인된 GDF+multigrid route만
  기본 활성화하며, OOM 이력이 있는 나머지 route는 `enabled: false`로 남긴다.

Benchmark root에서 다음처럼 실행한다.

```bash
python run_benchmark_queue.py --queue queues/current_correctness.json
python run_benchmark_queue.py --queue queues/current_water64_sz.json
python run_benchmark_queue.py --queue queues/current_water64_sz_uniform_vs_multigrid.json
python run_benchmark_queue.py --queue queues/current_water_ethanol_sz_uniform_vs_multigrid.json
python run_benchmark_queue.py --queue queues/current_gdf_rsdf_optimization_baseline.json
python run_benchmark_queue.py --queue queues/current_gdf_reverse_adjoint_gate.json
python run_benchmark_queue.py --queue queues/current_water64_prepost_optimized_omega1.json
python run_benchmark_queue.py --queue queues/current_water20_sz.json
python run_benchmark_queue.py --queue queues/current_water20_dz.json
python run_benchmark_queue.py --queue queues/current_water64_dz.json
```

`current_water64_dz.json`의 비활성 route를 실행하려면 별도 파일로 복사한 뒤 원하는
job만 `enabled: true`로 바꾼다. 과거 queue를 수정해서 새 실험에 재사용하지 않는다.
