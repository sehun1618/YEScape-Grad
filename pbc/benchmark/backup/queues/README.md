# Archived Benchmark Queue Index

이 directory에는 2026-07-20 이전에 사용한 입력 JSON을 보존한다. 계산 결과는
`../results/`, 구조는 `../../structures/`, compact validation summary는
`../evidence/`에 분리한다.

Benchmark root에서 다음 형식으로 실행한다.

```bash
python run_benchmark_queue.py --queue backup/queues/<queue-name>.json
```

## 시작점

- `example_benchmark_queue.json`: 작은 H2 CPU/GPU 및 DF 조합 예제

## Correctness 및 release gate

- `gdf_optimization_release_gate.json`: GDF correctness/release matrix
- `gdf_post_memory_repair_timing.json`: GDF memory repair 이후 Water8/20 timing
- `multigrid_correctness_gate.json`: target multigrid analytic/FD ownership gate
- `multigrid_performance_gate.json`: Water8 반복 및 제한된 Water20 multigrid 성능
- `rsdf_r0_baseline.json`: RSDF optimization 이전 baseline
- `rsdf_r1_release_gate.json`: RSDF 반복 timing과 selected FD release gate

## RSDF central diagnostics

- `rsdf_exact_central_probe.json`: exact GPU-source central factor의 Water8/20 probe
- `rsdf_exact_central_water64_sz.json`: Water64/SZ exact central scaling probe

## Large-system 및 scaling

- `water64_benchmark_queue.json`: Water64/DZ 단일 GDF benchmark
- `water64_gdf_rsdf_multigrid_dz_recheck.json`: Water64/DZ GDF/RSDF와 multigrid 재검증
- `water64_gdf_rsdf_multigrid_queue.json`: Water64/SZ 네 route 비교
- `water64_rsdf_benchmark_queue.json`: Water64/SZ RSDF 단일 route
- `water_ethanol_gdf_rsdf_multigrid_queue.json`: water-ethanol/SZ route matrix와 null-mesh 진단
- `water_ethanol_gdf_rsdf_multigrid_mesh240_queue.json`: explicit $240^3$ target-mesh 재검증
- `water_ethanol_rsdf_r6_benchmark_queue.json`: RSDF R6/R11 계열 optimization 입력

## 경로 규칙

Queue runner는 상대경로를 queue JSON의 위치에서 해석한다. 따라서 이 directory의 queue는
다음 convention을 사용한다.

```json
{
  "output_root": "../results/<result-name>",
  "defaults": {
    "atom_file": "../../structures/<structure>.xyz"
  }
}
```

새 계산에는 `../../queues/`의 current queue를 사용한다. 과거 결과 directory의
`run_input.json`과 `queue_manifest.json`은 실행 당시 provenance이므로 이동하거나
재작성하지 않는다.
