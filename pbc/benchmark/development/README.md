# Development Benchmarks

이 directory는 Codex가 구현, profiling, 회귀 진단 및 release gate에 사용하는
benchmark를 보관한다.

## 운영 규칙

- queue는 `queues/`, 진단 코드는 `diagnostics/`, 새 결과는 `results/`에 둔다.
- omega sweep, batching probe, storage policy 비교, pre/post gate와 실패 재현은 모두
  이곳에서 수행한다.
- 사용자가 직접 사용하는 안정 queue를 구현 과정에서 덮어쓰지 않는다.
- 결과가 정확성 및 성능 gate를 통과해도 자동으로 user benchmark가 되지 않는다.
  사용자가 승격을 요청하면 production 기본값으로 고정한 새 queue를
  `../user/queues/`에 복사한다.
- 대형 Water64/DZ 계산은 사용자의 명시적 허용 없이는 실행하지 않는다.

## 실행

```bash
python run_benchmark_queue.py \
  --queue development/queues/current_gdf_reverse_adjoint_gate.json
python run_benchmark_queue.py \
  --queue development/queues/current_gdf_rsdf_optimization_baseline.json
```

`development` queue의 출력은 runner에 의해 `development/results/` 아래로 제한된다.
