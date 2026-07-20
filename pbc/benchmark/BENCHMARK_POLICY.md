# Benchmark Ownership Policy

Periodic DC-DFT benchmark는 실행 주체와 목적에 따라 두 영역으로 분리한다.

| Scope | Owner | Queue | Results | 목적 |
| --- | --- | --- | --- | --- |
| user | user | `user/queues/` | `user/results/` | 안정된 production 계산과 직접 비교 |
| development | Codex | `development/queues/` | `development/results/` | 구현, profiling, 진단, release gate |

## Invariants

1. 모든 관리 queue는 `benchmark_scope`를 선언한다.
2. runner는 queue scope와 결과 root가 다르면 계산 전에 실패한다.
3. 기존 `benchmark/results/`와 `backup/results/`는 과거 증거이며 자동 이동하거나
   덮어쓰지 않는다.
4. development 결과는 자동으로 user queue를 변경하지 않는다.
5. user queue 승격은 production 기본값, 동일 조건 parity, 필요한 FD 및 memory gate를
   확인한 뒤 수행한다.
6. 향후 Codex의 새 최적화 queue와 결과는 기본적으로 development scope에 생성한다.
