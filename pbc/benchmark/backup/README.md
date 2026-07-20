# Benchmark Archive

2026-07-20 이전에 생성한 benchmark queue, compact evidence 및 raw result를 보존한다.
이 directory의 파일은 실행 당시 기록을 재현하기 위한 archive이며 새 실험의 시작점이
아니다.

```text
backup/
  queues/    과거 queue JSON과 당시 index
  evidence/  compact validation evidence
  results/   report, progress, stdout/stderr 및 attempt history
```

Archived queue의 `output_root`는 `backup/results/`, `atom_file`은 공용
`benchmark/structures/`를 가리키도록 이동 경로만 조정했다. 수치 설정과 job 내용은
바꾸지 않았다.

Raw `backup/results/`는 크고 장비 의존적인 산출물이므로 `.gitignore` 대상이다. Local
server에는 그대로 남지만 Git에는 queue, evidence 및 이 index만 기록한다.
