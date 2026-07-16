# Changelog

All notable changes to the public packages will be documented here.

## Unreleased

- Curate molecular CPU and GPU DC-DFT gradient implementations.
- Import independent periodic CPU and GPU packages.
- Add reproducible examples, validation scripts, and benchmark inputs.
- Align the repository and all four wheel distributions with the Apache-2.0
  licenses used by PySCF and GPU4PySCF, including upstream notices and
  file-level attribution for adapted implementations.
- Add bounded-memory Gamma GDF reciprocal planning, transactional OOM
  rebatching, workspace reuse, reportable phase metadata, and a profiled
  Gamma derivative SR/LR workload balance.
- Add profiled RSDF source/metric response ownership, all-atom primitive
  reuse, native one-electron H1 streaming, and a bounded source-contracted
  target-J/Z-vector response that retains no full dCDERI, dJ, or dK tensors.
- Separate RSDF full-tensor validation, production scalar contraction, and
  bounded GPU memory/OOM scheduling into independently testable modules.
