# Development Benchmarks

`benchmark_rhf_backends.py` compares the public optimized restricted GPU path
against the retained AO/dense reference path. It records gradient-only wall
time, internal stage times, process VRAM samples, and the maximum component
difference.

```bash
PYTHONPATH=../../gpu/src \
python benchmark_rhf_backends.py structure.xyz \
  --basis def2-svp --xc PBE --repeats 2 --output result.json
```

The reference path is a regression implementation, not a recommended
production setting.

`benchmark_uhf_backends.py` provides the corresponding alpha/beta comparison.
An explicit nonzero spin is required.

```bash
PYTHONPATH=../../gpu/src \
python benchmark_uhf_backends.py radical.xyz \
  --spin 1 --basis def2-svp --xc PBE --repeats 2 --output uhf-result.json
```
