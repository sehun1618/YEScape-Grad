# Legacy pre-DC-FORGE snapshot manifest

Date: 2026-07-26 KST

Status: archive snapshot, not a release

Archive branch: `archive/pre-dc-forge-2026-07-26`

Intended tag: `legacy-pre-dc-forge-2026-07-26`

This manifest freezes the source and observed behavior used as provenance for
the DC-FORGE migration. A completed archive commit means only that the files
were preserved. It does not mean that every route is correct, supported, or
release-ready.

## 1. Source provenance

- Dirty-state parent: `263ef0b2d4353806c3ecbd232e274dac040bec88`
- Parent branch before archive: `agent/gdf-rsdf-continuation`
- Molecular/PBC archive commit:
  `e21c00d85dbed0a00d2341e95436dbd5102e40a3`
- Stress prototype archive commit:
  `e3b765cd4c9c0265ca4a488ca00681209947f5ee`
- Tree after the two source commits:
  `451c56dce730b4e018791c184bd2d823bd8d797d`
- Original tracked binary diff SHA256:
  `4dd3bfb2abc2cc5efac7d3e10a56d6c79896b394e8040aa8beae2b2dac806665`
- Original untracked stream SHA256:
  `d99f635bbb4db121365b0b9ae001d72c7f7f3f149d6a145973fcd30645dc3481`
- Original tracked-plus-untracked stream SHA256:
  `8f7820db464771baee75781484bba6b1ce8766e3878e1d8c0bd327adbcc90662`

The frozen dirty state contained 33 modified tracked files (`+1809/-956`) and
220 untracked files totaling 1,488,428 bytes. The untracked files were split
into 18 molecular/PBC files and 202 stress prototype files. A token/secret
pattern scan found no candidate credentials.

`origin/main` was not used as the migration source. At the time of the audit,
`origin/main...HEAD` had 2 commits unique to origin and 9 commits unique to the
working branch.

## 2. Environment manifest

- Host: `x00`
- OS: Linux `5.14.0-611.5.1.el9_7.x86_64`, glibc 2.34
- Python: 3.11.6, conda-forge, GCC 12.3.0
- NumPy: 2.4.6
- SciPy: 1.17.1
- h5py: 3.16.0
- PySCF: 2.14.0
- GPU4PySCF: 1.8.0
- CuPy: 14.0.1
- CUDA runtime reported by CuPy: 13.0 (`13000`)
- CUDA toolkit: 13.2
- NVIDIA driver: 595.71.05
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- GPU memory: 97,887 MiB
- Compute capability: 12.0
- `cutensor-cu13`: 2.6.0
- `nvidia-cuest-cu13`: 0.1.1

All six project packages were editable imports from this repository:

```text
grad/cpu/src/pyscf_dcdft_grad
grad/gpu/src/gpu4pyscf_dcdft_grad
pbc/cpu/src/pyscf_pbc_dcdft
pbc/gpu/src/gpu4pyscf_pbc_dcdft
stress/cpu/src/pyscf_pbc_dcdft_stress
stress/gpu/src/gpu4pyscf_pbc_dcdft_stress
```

GPU4PySCF emitted `using cupy as the tensor contraction engine` even though
`cutensor-cu13` was installed. This is a degraded environment for performance
claims. Timing below must not be presented as cuTENSOR-enabled performance.

## 3. Full unit baseline

The suites were run with Python's standard `unittest` discovery in the `gpu`
conda environment.

| Scope | Result | Time |
|---|---:|---:|
| molecular CPU | 1/1 pass | 0.001 s |
| molecular GPU | 4/4 pass | 2.996 s |
| PBC benchmark | 33/33 pass | 0.142 s |
| PBC CPU | 56/56 pass | 298.432 s |
| PBC GPU | 125 pass, 18 fail, 2 error / 145 | 960.547 s |
| stress benchmark | 10/10 pass | 0.011 s |
| stress CPU | 76/76 pass | 401.324 s |
| stress GPU | 21/21 pass | 14.659 s |

Total: 346 tests, 326 pass, 18 fail, 2 error.

Important limitation: `grad/cpu/tests/test_h1_streaming.py` uses pytest-style
top-level functions. They were not collected by `unittest`, and pytest was not
installed in this environment. Therefore the molecular CPU count is not full
coverage of every file under that test directory.

The PBC GPU failures concentrated on sampled-k routes. Most stopped during
GPU4PySCF SCF initialization at `hermi_triu` because an overlap array was not
C-contiguous. Two k-point derivative tests instead failed when a non-contiguous
last axis was viewed as `complex128`. PBC and stress suites also repeatedly
emitted `ResourceWarning` messages for unclosed temporary files.

## 4. Fresh canonical calculations

These cases were rerun after the two source archive commits.

### 4.1 Molecular

- CPU H2O/STO-3G DC-PBE gradient example completed in 1.02 s with peak RSS
  289,472 KiB. The total gradient in Hartree/Bohr was:

```text
[[ 7.94461975e-17,  5.02590527e-16,  1.16919255e-01],
 [-4.65212226e-16, -5.18055357e-02, -5.84596276e-02],
 [ 3.87628385e-16,  5.18055357e-02, -5.84596276e-02]]
```

- GPU H2O/def2-SVP optimized RHF gradient matched the AO-dense reference at
  absolute tolerance `2e-10`. The test took 1.663 s; process wall time was
  2.60 s and peak RSS was 1,440,908 KiB.

### 4.2 PBC gradient

- CPU Gamma H2 GDF and RSDF analytic-gradient/energy-FD gates both passed.
  Two tests took 4.037 s; process wall time was 4.54 s and peak RSS was
  786,300 KiB.
- GPU Gamma H2 GDF energy parity passed.
- GPU Gamma H2 GDF gradient parity passed in 22.000 s; process wall time was
  23.13 s and peak RSS was 2,022,892 KiB.
- GPU Gamma H2 RSDF energy/gradient parity passed in 34.637 s; process wall
  time was 35.71 s and peak RSS was 2,131,936 KiB.
- Representative sampled-k GDF energy parity failed in 0.619 s at
  `gpu4pyscf.lib.cupy_helper.hermi_triu` with
  `AssertionError: mat.flags.c_contiguous`.

The Gamma results do not validate sampled-k support.

### 4.3 PBC stress

Fresh production-style Gamma H2 GPU stress calculations used selected `xx`
and `xy` central finite strain with a `2.5e-4` step.

| Backend | Execution | Accuracy | Max abs analytic-FD (Eh/Bohr^3) | Analytic stress | FD | Host peak RSS | GPU-used delta |
|---|---|---|---:|---:|---:|---:|---:|
| native GDF, GPU FT source adjoint | complete | pass | 2.8064111e-11 | 0.4213 s | 8.8319 s | 1,919,307,776 B | 1,264,582,656 B |
| exact-source RSDF, GPU FT source adjoint | complete | pass | 2.8257520e-11 | 0.3092 s | 13.6715 s | 1,913,913,344 B | 1,354,760,192 B |

The older `host_analytic_device_factorization` exact queues failed for both
GDF and RSDF before finite strain. They passed an already Voigt-like tensor of
shape `(6,4,2)` to `matrix_tensor_to_voigt`, which requires a leading `(3,3)`.
This is an active legacy route/queue contract defect, not an accuracy pass.

Fresh stress reports are under the ignored raw-result root:

```text
stress/benchmark/development/results/phase0_snapshot_2026-07-26/
```

## 5. Known blockers and interpretation

1. Sampled-k PBC GPU energy/gradient is not operational in the validated
   PySCF 2.14.0, GPU4PySCF 1.8.0, CuPy 14.0.1 environment.
2. The old exact stress response queue is incompatible with the current
   derivative tensor contract for both GDF and RSDF.
3. Temporary-file lifecycle warnings remain in PBC/stress tests.
4. The installed cuTENSOR package is not selected by GPU4PySCF.
5. Molecular CPU pytest-style tests were not runnable in the pinned
   environment because pytest was absent.
6. Water8/20/64 calculations were not rerun for this snapshot. Existing large
   reports are historical raw evidence only.
7. Some legacy compact JSON records contain absolute `/home/ksh/...` paths.
   They may be retained for history but cannot serve as portable DC-FORGE
   validation evidence.

Consequently this snapshot supports migration provenance and small Gamma
reference work. It does not support a release claim for general sampled-k PBC,
all stress routes, or production-size scaling.

## 6. Raw-result archive boundary

The external raw archive includes these ignored scientific-result roots:

```text
grad/benchmark/development/results
pbc/benchmark/results
pbc/benchmark/user/results
pbc/benchmark/development/results
pbc/benchmark/backup/results
pbc/benchmark/diagnostics
stress/benchmark/development/results
stress/benchmark/user/results
```

Build directories, `__pycache__`, pytest caches, egg-info, compiled extensions,
and other regenerable caches are excluded. Compact validation JSON files that
were source-adjacent and reviewed were preserved in the source commits.

## 7. License/provenance caution

PySCF/GPU4PySCF-derived Python and C sources contain Apache attribution or
project-local third-party notices in several locations. This snapshot preserves
those notices, but no final DC-FORGE license conclusion is made here. Public
release remains blocked until every migrated source file is classified as
original, adapted, copied, or generated and its upstream revision and license
obligations are recorded.
