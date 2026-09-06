# PBC migration record

## Source

- Imported from: `/home/ksh/2026_project/dc_forge_repo/src`
- Imported on: 2026-09-04
- The source directory was not a Git worktree at migration time.
- Aggregate SHA256 over the 81 source files that have matching relative paths
  in this migration: `65905cfefb6045e7207769d1c3ccbf0ec848e5119b60748fadf634b41c32586b`

The aggregate is formed by sorting relative paths, then hashing each path,
a null byte, and the SHA256 digest of that source file. It identifies the
read-only source snapshot; the migrated files intentionally differ where the
capability boundary was narrowed.

## Included capability

- CPU and GPU
- three-dimensional GTH pseudopotential cells
- closed-shell RHF source
- one real Gamma point
- FFTDF and GDF
- PBE and r2SCAN fixed-density target energy
- analytic fixed-cell nuclear gradient
- optional real-Gamma MultiGrid target XC, or target J+XC
- CPU explicit full-BZ sampled-k FFTDF/GDF pure LDA/GGA/tau-MGGA@KRHF
  energy and analytic fixed-cell gradient, with equal k-point weights and
  insulating 2/0 occupations
- GPU explicit full-BZ sampled-k FFTDF/GDF pure LDA/GGA or r2SCAN@KRHF
  energy and analytic fixed-cell gradient, with equal k-point weights and
  insulating 2/0 occupations
- GPU sampled-k FFTDF/GDF target J plus XC-only MultiGrid for PBE/r2SCAN

CPU PySCF 2.14.0 MultiGrid does not provide the meta-GGA tau derivative used
by r2SCAN, so CPU r2SCAN remains on its ordinary uniform-grid integrator.
GPU4PySCF 1.8.1 supports the r2SCAN MultiGrid route used here.

## Excluded capability

- RSDF and RSJK
- ADMM
- GPU sampled-k MultiGrid J+XC and complex Gamma orbitals
- symmetry-reduced/IBZ or nonuniform-weight k-point sampling
- CPU sampled-k MultiGrid; PySCF 2.14 pair-MultiGrid rejects non-Gamma
  density collocation and AO derivative pass-2, so this route fails before
  source SCF instead of silently replacing it with uniform-grid XC
- UHF and ROHF PBC references
- stress and cell derivatives
- hybrid or range-separated target functionals
- density-Laplacian-dependent MGGA target functionals
- NLC and VV10
- one- and two-dimensional or all-electron PBC

Unsupported routes fail at provider or workflow validation rather than
silently changing the Hamiltonian.

The migrated GDF acceleration ownership, default/opt-in boundary, omitted
checkpoint path and inherited performance evidence are audited separately in
`pbc_gdf_acceleration.md`.

## Finite-mesh translation diagnostic

The sampled-k gradient records ``translation_residual = gradient.sum(0)``.
This is a diagnostic, not a constraint imposed on the result.  At finite
uniform mesh the translated atoms move relative to the fixed real-space grid,
so the discretized pseudopotential, Hartree, and XC energies can exhibit an
eggbox error.  A rigid-translation finite difference must reproduce this
nonzero analytic sum; increasing ``cell.mesh`` should then converge it toward
zero.  DC-Forge does not subtract the sum because doing so would hide the
finite Hamiltonian actually differentiated.

The reproducible K8 validation driver is
``tools/validation/pbc_kpts_mgga.py``.  It separates all-coordinate relaxed-
source finite differences, full-BZ ordering/time-reversal checks, and the
rigid-translation mesh sweep.  For example:

```bash
python tools/validation/pbc_kpts_mgga.py \
  --phase gradient symmetry eggbox \
  --xc TPSS SCAN M06-L \
  --provider fft gdf \
  --threads 64 \
  --output /tmp/pbc_kpts_mgga.json
```

``--threads`` calls ``pyscf.lib.num_threads`` and records the effective value
in the JSON artifact.  Omitting it preserves the thread count selected by the
active environment.  A 64-core machine should still compare representative
16-, 32-, and 64-thread production calculations: more OpenMP threads need not
reduce wall time for every FFT, integral, or linear-algebra stage.

On the asymmetric Water1/GTH-SZV/GTH-PBE, 7 Bohr cell, ``2x1x1`` k mesh,
and ``21^3`` uniform mesh used for K8, the complete relaxed-source gradient
errors were:

| Provider | Target | RMS error | Max error |
|---|---|---:|---:|
| FFTDF | TPSS | 4.43e-7 | 9.59e-7 |
| FFTDF | SCAN | 4.69e-7 | 1.06e-6 |
| FFTDF | M06-L | 2.78e-7 | 6.90e-7 |
| GDF | TPSS | 8.09e-9 | 1.82e-8 |
| GDF | SCAN | 7.73e-8 | 2.10e-7 |
| GDF | M06-L | 1.73e-7 | 4.83e-7 |

Errors are in Ha/Bohr.  Physical Z residuals were at most ``1.87e-11``;
term recomposition was at most ``2.22e-16 Ha/Bohr``.  Reversing the k-point
order or applying time reversal changed the energy by at most ``1.07e-14 Ha``
and the gradient by at most ``1.05e-13 Ha/Bohr``.

The same test also demonstrates why analytic-versus-FD agreement must not be
confused with mesh convergence.  For TPSS, the maximum translation sum was:

| Mesh | FFTDF | GDF |
|---:|---:|---:|
| 15^3 | 5.33 | 2.83e-1 |
| 21^3 | 1.35 | 3.24e-2 |
| 31^3 | 8.20e-2 | 2.13e-2 |
| 41^3 | 1.06e-2 | 1.03e-2 |

These values are in Ha/Bohr and are intentionally not acceptance thresholds.
Even ``41^3`` is not converged for this deliberately compact oxygen-containing
cell.  Nevertheless, the analytic sum and rigid-translation FD agree to
``2.48e-8 Ha/Bohr`` at ``41^3``.  Production work must converge
``cell.ke_cutoff`` or ``cell.mesh`` against energy and individual forces; it
must not infer adequacy from the analytic-FD error alone.

PySCF issue #819 reports the same combination of analytic/finite-difference
agreement and a nonzero total periodic DFT force.  The finite uniform grid and
atom-centered Gaussian representation make the discretized XC energy slightly
translation dependent.  GDF also has independently resolved grids for ERIs
and pseudopotential/nuclear terms, so ``with_df.mesh`` alone is not a complete
force-convergence control; ``cell.precision`` and the resolved target-XC mesh
must be recorded as well.  DC-Forge treats this as a convergence diagnostic,
not as a missing response term.

## Release validation

The migrated PBC slice is ready to version only after all of the following
checks pass from the ``gpu`` Conda environment:

```bash
# Source-tree contract and CPU regression checks.
python -m pytest -q tests/unit tests/cpu -m "not fd"

# Relaxed-source molecular finite differences and the opt-in PBC FD check.
python -m pytest -q tests/cpu -m fd
DCFORGE_RUN_PBC_FD=1 python -m pytest -q \
  tests/cpu/test_pbc.py -k selected_finite_difference

# GPU molecular and real-Gamma PBC regression checks.
python -m pytest -q tests/gpu -m gpu

# Build an isolated wheel and inspect/import it outside the source tree.
python -m pip wheel . --no-deps --wheel-dir /tmp/dcforge-wheel
```

The wheel check must call
``dcforge.cpu.pbc.df.gaussian.ft.load_extension()`` and complete one small GDF
gradient.  The loader first exposes PySCF's ``libcgto`` and ``libpbc`` symbols,
then imports the internal ``_ft_ip_ovlp`` callback; importing that private
extension directly is not its supported loading contract.  An isolated
installation is important because the source checkout can otherwise hide a
missing extension behind a stale in-place ``.so`` file.  Stress/cell
derivatives and the other excluded capabilities above are not release blockers
for this slice.

When examples are executed before installing the new wheel, use
``PYTHONPATH=. python examples/pbc/...`` from the repository root. A direct
script invocation otherwise puts the script directory first and may resolve an
older installed ``dcforge`` package instead of this checkout.

### K9 integration result — 2026-09-04

The K9 audit used Python 3.11.6, PySCF 2.14.0, GPU4PySCF 1.8.1, CuPy 14.0.1,
one RTX PRO 6000 Blackwell Server Edition, and a 64-thread PySCF default.

- Unit plus CPU non-FD suite: 102 passed, one opt-in PBC FD skipped.
- CPU relaxed-source FD suite: 12 passed.
- CPU Gamma GDF selected FD: one passed.
- GPU molecular and real-Gamma PBC suite: 26 passed.
- All four CPU PBC examples completed from the current source checkout.
- An isolated CPython 3.11 wheel contained the FFT/GDF, sampled-k, GPU, and
  native callback modules. Its public CPU/GPU imports, native callback loader,
  and a small CPU GDF energy/gradient calculation completed successfully.
- The thread-option smoke test recorded two requested PySCF threads and kept
  reverse-order sampled-k energy/gradient differences below ``1e-10 Ha`` and
  ``1e-9 Ha/Bohr``, respectively.

One initial 64-thread run of the deliberately difficult molecular OH tail gave
a ``3.423e-6 Ha/Bohr`` FD difference, just outside its ``3e-6`` diagnostic
gate; immediate repeats gave ``1.713e-6`` at 64 threads and ``1.897e-7`` at one
thread, with essentially unchanged physical Z residual ``8.74e-7``. The
stress-structure acceptance test now uses PySCF's scoped one-thread context to
make its floating-point reduction order reproducible, then restores the
machine default. Production and thread-scaling calculations remain free to
use all requested cores.

## Native extension

CPU GDF nuclear gradients require one small callback that PySCF does not
provide: the i/j-center derivative recurrence used by PySCF's periodic
Fourier AO-pair driver. `setup.py` builds only `_ft_ip_ovlp` from source.
No stress-only extension or prebuilt binary is migrated.

### GPU sampled-k GDF result — 2026-09-04

The GPU sampled-k GDF gradient reuses the FFTDF complex Z-vector, Pulay, and
target-XC algebra.  Only the provider-dependent pieces differ: PySCF contracts
the GDF Hcore derivative on the host, while GPU4PySCF contracts target-J and
source-Fock derivatives to per-atom scalars on the device.  The latter is a
three-call polarization of its private quadratic endpoint
``_jk_energy_per_atom``; one shared ``SRInt3c2eOpt`` supplies the provider's
resolved omega and the full-BZ BvK mesh.  The endpoint does not expose separate
source and metric terms, so the ledger labels that result ``native_combined``.

For H2/GTH-SZV/GTH-PBE, a 6 Bohr cell, ``2x1x1`` k mesh, ``15^3`` uniform
mesh, and ``cell.precision=1e-7``:

| Target | CPU-GPU energy | CPU-GPU gradient max | physical Z residual |
|---|---:|---:|---:|
| PBE | 1.36e-10 Ha | 2.53e-9 Ha/Bohr | 1.11e-21 |
| r2SCAN | 1.36e-10 Ha | 2.56e-9 Ha/Bohr | 1.08e-19 |

The selected relaxed-source PBE finite difference differed by
``2.43e-9 Ha/Bohr``.  The complete GPU regression suite passed 36 tests.  The
finite-mesh translation sum was ``4.81e-4 Ha/Bohr`` and is retained as the
eggbox diagnostic described above, not subtracted from the force.

This sampled-k GDF derivative depends on private GPU4PySCF 1.8.1 interfaces.
The compatibility code is isolated in
``dcforge.gpu.pbc.df.gaussian.kpts_response`` and records the private endpoint,
BvK mesh, resolved omega, call count, host-result bytes, and any complex-layout
copy.  It must be re-audited when GPU4PySCF changes that ABI.

### GPU sampled-k XC-only MultiGrid result — 2026-09-04

``KRHFDFT.multigrid_numint(with_j=False)`` delegates only target semilocal XC
and its explicit derivative to GPU4PySCF multigrid-v2.  Source KRHF J/K, target
Hartree J, CPHF actions, and the provider nuclear response remain on the shared
FFTDF or GDF object.  Therefore the target Hartree energy is unchanged exactly;
the finite-mesh energy difference from UniformGrids belongs entirely to XC.

For H2/GTH-SZV/GTH-PBE, a 6 Bohr cell, ``2x1x1`` k mesh, 30 Ha cutoff, and
``cell.precision=1e-7``, selected relaxed-source finite differences gave:

| Target | Provider | MG-uniform energy | Hartree difference | FD error |
|---|---|---:|---:|---:|
| PBE | FFTDF | -5.44e-4 Ha | 0.00 Ha | 2.15e-9 Ha/Bohr |
| PBE | GDF | -5.44e-4 Ha | 0.00 Ha | 2.59e-9 Ha/Bohr |
| r2SCAN | FFTDF | 2.39e-4 Ha | 0.00 Ha | 2.07e-9 Ha/Bohr |
| r2SCAN | GDF | 2.39e-4 Ha | 0.00 Ha | 2.52e-9 Ha/Bohr |

The MG-versus-uniform energy column is not an accuracy error: the two
integrators use different finite quadratures at this deliberately small
cutoff.  Production calculations must converge ``cell.ke_cutoff``.  The FD
column instead verifies that the analytic derivative differentiates the same
finite MG Hamiltonian used by the energy.  Sampled-k MG J+XC remains blocked
until its independent target-J derivative and provider-mixing gates pass.
The complete GPU regression suite passed 43 tests and the unit suite passed
24 tests after this addition.

GPU4PySCF 1.8.1's sampled-k GDF CDERI builder failed for a deliberately
reordered full-BZ array before target XC was evaluated.  DC-Forge now rejects
that input before CDERI construction and requires canonical
``cell.make_kpts`` ordering for GPU GDF.  FFTDF passed the r2SCAN MultiGrid
k-point reordering gate; this provider limitation is not attributed to MG.

### Post-MultiGrid integration audit — 2026-09-04

The final audit after adding GPU sampled-k GDF derivatives and XC-only
MultiGrid used the ``gpu`` Conda environment at starting commit
``dbedad2f00819a32ff2aed2a2f088c84cd2c2104``.  The PBC migration was still an
uncommitted working-tree change during this audit.

- Unit and CPU non-FD regression: 102 passed, one opt-in PBC FD skipped.
- GPU molecular, Gamma, and sampled-k regression: 43 passed.
- ``git diff --check`` and ``compileall`` completed without errors.
- All three GPU PBC examples completed from the source checkout, including
  sampled-k GDF plus XC-only MultiGrid.
- An isolated CPython 3.11 wheel loaded the internal GDF callback and public
  CPU/GPU PBC APIs, then completed a finite H2/GTH-SZV GDF energy and analytic
  gradient calculation.  The wheel SHA256 was
  ``8374ec320be9562d6b13b090efb3e9c284d3223d53d326859cb4112e42552035``.

This closes the sampled-k FFTDF/GDF plus XC-only MultiGrid migration slice.
Sampled-k MultiGrid J+XC and scanners remain explicit non-goals for this
slice; neither is silently enabled by the completed audit.

### GDF acceleration structure audit — 2026-09-04

The post-migration audit compared the active GDF modules with the original
source snapshot, removed two unreachable sampled-k RSDF assembly functions,
and documented the default, reference, opt-in and omitted acceleration paths in
``pbc_gdf_acceleration.md``.  No numerical Hamiltonian or production default
was changed.

- Unit plus CPU non-FD regression: 102 passed, one opt-in PBC FD skipped.
- Complete GPU regression: 49 passed.
- The focused GDF policy and FP64-versus-default-Z3 checks: 8 passed.
- Ruff, ``compileall`` and ``git diff --check`` completed without errors.
- The new GDF acceleration example completed on the active GPU.  It resolved
  the FP64 source profile, mixed Z response, MO-direct adjoint and a physical
  Z residual of ``5.34e-10``.
- ``pip wheel`` built the CPython 3.11 wheel, and an isolated install loaded
  the public CPU/GPU PBC APIs and the package-owned Fourier derivative
  callback.

The original provider CDERI checkpoint path remains intentionally absent.  It
is the principal omitted GDF lifecycle optimization and needs a separate
multi-frame identity/cache audit before migration.
