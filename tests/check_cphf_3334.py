'''Compare native and #3334 gradients at exactly the same source HF state.

Run in conda gpu from the repository root, e.g.:
    python -m tests.check_cphf_3334 --backend cpu --output cpu.json
    python -m tests.check_cphf_3334 --backend gpu --output gpu.json

No production files or global upstream solvers are modified. This small
diagnostic uses a single CPU thread for reproducible difficult open-shell
tails, not as a CPU performance recommendation. GPU runs require cuTENSOR.
'''

import argparse
from datetime import datetime
import importlib
import json
from pathlib import Path
import time
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pyscf
from pyscf import lib
from pyscf.data import nist

from tests.common import as_numpy, finite_difference, make_mol, run_workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('cpu', 'gpu'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'Refusing to overwrite {args.output}')
    if args.backend == 'gpu':
        from gpu4pyscf.lib import cutensor
        if cutensor.contract_engine is not None:
            raise RuntimeError('Activate conda gpu with cuTENSOR before this check')
    energy = importlib.import_module(f'dcforge.{args.backend}.energy')
    cases = [
        ('water1.xyz', energy.RHFDFT, 'rhf', 1, 2, 1e-4, 3),
        ('oh.xyz', energy.UHFDFT, 'uhf', 1, 2, 2e-3, 4),
        ('ch2_triplet_distorted.xyz', energy.UHFDFT, 'uhf', 1, 2, 1e-4, 3),
    ]
    report = dict(backend=args.backend, basis='sto-3g', xc='PBE',
                  scf_conv_tol=1e-11, scf_conv_tol_grad=1e-9,
                  cphf_tol=1e-12, threads=1, pr='pyscf/pyscf#3334',
                  time_kst=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(), cases=[])
    report['pyscf_version'] = pyscf.__version__
    if args.backend == 'gpu':
        import gpu4pyscf
        import cupy as cp
        report['gpu4pyscf_version'] = gpu4pyscf.__version__
        report['cupy_version'] = cp.__version__
        report['cutensor'] = True
        report['gpu'] = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    report['timing_note'] = 'Diagnostic only; not an isolated performance comparison.'
    for name, workflow, spin, atom, axis, step, level in cases:
        for df in (False, True):
            module = importlib.import_module(f'dcforge.{args.backend}.grad.{spin}_zvec')
            mol = make_mol(name, basis='sto-3g')
            with lib.with_omp_threads(1):
                mf, target = run_workflow(workflow, mol, grid_level=level, density_fit=df)
                if not mf.mf_hf.converged:
                    raise RuntimeError(f'Source SCF unconverged: {name}')
                row = dict(structure=name, density_fit=df, target_energy_ha=target,
                           fd_atom=atom, fd_axis=axis, fd_step_bohr=step, grids_level=level)
                gradients = {}
                for label in ('native', 'backport'):
                    driver = mf.nuc_grad_method()
                    driver.cphf_conv_tol = 1e-12
                    if args.backend == 'gpu':
                        import cupy as cp
                        cp.cuda.get_current_stream().synchronize()
                    start = time.perf_counter()
                    if label == 'native':
                        with patch.object(module, 'solve_cphf',
                                          lambda backend, *a, **k: backend.solve(*a, **k)):
                            grad = driver.kernel()
                    else:
                        grad = driver.kernel()
                    gradients[label] = as_numpy(grad)
                    row[label] = dict(z_residual_max_abs=driver.z_residual_max_abs,
                                      gradient_ha_bohr=gradients[label].tolist(),
                                      wall_seconds=time.perf_counter() - start)
                fd = finite_difference(workflow, mol, atom, axis, step=step,
                                       grid_level=level, density_fit=df)
            row['fd_gradient_ha_bohr'] = fd
            conversion = nist.HARTREE2EV / nist.BOHR
            for label in gradients:
                error = abs(gradients[label][atom, axis] - fd)
                row[label]['fd_error_ha_bohr'] = error
                row[label]['fd_error_ev_angstrom'] = error * conversion
            delta = float(np.max(np.abs(gradients['backport'] - gradients['native'])))
            row['gradient_change_max_ha_bohr'] = delta
            row['gradient_change_max_ev_angstrom'] = delta * conversion
            report['cases'].append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps(row), flush=True)
            assert row['backport']['fd_error_ha_bohr'] < 3e-6
            assert np.isfinite(row['backport']['z_residual_max_abs'])
            assert row['backport']['z_residual_max_abs'] <= (
                row['native']['z_residual_max_abs'] + 1e-12)
            if name == 'ch2_triplet_distorted.xyz':
                assert row['backport']['z_residual_max_abs'] < 2e-6


if __name__ == '__main__':
    main()
