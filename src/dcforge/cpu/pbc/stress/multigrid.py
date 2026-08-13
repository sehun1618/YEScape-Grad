'''XC-only strain references for CPU periodic multigrid targets.'''

import copy

import numpy as np
from pyscf.pbc.dft import gen_grid, multigrid, numint
from pyscf.pbc.grad import rks_stress

from dcforge.cpu.pbc.stress.finite_difference import (
    frozen_density_finite_difference,
)
from dcforge.cpu.pbc.stress.strain import snapshot_strain_state


def get_vxc_strain_finite_difference(mf_grad, dm=None, step=1.e-4,
                                     return_metadata=False,
                                     return_report=False):
    r'''Evaluate matched CPU pair-multigrid XC strain by Cell finite difference.

    The AO density matrix is held fixed while the Cell, atomic centers and AO
    basis are affinely strained.  This isolates the explicit target XC term;
    source-orbital response belongs to the later Z-vector ledger.  Because the
    density strain is numerical, this helper is an oracle and never a
    production-analytic implementation.
    '''
    mf = mf_grad.base
    if not isinstance(mf._numint, multigrid.MultiGridNumInt2):
        raise TypeError('CPU XC strain requires a pair-multigrid target')
    if mf.multigrid_with_j or bool(mf._numint.xc_with_j):
        raise NotImplementedError('CPU XC strain requires with_j=False')
    if str(mf.xc).replace(' ', '').upper() != 'PBE':
        raise NotImplementedError(
            'CPU pair-multigrid XC strain currently supports PBE only')
    if dm is None:
        if not mf.mf_hf.converged:
            raise RuntimeError('run source RHF before requesting XC strain')
        dm = mf.make_hf_rdm1()
    dm = np.asarray(dm)
    if dm.shape != (mf.cell.nao_nr(),) * 2:
        raise ValueError('CPU XC strain density must have shape (nao, nao)')
    step = float(step)

    def xc_energy(strained_cell):
        from dcforge.cpu.pbc.energy.rhf_dft import RHFDFT

        trial = RHFDFT(strained_cell, xc=mf.xc).multigrid_numint(
            ntasks=mf._numint.ntasks,
            ke_ratio=mf._numint.ke_ratio,
            rel_cutoff=mf._numint.rel_cutoff,
            with_j=False)
        _nelec, exc, _vxc = trial._numint.nr_rks(
            strained_cell, trial.grids, trial.xc, dm,
            0, 1, None, None, max_memory=trial.max_memory,
            verbose=trial.verbose)
        return {
            'energy': float(exc),
            'state': snapshot_strain_state(
                strained_cell, numint=trial._numint),
        }

    report = frozen_density_finite_difference(
        mf.cell, xc_energy, steps=(step,), directions='voigt')
    work_xc = np.asarray(report['assembled'][0]['work_raw_hartree'])
    metadata = {
        'target_xc_stress_owner': 'pyscf_pair_multigrid_energy_fd',
        'with_j': False,
        'with_nuc': False,
        'mesh': np.asarray(mf._numint.mesh, dtype=int).tolist(),
        'step': step,
        'production_analytic': False,
        'matched_target_integrator': True,
        'low_level_numerical_components': [
            'pair_multigrid_xc_cell_finite_difference'],
    }
    if return_report:
        metadata['finite_difference_report'] = report
    if return_metadata:
        return work_xc, metadata
    return work_xc


def get_vxc_strain_uniform_grid_oracle(mf_grad, dm=None,
                                       return_metadata=False):
    r'''Evaluate the same functional on a uniform-grid XC strain oracle.

    PySCF 2.14.0 exposes an analytic uniform-grid PBC XC strain derivative but
    no pair-multigrid stress primitive.  This routine therefore replaces only
    the numerical integrator on shallow local copies and calls
    :func:`pyscf.pbc.grad.rks_stress.get_vxc` with both Hartree and nuclear
    pseudopotential ownership disabled.  It is a mathematical oracle, not a
    matched pair-multigrid production stress term.

    Args:
        mf_grad : dcforge.cpu.pbc.grad.rhf_dft.Gradients
            Gradient object whose base owns the target functional and Cell.

    Kwargs:
        dm : ndarray or None
            Fixed spin-summed RHF AO density.  The converged source density is
            used when omitted.
        return_metadata : bool
            Return ``(work_xc, metadata)`` when true.

    Returns:
        ndarray of shape (3, 3) or tuple
            Energy-like XC strain derivative in Hartree and optional owner
            metadata.
    '''
    mf = mf_grad.base
    if not isinstance(mf._numint, multigrid.MultiGridNumInt2):
        raise TypeError('CPU XC strain oracle requires a multigrid target')
    if mf.multigrid_with_j or bool(mf._numint.xc_with_j):
        raise NotImplementedError(
            'CPU XC strain oracle requires with_j=False')
    if dm is None:
        if not mf.mf_hf.converged:
            raise RuntimeError('run source RHF before requesting XC strain')
        dm = mf.make_hf_rdm1()
    dm = np.asarray(dm)

    oracle_mf = mf.copy()
    oracle_mf._numint = numint.NumInt()
    grids = gen_grid.UniformGrids(mf.cell)
    grids.mesh = np.asarray(mf.cell.mesh, dtype=int).copy()
    oracle_mf.grids = grids
    oracle_grad = copy.copy(mf_grad)
    oracle_grad.base = oracle_mf
    oracle_grad.grids = grids
    work_xc = np.asarray(rks_stress.get_vxc(
        oracle_grad, mf.cell, dm, with_j=False, with_nuc=False))
    if work_xc.shape != (3, 3) or not np.all(np.isfinite(work_xc)):
        raise RuntimeError('uniform-grid XC strain returned an invalid tensor')

    metadata = {
        'target_xc_stress_owner': 'pyscf_uniform_grid_oracle',
        'with_j': False,
        'with_nuc': False,
        'mesh': np.asarray(grids.mesh, dtype=int).tolist(),
        'production_analytic': False,
        'matched_target_integrator': False,
        'low_level_numerical_components': [],
    }
    if return_metadata:
        return work_xc, metadata
    return work_xc
