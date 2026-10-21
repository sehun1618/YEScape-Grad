# DC-Forge
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

'''DC-Forge result logging using the existing PySCF logger.

These helpers only read completed results. They never evaluate energy, Fock,
response, or residuals. Total energy and forces print at every verbosity.
NOTE adds summaries and timings; INFO adds settings and detailed diagnostics.
No output file, verbosity switch, GPU timer or persistence format is introduced.
Public kernels report host elapsed time without GPU synchronization.
'''

from functools import wraps
from time import perf_counter

import numpy as np
from pyscf.lib import logger
from pyscf.grad.rhf import _write


def log_kernel_time(label):
    '''Log successful kernel elapsed time at NOTE level and above.

    This is host wall time, not CPU process time or a CUDA-event profile.
    Do not synchronize GPUs or change PySCF's process-wide TIMER_LEVEL.
    A local start time prevents reuse of stale timings across repeated calls.
    '''
    def decorate(kernel):
        @wraps(kernel)
        def timed(self, *args, **kwargs):
            start = perf_counter()
            if label == 'Energy workflow':
                self._last_energy_timing = None
            result = kernel(self, *args, **kwargs)
            end = perf_counter()
            elapsed = end-start
            log = logger.new_logger(self)
            logger.note(log, '%s wall time = %.3f s', label, elapsed)
            if label == 'Energy workflow':
                self._last_energy_timing = (start, elapsed)
            elif label == 'Gradient workflow':
                energy = getattr(self.base, '_last_energy_timing', None)
                # Measure the actual interval, including time between calls.
                # Nested energy must not exclude the start of the gradient.
                total_start = min(start, energy[0]) if energy is not None else start
                logger.note(log, 'Total wall time = %.3f s', end-total_start)
            return result
        return timed
    return decorate


def _header(obj, mf, title, unrestricted):
    logger.note(obj, '\n================ DC-Forge %s ================', title)
    source_xc = getattr(mf.mf_source, 'xc', 'HF')
    logger.note(obj, '%-28s %s-%s', 'Method', source_xc, mf.xc)
    logger.info(obj, '%-28s %s', 'Source XC', source_xc)
    logger.info(obj, '%-28s %s', 'Target XC', mf.xc)
    if hasattr(mf.mf_source, 'grids'):
        grids = mf.mf_source.grids
        _setting(obj, 'Source grid', grids.atom_grid or 'level = %s' % grids.level)
    logger.note(obj, '%-28s %s', 'Spin treatment',
                'Unrestricted' if unrestricted else 'Restricted')
    dispersion = getattr(mf, 'dispersion', None)
    logger.note(obj, '%-28s %s', 'Dispersion',
                str(dispersion).upper() if dispersion else 'None')
    gpu = any(cls.__module__.startswith('dcforge.gpu.') for cls in type(mf).__mro__)
    logger.info(obj, '%-28s %s', 'Backend', 'GPU' if gpu else 'CPU')
    provider = getattr(mf, 'with_df', None)
    logger.info(obj, '%-28s %s', 'Integral treatment',
                ('Density fitting (%s)' % type(provider).__name__
                 if provider is not None else 'Conventional'))
    if provider is not None:
        auxbasis = getattr(provider, 'auxbasis', None)
        if auxbasis is None:
            auxbasis = getattr(getattr(provider, 'auxmol', None), 'basis', None)
        _setting(obj, 'Auxbasis', auxbasis if auxbasis is not None
                 else 'Automatic (not resolved)')
    grids = getattr(mf, 'grids', None)
    if grids is not None:
        atom_grid = grids.atom_grid
        if isinstance(atom_grid, dict):
            if atom_grid:
                _setting(obj, 'Grid', atom_grid)
                elements = set(mf.mol.atom_symbol(i) for i in range(mf.mol.natm))
                if 'default' not in atom_grid and not elements <= atom_grid.keys():
                    logger.info(obj, '%-28s Other elements: level = %s',
                                '', grids.level)
            else:
                logger.info(obj, '%-28s level = %s', 'Grid', grids.level)
        else:
            logger.info(obj, '%-28s atom_grid = %s', 'Grid', atom_grid)


def _setting(log, label, value):
    '''Print element-specific settings without dumping custom basis arrays.'''
    is_auxbasis = label == 'Auxbasis'
    if isinstance(value, dict):
        for element, setting in value.items():
            if is_auxbasis and not isinstance(setting, str):
                setting = 'Custom basis'
            logger.info(log, '%-28s %s: %s', label, element, setting)
            label = ''
    else:
        if label == 'Auxbasis' and not isinstance(value, str):
            value = 'Custom basis'
        logger.info(log, '%-28s %s', label, value)


def _energy_value(log, mf, label, attribute):
    value = getattr(mf, attribute, None)
    if value is None:
        logger.note(log, '%-28s not available', label)
    else:
        logger.note(log, '%-28s %.15f Ha', label, value)


def energy_summary(mf, unrestricted=False):
    '''Print completed source, target and dispersion-inclusive total energies.

    Called only by the public workflow kernel, never by energy_tot().
    Source/target energies include their usual nuclear contribution;
    target energy excludes the separately reported dispersion correction.
    '''
    log = logger.new_logger(mf, max(mf.verbose, logger.NOTE))
    if mf.verbose < logger.NOTE:
        _energy_value(log, mf, 'DC-Forge Total energy', 'e_tot')
        return
    _header(log, mf, 'Energy', unrestricted)
    logger.note(log, '%-28s %s', 'Source SCF converged',
                'Yes' if mf.mf_source.converged else 'No')
    _energy_value(log, mf, 'Source energy', 'e_source')
    _energy_value(log, mf, 'Target energy', 'e_dft')
    if getattr(mf, 'dispersion', None):
        _energy_value(log, mf, 'D2C dispersion energy', 'e_d2c')
    else:
        logger.note(log, '%-28s not applied', 'Dispersion energy')
    _energy_value(log, mf, 'Total energy', 'e_tot')
    logger.note(log, '=================================================')


def gradient_summary(grad, unrestricted=False):
    '''Replace the upstream gradient finalizer without duplicating its table.

    Uses the existing final host gradient and physical residual. No device
    arrays are copied for reporting. Missing diagnostics are not a PASS.
    The residual verdict is diagnostic only and never changes solver policy.
    '''
    log = logger.new_logger(grad, max(grad.verbose, logger.NOTE))
    mf = grad.base
    de = grad.de
    atoms = getattr(grad, 'atmlst', None)
    if grad.verbose < logger.NOTE:
        logger.note(log, '\nDC-Forge Forces [Ha/Bohr]')
        if isinstance(de, np.ndarray):
            _write(log, grad.mol, -de, atoms)
        else:
            logger.note(log, 'Force table unavailable: result is not a host array')
        return
    _header(log, mf, 'Gradient', unrestricted)
    logger.note(log, '%-28s Forces (including selected dispersion)', 'Quantity')
    logger.note(log, '%-28s Ha/Bohr', 'Unit')
    logger.note(log, '%-28s F = -dE/dR', 'Force convention')
    if not isinstance(de, np.ndarray):
        # Do not introduce a device transfer/synchronization just for logging.
        logger.note(log, 'Force table unavailable: result is not a host array')
        return
    _write(log, grad.mol, -de, atoms)
    for label, attribute in (
            ('Grid response', 'grid_response'),
            ('Source grid response', 'source_grid_response'),
            ('Grid response backend', 'grid_response_backend_used'),
            ('Auxiliary-basis response', 'auxbasis_response'),
            ('Z-vector solver', 'z_solver')):
        value = getattr(grad, attribute, None)
        if value is not None:
            logger.info(log, '%-28s %s', label, value)
    logger.note(log, '%-28s %s', 'Source SCF converged',
                'Yes' if mf.mf_source.converged else 'No')
    tol = getattr(grad, 'cphf_conv_tol', None)
    if tol is not None:
        logger.info(log, '%-28s %.3e', 'CPHF tolerance', tol)
    info = getattr(grad, 'z_info', None) or {}
    residual = getattr(grad, 'z_residual_max_abs', None)
    if residual is None:
        residual = info.get('residual_max_abs')
    limit = getattr(grad, 'cphf_residual_warn_tol',
                    getattr(grad, 'cphf_residual_tol', None))
    if residual is None:
        logger.note(log, '%-28s not available', 'Z-vector residual (max abs)')
    else:
        logger.note(log, '%-28s %.6e', 'Z-vector residual (max abs)', residual)
    if limit is not None:
        logger.info(log, '%-28s %.3e', 'Residual check limit', limit)
    verdict = 'NOT CHECKED'
    if residual is not None and limit is not None:
        verdict = 'PASS' if np.isfinite(residual) and residual <= limit else 'WARNING'
    logger.note(log, '%-28s %s', 'Residual check', verdict)
    if de.size:
        logger.info(log, '%-28s %.6e Ha/Bohr', 'Max |force component|',
                    np.max(np.abs(de)))
        logger.info(log, '%-28s %.6e Ha/Bohr', 'RMS force component',
                    np.sqrt(np.mean(de**2)))
    full = atoms is None or sorted(atoms) == list(range(grad.mol.natm))
    label = 'Sum of forces' if full else 'Selected-atom force sum'
    logger.info(log, '%s [X Y Z] = % .6e % .6e % .6e Ha/Bohr',
                label, *(-np.sum(de, axis=0)))
    logger.note(log, '=================================================')
