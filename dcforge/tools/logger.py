'''DC-Forge result logging using the existing PySCF logger.

These helpers only read completed results. They never evaluate energy, Fock,
response, or residuals. NOTE prints results; INFO adds implementation choices.
No output file, verbosity switch, GPU timer or persistence format is introduced.
'''

import numpy as np
from pyscf.lib import logger
from pyscf.grad.rhf import _write


def _header(obj, mf, title, unrestricted):
    logger.note(obj, '\n================ DC-Forge %s ================', title)
    # All currently supported workflows use HF source densities. A future
    # DFA-source implementation must pass its actual source method here.
    logger.note(obj, '%-28s HF-%s', 'Method', mf.xc)
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


def _energy_value(mf, label, attribute):
    value = getattr(mf, attribute, None)
    if value is None:
        logger.note(mf, '%-28s not available', label)
    else:
        logger.note(mf, '%-28s %.15f Ha', label, value)


def energy_summary(mf, unrestricted=False):
    '''Print completed source, target and dispersion-inclusive total energies.

    Called only by the public workflow kernel, never by energy_tot().
    Source/target energies include their usual nuclear contribution;
    target energy excludes the separately reported dispersion correction.
    '''
    if mf.verbose < logger.NOTE:
        return
    _header(mf, mf, 'Energy', unrestricted)
    logger.note(mf, '%-28s %s', 'Source SCF converged',
                'Yes' if mf.mf_hf.converged else 'No')
    _energy_value(mf, 'Source energy', 'e_hf')
    _energy_value(mf, 'Target energy', 'e_dft')
    if getattr(mf, 'dispersion', None):
        _energy_value(mf, 'Dispersion energy', 'e_d2c')
    else:
        logger.note(mf, '%-28s not applied', 'Dispersion energy')
    _energy_value(mf, 'Total energy', 'e_tot')
    logger.note(mf, '=================================================')


def gradient_summary(grad, unrestricted=False):
    '''Replace the upstream gradient finalizer without duplicating its table.

    Uses the existing final host gradient and physical residual. No device
    arrays are copied for reporting. Missing diagnostics are not a PASS.
    The residual verdict is diagnostic only and never changes solver policy.
    '''
    if grad.verbose < logger.NOTE:
        return
    mf = grad.base
    _header(grad, mf, 'Gradient', unrestricted)
    logger.note(grad, '%-28s dE/dR (including selected dispersion)', 'Quantity')
    logger.note(grad, '%-28s Ha/Bohr', 'Unit')
    logger.note(grad, '%-28s F = -dE/dR', 'Force convention')
    de = grad.de
    if not isinstance(de, np.ndarray):
        # Do not introduce a device transfer/synchronization just for logging.
        logger.note(grad, 'Gradient table unavailable: result is not a host array')
        return
    atoms = getattr(grad, 'atmlst', None)
    _write(grad, grad.mol, de, atoms)
    for label, attribute in (
            ('Grid response', 'grid_response'),
            ('Grid response backend', 'grid_response_backend_used'),
            ('Auxiliary-basis response', 'auxbasis_response'),
            ('Z-vector solver', 'z_solver')):
        value = getattr(grad, attribute, None)
        if value is not None:
            logger.info(grad, '%-28s %s', label, value)
    logger.note(grad, '%-28s %s', 'Source SCF converged',
                'Yes' if mf.mf_hf.converged else 'No')
    tol = getattr(grad, 'cphf_conv_tol', None)
    if tol is not None:
        logger.info(grad, '%-28s %.3e', 'CPHF tolerance', tol)
    info = getattr(grad, 'z_info', None) or {}
    residual = getattr(grad, 'z_residual_max_abs', None)
    if residual is None:
        residual = info.get('residual_max_abs')
    limit = getattr(grad, 'cphf_residual_warn_tol',
                    getattr(grad, 'cphf_residual_tol', None))
    if residual is None:
        logger.note(grad, '%-28s not available', 'Z-vector residual (max abs)')
    else:
        logger.note(grad, '%-28s %.6e', 'Z-vector residual (max abs)', residual)
    if limit is not None:
        logger.note(grad, '%-28s %.3e', 'Residual check limit', limit)
    verdict = 'NOT CHECKED'
    if residual is not None and limit is not None:
        verdict = 'PASS' if np.isfinite(residual) and residual <= limit else 'WARNING'
    logger.note(grad, '%-28s %s', 'Residual check', verdict)
    if de.size:
        logger.note(grad, '%-28s %.6e Ha/Bohr', 'Max |gradient component|',
                    np.max(np.abs(de)))
        logger.note(grad, '%-28s %.6e Ha/Bohr', 'RMS gradient component',
                    np.sqrt(np.mean(de**2)))
    full = atoms is None or sorted(atoms) == list(range(grad.mol.natm))
    label = 'Sum of gradients' if full else 'Selected-atom gradient sum'
    logger.note(grad, '%s [X Y Z] = % .6e % .6e % .6e Ha/Bohr',
                label, *np.sum(de, axis=0))
    logger.note(grad, '=================================================')
