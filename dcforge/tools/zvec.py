# DC-Forge
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

'''Shared reporting helpers for independently checked Z-vector residuals.'''

import math


def warn_z_residual(log, info, warn_tol):
    '''Warn when the reconstructed physical Z residual exceeds ``warn_tol``.

    ``cphf_conv_tol`` controls the internal PySCF Krylov iteration and is not
    itself a bound on the original Z-vector equation. Callers therefore pass
    a separate warning threshold after reconstructing that physical equation.
    This helper reports the discrepancy but deliberately neither retries the
    solve nor rejects the gradient.

    Args:
        log : pyscf.lib.logger.Logger-like object
            Logger receiving the warning.
        info : dict
            Solver diagnostics containing ``residual_max_abs`` and
            ``residual_relative``.
        warn_tol : float or None
            Positive absolute physical-residual threshold. ``None`` disables
            the warning while retaining the diagnostic fields.

    Returns:
        bool
            Whether the physical residual triggered a warning.
    '''
    if warn_tol is None:
        info['physical_residual_warn_tol'] = None
        info['physical_residual_warning'] = False
        return False

    warn_tol = float(warn_tol)
    if not math.isfinite(warn_tol) or warn_tol <= 0:
        raise ValueError('cphf_residual_warn_tol must be positive and finite')

    residual = float(info['residual_max_abs'])
    warned = not math.isfinite(residual) or residual > warn_tol
    info['physical_residual_warn_tol'] = warn_tol
    info['physical_residual_warning'] = warned
    if warned:
        # PySCF's Logger exposes ``warn`` rather than stdlib ``warning``.
        emit_warning = log.warn
        emit_warning(
            'Z-vector residual (max abs) %.3e exceeds warning threshold %.3e '
            '(RHS-relative residual %.3e). The gradient is returned unchanged; '
            'validate sensitive cases against finite differences.',
            residual, warn_tol, float(info['residual_relative']))
    return warned


__all__ = ['warn_z_residual']
