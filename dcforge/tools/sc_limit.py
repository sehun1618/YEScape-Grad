"""Conservative identical-functional limit; bypass all orbital-response work."""
import numpy as np
from pyscf.lib import logger
from dcforge.cpu.energy.rnsc import _source_dfa_signature, _make_source_signature
from dcforge.tools.logger import gradient_stage


def equivalent_source(grad, mo_energy, mo_coeff, mo_occ):
    """Require matching finite-quadrature Hamiltonians and the recorded SCF state.

    This deliberately uses exact XC strings/arrays. Equivalent aliases or different
    grid orderings may take the slower general path; no tolerance-based shortcut.
    """
    mf = grad.base
    source = mf.mf_source
    if not grad.sc_limit or not source.converged or not hasattr(source, 'xc'):
        return False
    if not grad.grid_response or not grad.source_grid_response:
        return False
    if mf.dispersion or mf._source_signature != _make_source_signature(mf):
        return False
    # ponytail: exact signatures may miss equivalent aliases; canonicalize only with tests.
    a, b = _source_dfa_signature(mf), _source_dfa_signature(source)
    # Density pruning settings need not match if the resulting quadratures do.
    if a[:2] != b[:2] or a[3:] != b[3:]:
        return False
    if type(mf._numint) is not type(source._numint):
        return False
    if mf._numint.libxc is not source._numint.libxc:
        return False
    gpu = any(c.__module__.startswith('dcforge.gpu.') for c in type(mf).__mro__)
    if gpu:
        import cupy as xp
    else:
        xp = np
    for actual, stationary in ((mo_energy, source.mo_energy),
                               (mo_coeff, source.mo_coeff), (mo_occ, source.mo_occ),
                               (mf.grids.coords, source.grids.coords),
                               (mf.grids.weights, source.grids.weights)):
        if actual is None or stationary is None:
            return False
        if not bool(xp.array_equal(xp.asarray(actual), xp.asarray(stationary))):
            return False
    return True


def sc_gradient_if_equivalent(grad, mo_energy, mo_coeff, mo_occ, dm, Co, atmlst):
    """Return the ordinary electronic SC gradient, or None for the NSC path."""
    grad.sc_limit_used = False
    grad.source_grid_response_backend_used = 'inactive'
    if not equivalent_source(grad, mo_energy, mo_coeff, mo_occ):
        return None
    grad.sc_limit_used = True
    logger.info(grad, 'Identical source/target: ordinary SC gradient; NSC response skipped')
    gpu = any(c.__module__.startswith('dcforge.gpu.') for c in type(grad).__mro__)
    unrestricted = np.ndim(mo_occ) == 2
    with gradient_stage(grad, 'SC explicit gradient (no Z-vector or source-Fock response)'):
        if gpu:
            from gpu4pyscf.grad import rhf, uhf
            # Native providers dispatch to target UKS/RKS and DF energy_ee.
            fn = uhf.grad_elec if unrestricted else rhf.grad_elec
            de = np.asarray(fn(grad, mo_energy, mo_coeff, mo_occ, atmlst))
            grad.de_standard_elec = de
            grad.de_ovlp_correction = np.zeros_like(de)
        else:
            from dcforge.cpu.grad.skeleton import skeleton_gradient, skeleton_gradient_uhf
            s1 = grad.get_ovlp(grad.mol)
            if unrestricted:
                foo = tuple(np.diag(e[o > 0]) for e, o in zip(mo_energy, mo_occ))
                de = skeleton_gradient_uhf(grad, dm, foo, Co, s1, atmlst)
            else:
                foo = np.diag(mo_energy[mo_occ > 0])
                de = skeleton_gradient(grad, dm, foo, Co, s1, atmlst)
    grad.de_skel_elec = de
    for name in ('de_fock', 'de_ovlp_energy', 'de_ovlp_response', 'de_nvar',
                 'de_d2c', 'de_source_hjk', 'de_source_xc'):
        setattr(grad, name, np.zeros_like(de))
    grad.d2c_result = None
    grad.s1oo = grad.s1vo = grad.zvec = None
    # No solve happened: do not report fabricated zero residual / PASS.
    for name in ('z_info', 'z_rhs_max_abs', 'z_rhs_max_abs_spin',
                 'z_residual_max_abs', 'z_residual_max_abs_spin', 'z_residual_relative'):
        setattr(grad, name, None)
    grad.z_residual_warning = False
    grad.z_solver = 'not required (SC limit)'
    grad.response_backend_used = 'inactive (SC limit)'
    grad.fock_deriv_method_used = 'inactive (SC limit)'
    return de
