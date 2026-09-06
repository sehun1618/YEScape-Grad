'''Localized GPU4PySCF periodic density-fitting compatibility helpers.

GPU4PySCF generalized the short-range ``int3c2e`` launch planner after its
first periodic-gradient implementation.  New runtimes expose the planner in
``gpu4pyscf.pbc.df.int3c2e`` with an explicit derivative tuple, whereas older
gradient modules expose an ``omega``-based positional helper.  DC-Forge uses
one capability-detected boundary so scientific modules do not depend on a
version string or duplicate the upstream shared-memory algebra.
'''

import inspect
import threading
from contextlib import contextmanager

from gpu4pyscf import __version__ as gpu4pyscf_version
from pyscf.pbc import df as _public_pbc_df
from pyscf.pbc.df import df as _pbc_df_implementation

try:
    from gpu4pyscf.pbc.df.int3c2e import (
        int3c2e_scheme as _int3c2e_scheme,
    )
    _INT3C2E_SCHEME_MODULE = 'gpu4pyscf.pbc.df.int3c2e'
except ImportError:  # pragma: no cover - retained for older GPU4PySCF
    from gpu4pyscf.pbc.df.grad.rhf import (
        int3c2e_scheme as _int3c2e_scheme,
    )
    _INT3C2E_SCHEME_MODULE = 'gpu4pyscf.pbc.df.grad.rhf'

_INT3C2E_SCHEME_SIGNATURE = inspect.signature(_int3c2e_scheme)
_INT3C2E_SCHEME_HAS_DERIV = (
    'deriv' in _INT3C2E_SCHEME_SIGNATURE.parameters)
_AUXCELL_LOCK = threading.RLock()


def make_auxcell(cell, auxbasis=None, exp_to_discard=None):
    '''Build a periodic auxiliary Cell across the PySCF 2.14 API boundary.

    PySCF's implementation-level ``make_modrho_basis`` retains the historical
    third exponent-cutoff argument.  The package-level ``make_auxcell`` alias
    in PySCF 2.14 accepts only ``(cell, auxbasis)``, while GPU4PySCF 1.8.1
    still calls it with three positional arguments.  This localized helper
    preserves the intended PySCF operation without copying basis logic.
    '''
    return _pbc_df_implementation.make_modrho_basis(
        cell, auxbasis, exp_to_discard)


@contextmanager
def compatible_auxcell_builder():
    '''Temporarily provide GPU4PySCF 1.8.1 its expected auxcell call ABI.'''
    parameters = inspect.signature(
        _public_pbc_df.make_auxcell).parameters
    if len(parameters) >= 3:
        yield
        return
    with _AUXCELL_LOCK:
        original = _public_pbc_df.make_auxcell
        _public_pbc_df.make_auxcell = make_auxcell
        try:
            yield
        finally:
            _public_pbc_df.make_auxcell = original


def derivative_int3c2e_scheme(gout_width=54, deriv=(1, 0, 0)):
    '''Return the upstream short-range derivative launch plan.

    Kwargs:
        gout_width : int
            Number of derivative output components consumed by one CUDA
            launch.  Periodic three-center nuclear derivatives use 54.
        deriv : tuple of three int
            Derivative order on the auxiliary, first AO and second AO
            centers.  DC-Forge currently requests ``(1, 0, 0)`` and obtains
            the remaining center permutations through the native kernel's
            established output convention.

    Returns:
        tuple
            ``(nsp_per_block, gout_stride, shared_memory_bytes)`` in the
            active GPU4PySCF layout.

    Notes:
        The legacy helper has no derivative tuple.  ``omega=-1`` selects its
        short-range derivative root count and is algebraically the old form
        of the explicit ``deriv`` request.  Capability detection, rather than
        the package version, chooses between the two APIs.
    '''
    if _INT3C2E_SCHEME_HAS_DERIV:
        return _int3c2e_scheme(
            gout_width=int(gout_width), deriv=tuple(int(x) for x in deriv))
    if tuple(deriv) != (1, 0, 0):
        raise NotImplementedError(
            'legacy GPU4PySCF int3c2e_scheme supports only the periodic '
            'first-derivative launch used by DC-Forge')
    return _int3c2e_scheme(-1, int(gout_width))


def derivative_int3c2e_scheme_metadata():
    '''Return provenance for the resolved GPU4PySCF launch-planner ABI.'''
    return {
        'gpu4pyscf_version': str(gpu4pyscf_version),
        'module': _INT3C2E_SCHEME_MODULE,
        'signature': str(_INT3C2E_SCHEME_SIGNATURE),
        'explicit_derivative_tuple': bool(_INT3C2E_SCHEME_HAS_DERIV),
    }
