# YEScape-Grad
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

'''Local compatibility for PySCF PR #3334 (CPHF Krylov termination).

PySCF 2.14 and GPU4PySCF 1.8.1 use a fixed Krylov ``lindep=1e-13``.
Because this threshold is compared with squared vector norms, requesting a
smaller CPHF tolerance alone does not prevent early termination. PR #3334
uses ``max(tol**2, eps)`` and also forwards it to the CPU multi-RHS QR.

Reference: https://github.com/pyscf/pyscf/pull/3334
The CPHF/UCPHF equations remain the installed upstream functions. Private
function globals bind only their Krylov call; neither site-packages nor
process-wide PySCF/GPU4PySCF functions are monkeypatched. This matters when
independent solvers run concurrently or a response callback nests a solve.

GPU4PySCF >= 1.9.0 uses its native CPHF/UCPHF implementation, which already
sets this cutoff in both overlap branches. Older GPU versions and CPU PySCF
retain the local shim. Before retiring the CPU shim, check BOTH CPHF/UCPHF
and Krylov QR (notably on upgrading to PySCF 2.15).
The epsilon floor is upstream policy, NOT a physical-residual guarantee.
Keep the caller's independent residual checks and existing fallback policy.
'''

from inspect import signature
from types import FunctionType, SimpleNamespace

import numpy as np


def _bind_globals(function, namespace):
    '''Reuse upstream bytecode with private name bindings and defaults.'''
    bound = FunctionType(function.__code__, namespace, function.__name__,
                         function.__defaults__, function.__closure__)
    bound.__kwdefaults__ = function.__kwdefaults__
    return bound


def solve(module, *args, **kwargs):
    '''Call upstream CPHF/UCPHF with the PR #3334 linear-dependence policy.

    Args:
        module : pyscf.scf.cphf/ucphf or gpu4pyscf.scf.cphf/ucphf
            Backend owning the equations and array representation.
        *args, **kwargs :
            Unchanged arguments to that module's ``solve``. Both ``s1=None``
            and overlap-dependent response, level shifts, batched RHS and
            backend-specific initial guesses retain upstream semantics.

    Returns:
        Upstream ``(mo1, mo_e1)`` without changing signs or response equations.

    Notes:
        Only YEScape-Grad callers opt into this compatibility layer. SCF, SOSCF,
        unrelated native PySCF calls and the separate mixed-Z3 solver are
        not modified. The linear-dependence cutoff is never below machine eps.
    '''
    bound = signature(module.solve).bind(*args, **kwargs)
    bound.apply_defaults()
    tol = float(bound.arguments['tol'])
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError('CPHF tolerance must be finite and positive')
    gpu = module.__name__.startswith('gpu4pyscf.')
    if gpu:
        import gpu4pyscf
        if np.lib.NumpyVersion(gpu4pyscf.__version__) >= '1.9.0':
            return module.solve(*args, **kwargs)
    namespace = vars(module).copy()
    native_krylov = namespace['krylov'] if gpu else namespace['lib'].krylov

    def krylov(aop, b, **options):
        tol = float(options['tol'])
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError('CPHF tolerance must be finite and positive')
        lindep = max(tol**2, np.finfo(b.real.dtype).eps)
        options['lindep'] = lindep
        if gpu:
            # GPU4PySCF already forwards lindep to both _stable_qr calls.
            return native_krylov(aop, b, **options)

        # PySCF 2.14's two _qr calls omit lindep. Bind the default locally;
        # upstream versions passing an explicit third argument also work.
        native_qr = native_krylov.__globals__['_qr']

        def qr(xs, dot, lindep=lindep):
            return native_qr(xs, dot, lindep)

        krylov_globals = dict(native_krylov.__globals__, _qr=qr)
        return _bind_globals(native_krylov, krylov_globals)(aop, b, **options)

    if gpu:
        namespace['krylov'] = krylov
    else:
        namespace['lib'] = SimpleNamespace(**dict(vars(module.lib), krylov=krylov))
    for name in ('solve_nos1', 'solve_withs1'):
        namespace[name] = _bind_globals(getattr(module, name), namespace)
    return _bind_globals(module.solve, namespace)(*args, **kwargs)
