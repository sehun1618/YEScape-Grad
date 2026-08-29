"""Bounded GPU4PySCF memory planning for molecular DF derivatives."""

from contextlib import contextmanager
from operator import index
from threading import RLock

import gpu4pyscf
from gpu4pyscf.df.grad import tdrhf as df_tdrhf_grad

DEFAULT_FOCK_DERIV_PLANNER_LIMIT_BYTES = 4 * 1024**3
_PLANNER_LOCK = RLock()


def _gpu4pyscf_version():
    return getattr(gpu4pyscf, "__version__", None)


def normalize_fock_deriv_planner_limit(limit_bytes):
    """Return one positive byte ceiling or ``None`` for upstream auto planning."""
    if limit_bytes is None:
        return None
    if isinstance(limit_bytes, bool):
        raise TypeError("fock_deriv_planner_limit_bytes must be an integer or None")
    try:
        limit_bytes = index(limit_bytes)
    except TypeError as error:
        raise TypeError(
            "fock_deriv_planner_limit_bytes must be an integer or None"
        ) from error
    if limit_bytes <= 0:
        raise ValueError("fock_deriv_planner_limit_bytes must be positive")
    return int(limit_bytes)


@contextmanager
def bounded_derivative_planner(limit_bytes):
    """Temporarily bound the upstream derivative planner's free-memory view.

    GPU4PySCF 1.8.1 derives both the auxiliary batch and dense AO block from
    its module-local ``get_avail_mem`` symbol.  The adapter caps only that
    planning query; it does not impose a CuPy pool limit or change integral
    precision.  A process-wide lock serializes DC-Forge adapter scopes, and
    the original upstream symbol is restored on every exit path.
    """
    limit_bytes = normalize_fock_deriv_planner_limit(limit_bytes)
    detail = {
        "mode": "upstream_auto" if limit_bytes is None else "bounded",
        "requested_limit_bytes": limit_bytes,
        "gpu4pyscf_version": _gpu4pyscf_version(),
        "upstream_module": df_tdrhf_grad.__name__,
        "upstream_symbol": "get_avail_mem",
        "queries": [],
    }
    if limit_bytes is None:
        yield detail
        return

    with _PLANNER_LOCK:
        original = getattr(df_tdrhf_grad, "get_avail_mem", None)
        if not callable(original):
            raise TypeError("GPU4PySCF derivative planner get_avail_mem is unavailable")

        def bounded(*args, **kwargs):
            available_bytes = int(original(*args, **kwargs))
            effective_bytes = min(available_bytes, limit_bytes)
            exclude_memory_pool = (
                bool(args[0])
                if args
                else bool(kwargs.get("exclude_memory_pool", False))
            )
            detail["queries"].append(
                {
                    "exclude_memory_pool": exclude_memory_pool,
                    "upstream_available_bytes": available_bytes,
                    "effective_available_bytes": effective_bytes,
                }
            )
            return effective_bytes

        df_tdrhf_grad.get_avail_mem = bounded
        try:
            yield detail
        finally:
            df_tdrhf_grad.get_avail_mem = original


def jk_energies_per_atom_bounded(
    mf_grad, int3c2e_opt, dm_pairs, *, j_factor, k_factor, sum_results, verbose
):
    """Call the private upstream derivative kernel under the configured ceiling."""
    limit_bytes = getattr(
        mf_grad,
        "fock_deriv_planner_limit_bytes",
        DEFAULT_FOCK_DERIV_PLANNER_LIMIT_BYTES,
    )
    with bounded_derivative_planner(limit_bytes) as detail:
        detail.update(
            {
                "nao": int(int3c2e_opt.mol.nao),
                "naux": int(int3c2e_opt.auxmol.nao),
                "density_pair_count": len(dm_pairs),
                "sum_results": bool(sum_results),
            }
        )
        try:
            return df_tdrhf_grad._jk_energies_per_atom(
                int3c2e_opt,
                dm_pairs,
                j_factor=j_factor,
                k_factor=k_factor,
                sum_results=sum_results,
                verbose=verbose,
            )
        finally:
            detail["query_count"] = len(detail["queries"])
            mf_grad.fock_deriv_planner = detail
