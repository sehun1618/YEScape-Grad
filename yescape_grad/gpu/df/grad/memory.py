# YEScape-Grad
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

"""Bounded GPU4PySCF memory planning for molecular DF derivatives."""

from contextlib import contextmanager
from operator import index
from threading import RLock

import gpu4pyscf
import cupy as cp
from gpu4pyscf.df.grad import tdrhf as df_tdrhf_grad

ADAPTIVE_FOCK_DERIV_PLANNER_LIMIT = "adaptive"
DEFAULT_FOCK_DERIV_PLANNER_LIMIT_BYTES = ADAPTIVE_FOCK_DERIV_PLANNER_LIMIT
DEFAULT_FOCK_DERIV_PLANNER_WORKSPACE_BYTES = 4 * 1024**3
_PLANNER_LOCK = RLock()


def _gpu4pyscf_version():
    return getattr(gpu4pyscf, "__version__", None)


def normalize_fock_deriv_planner_limit(limit_bytes):
    """Return an adaptive request, positive byte ceiling, or ``None``."""
    if limit_bytes is None:
        return None
    if limit_bytes == ADAPTIVE_FOCK_DERIV_PLANNER_LIMIT:
        return ADAPTIVE_FOCK_DERIV_PLANNER_LIMIT
    if isinstance(limit_bytes, bool):
        raise TypeError(
            "fock_deriv_planner_limit_bytes must be 'adaptive', an integer, or None"
        )
    try:
        limit_bytes = index(limit_bytes)
    except TypeError as error:
        raise TypeError(
            "fock_deriv_planner_limit_bytes must be 'adaptive', an integer, or None"
        ) from error
    if limit_bytes <= 0:
        raise ValueError("fock_deriv_planner_limit_bytes must be positive")
    return int(limit_bytes)


def _factor_rank(dm, nao):
    """Return the rank seen by GPU4PySCF after tagged-DM symmetrization."""
    factor_l = getattr(dm, "factor_l", None)
    factor_r = getattr(dm, "factor_r", None)
    if factor_l is None:
        # Dense untagged matrices are factorized inside GPU4PySCF.  Use the
        # full AO rank as a conservative estimate without doing an SVD here.
        return int(nao)

    rank = int(factor_l.shape[-1])
    if getattr(dm, "symmetrize", 0) in (1, 2):
        rank += int((factor_r if factor_r is not None else factor_l).shape[-1])
    return rank


def _density_pair_ranks(dm_pair, nao):
    """Return the two effective factor ranks for one upstream density pair."""
    if isinstance(dm_pair, (tuple, list)):
        if len(dm_pair) != 2:
            raise ValueError("each density pair must contain exactly two matrices")
        dm1, dm2 = dm_pair
    elif getattr(dm_pair, "ndim", 0) == 2:
        dm1 = dm2 = dm_pair
    elif getattr(dm_pair, "shape", (0,))[0] == 2:
        dm1, dm2 = dm_pair[0], dm_pair[1]
    else:
        raise ValueError("density pairs must be two matrices or one matrix pair")
    return _factor_rank(dm1, nao), _factor_rank(dm2, nao)


def _round_up(value, quantum):
    return (value + quantum - 1) // quantum * quantum


def adaptive_derivative_planner_limit(
    int3c2e_opt, dm_pairs, workspace_bytes=DEFAULT_FOCK_DERIV_PLANNER_WORKSPACE_BYTES,
    *, sum_results=True,
):
    """Estimate a size-aware ceiling for GPU4PySCF DF derivatives.

    GPU4PySCF subtracts the two occupied-pair tensors and dense AO matrices
    from its apparent free memory before choosing auxiliary batch sizes.  A
    fixed 4 GiB ceiling is adequate only while those persistent intermediates
    fit inside it.  This model keeps that established ceiling for small jobs;
    otherwise it adds the same 4 GiB working headroom to the calculated fixed
    storage and rounds to one headroom-sized allocation quantum.

    Args:
        int3c2e_opt : gpu4pyscf.df.int3c2e.Int3c2eOpt
            Built three-center integral optimizer.  Its sorted Cartesian AO
            and auxiliary dimensions match the private derivative kernel.
        dm_pairs : sequence
            Density-matrix pairs passed to ``_jk_energies_per_atom``.
        workspace_bytes : int
            Small-system ceiling and extra auxiliary-batch headroom.
        sum_results : bool
            Whether upstream sums density-pair contributions into two buffers.

    Returns:
        dict
            Resolved byte ceiling and the dimensions used to derive it.
    """
    workspace_bytes = normalize_fock_deriv_planner_limit(workspace_bytes)
    if not isinstance(workspace_bytes, int):
        raise TypeError("workspace_bytes must be a positive integer")
    if not dm_pairs:
        raise ValueError("dm_pairs must not be empty")

    nao = int(int3c2e_opt.mol.nao)
    naux = int(int3c2e_opt.auxmol.nao)
    n_dm = len(dm_pairs)
    pair_ranks = []
    fixed_intermediate_bytes = 0
    for dm_pair in dm_pairs:
        rank1, rank2 = _density_pair_ranks(dm_pair, nao)
        pair_bytes = 2 * naux * rank1 * rank2 * 8 + 2 * nao**2 * 8
        pair_ranks.append(
            {
                "left": rank1,
                "right": rank2,
                "fixed_intermediate_bytes": pair_bytes,
            }
        )
        fixed_intermediate_bytes += pair_bytes

    pair_addresses = int3c2e_opt.pair_and_diag_indices(
        cart=True, original_ao_order=False
    )[0]
    nao_pair = len(pair_addresses)
    laux = int(int3c2e_opt.auxmol.uniq_l_ctr[:, 0].max())
    minimum_aux_batch_size = (laux + 1) * (laux + 2) // 2 + 1
    # GPU4PySCF 1.9 uses two AO-pair buffers when summing observables,
    # independently of the number of density pairs. The planner reserves
    # only half of its remaining memory for these buffers.
    n_pair_buffers = 2 if sum_results else n_dm
    minimum_batch_workspace_bytes = (
        2 * n_pair_buffers * nao_pair * 8 * minimum_aux_batch_size
    )
    minimum_required_bytes = (
        fixed_intermediate_bytes + minimum_batch_workspace_bytes
    )

    if minimum_required_bytes <= workspace_bytes:
        resolved_limit_bytes = workspace_bytes
    else:
        resolved_limit_bytes = _round_up(
            max(
                fixed_intermediate_bytes + workspace_bytes,
                minimum_required_bytes,
            ),
            workspace_bytes,
        )

    return {
        "workspace_bytes": workspace_bytes,
        "fixed_intermediate_bytes": fixed_intermediate_bytes,
        "minimum_batch_workspace_bytes": minimum_batch_workspace_bytes,
        "minimum_required_bytes": minimum_required_bytes,
        "resolved_limit_bytes": resolved_limit_bytes,
        "nao_pair": nao_pair,
        "minimum_aux_batch_size": minimum_aux_batch_size,
        "factor_ranks": pair_ranks,
    }


@contextmanager
def bounded_derivative_planner(limit_bytes):
    """Temporarily bound the upstream derivative planner's free-memory view.

    GPU4PySCF 1.9.0 derives both the auxiliary batch and dense AO block from
    its module-local ``get_avail_mem`` symbol.  The adapter caps only that
    planning query; it does not impose a CuPy pool limit or change integral
    precision.  A process-wide lock serializes YEScape-Grad adapter scopes, and
    the original upstream symbol is restored on every exit path.
    """
    limit_bytes = normalize_fock_deriv_planner_limit(limit_bytes)
    if limit_bytes == ADAPTIVE_FOCK_DERIV_PLANNER_LIMIT:
        raise ValueError("adaptive planning requires density-pair dimensions")
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
    # Symmetrizing an occupied/transition factor can give 2*nocc > nao.
    # The 1.9 derivative contraction uses an AO-sized transpose workspace.
    # For this case only, let upstream refactor the identical dense density
    # (rank <= nao); retain inexpensive tagged factors for the usual case.
    nao = int(int3c2e_opt.mol.nao)
    dm_pairs = [
        [dm.view(cp.ndarray) if _factor_rank(dm, nao) > nao else dm
         for dm in pair] if isinstance(pair, (tuple, list)) else (pair.view(cp.ndarray) if _factor_rank(pair, nao) > nao else pair)
        for pair in dm_pairs
    ]
    requested_limit = getattr(
        mf_grad,
        "fock_deriv_planner_limit_bytes",
        DEFAULT_FOCK_DERIV_PLANNER_LIMIT_BYTES,
    )
    requested_limit = normalize_fock_deriv_planner_limit(requested_limit)
    model = None
    if requested_limit == ADAPTIVE_FOCK_DERIV_PLANNER_LIMIT:
        model = adaptive_derivative_planner_limit(
            int3c2e_opt, dm_pairs, sum_results=sum_results)
        limit_bytes = model["resolved_limit_bytes"]
    else:
        limit_bytes = requested_limit

    with bounded_derivative_planner(limit_bytes) as detail:
        detail.update(
            {
                "nao": int(int3c2e_opt.mol.nao),
                "naux": int(int3c2e_opt.auxmol.nao),
                "density_pair_count": len(dm_pairs),
                "sum_results": bool(sum_results),
            }
        )
        if model is not None:
            detail.update(model)
            detail["mode"] = "adaptive_bounded"
            detail["requested_limit_bytes"] = requested_limit
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
