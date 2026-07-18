"""Explicit ownership for uniform and multigrid target-DFA integration.

Source HF always retains the selected FFTDF/GDF/RSDF provider.  This module
only changes the fixed-density target J/XC evaluation and never changes the
target hcore or pseudopotential owner implicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cupy as cp
import numpy as np
from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.pbc.dft import multigrid_v2
from pyscf_pbc_dcdft.dft.target_integrator import (
    MULTIGRID_J_XC,
    MULTIGRID_XC,
    UNIFORM,
    normalize_target_integrator,
    normalize_target_multigrid_mesh,
)
from pyscf_pbc_dcdft.lib.kpoints import normalize_kpts


@dataclass(frozen=True)
class TargetVeffEvaluation:
    """One target J/XC evaluation with explicit numerical ownership."""

    veff: cp.ndarray
    numerical_integrator: object | None
    metadata: dict[str, object]


def evaluate_target_hcore(target):
    """Return hcore from the selected DF provider, never from multigrid."""

    return target.get_hcore()


def evaluate_target_veff(
    target,
    density,
    *,
    with_df,
    kpts,
    target_integrator=UNIFORM,
    target_multigrid_mesh=None,
) -> TargetVeffEvaluation:
    """Return target ``J + Vxc`` under one explicit ownership policy."""

    mode = normalize_target_integrator(target_integrator)
    mesh = normalize_target_multigrid_mesh(target_multigrid_mesh)
    density = cp.asarray(density)
    if mode == UNIFORM:
        if mesh is not None:
            raise ValueError(
                "target_multigrid_mesh requires a multigrid target integrator"
            )
        veff = target.get_veff(target.cell, density)
        return TargetVeffEvaluation(
            veff=veff,
            numerical_integrator=None,
            metadata=_metadata(
                mode,
                target,
                mesh=np.asarray(target.cell.mesh, dtype=int),
            ),
        )

    ni = multigrid_v2.MultiGridNumInt(target.cell)
    if mesh is not None:
        ni.mesh = mesh
    points = normalize_kpts(kpts)
    with_j = mode == MULTIGRID_J_XC
    cp.cuda.Stream.null.synchronize()
    evaluation_start = perf_counter()
    _nelec, exc, multigrid_veff = ni.nr_rks(
        target.cell,
        target.grids,
        target.xc,
        density,
        hermi=1,
        kpts=points,
        with_j=with_j,
        verbose=target.verbose,
    )
    cp.cuda.Stream.null.synchronize()
    evaluation_seconds = perf_counter() - evaluation_start
    if with_j:
        ecoul = getattr(multigrid_veff, "ecoul", None)
        if ecoul is None:
            raise RuntimeError("multigrid target J+XC did not expose ecoul")
        veff = tag_array(cp.asarray(multigrid_veff), ecoul=ecoul, exc=exc)
    elif mode == MULTIGRID_XC:
        selected_j = _selected_df_coulomb(with_df, density, points)
        ecoul = _coulomb_energy(density, selected_j, len(points))
        veff = tag_array(
            cp.asarray(multigrid_veff) + selected_j,
            ecoul=ecoul,
            exc=exc,
        )
    else:  # pragma: no cover - normalize_target_integrator owns this guard
        raise AssertionError(mode)

    return TargetVeffEvaluation(
        veff=veff,
        numerical_integrator=ni,
        metadata=_metadata(
            mode,
            target,
            mesh=np.asarray(ni.mesh, dtype=int),
            evaluation_seconds=evaluation_seconds,
            task_metadata=_multigrid_task_metadata(ni),
        ),
    )


def _selected_df_coulomb(with_df, density, kpts):
    value, _exchange = with_df.get_jk(
        density,
        hermi=1,
        kpts=kpts,
        kpts_band=None,
        with_j=True,
        with_k=False,
        omega=None,
        exxdiv=None,
    )
    if value is None:
        raise RuntimeError("selected DF provider did not return target J")
    return cp.asarray(value)


def _coulomb_energy(density, coulomb, nkpts: int):
    density = cp.asarray(density)
    coulomb = cp.asarray(coulomb)
    if density.ndim == 2:
        density = density[None]
    if coulomb.ndim == 2:
        coulomb = coulomb[None]
    return 0.5 * cp.einsum("kij,kji->", density, coulomb).real / int(nkpts)


def _multigrid_task_metadata(ni):
    tasks = getattr(ni, "tasks", None) or ()
    levels = []
    for localized, diffused in tasks:
        entry = {
            "localized_mesh": [
                int(value) for value in np.asarray(localized.mesh).reshape(3)
            ],
            "localized_grid_points": int(np.prod(localized.mesh)),
            "diffused_mesh": None,
            "diffused_grid_points": 0,
        }
        if diffused is not None:
            entry["diffused_mesh"] = [
                int(value)
                for value in np.asarray(diffused.mesh).reshape(3)
            ]
            entry["diffused_grid_points"] = int(np.prod(diffused.mesh))
        levels.append(entry)
    return {
        "task_count": len(tasks),
        "task_levels": levels,
        "sorted_gaussian_pair_groups": len(
            getattr(ni, "sorted_gaussian_pairs", None) or ()
        ),
    }


def _metadata(
    mode,
    target,
    *,
    mesh,
    evaluation_seconds=None,
    task_metadata=None,
):
    multigrid = mode != UNIFORM
    metadata = {
        "backend": "multigrid_v2" if multigrid else "numint",
        "mode": mode,
        "scope": "target_dfa",
        "device": "gpu",
        "target_hcore_owner": "selected_df_backend",
        "target_j_owner": (
            "multigrid_v2" if mode == MULTIGRID_J_XC else "selected_df_backend"
        ),
        "target_xc_owner": "multigrid_v2" if multigrid else "numint",
        "target_pp_owner": "selected_df_backend",
        "mesh": [int(value) for value in np.asarray(mesh).reshape(3)],
        "target_xc": ["PBE", "SCAN", "R2SCAN"],
        "silent_cpu_fallback": False,
        "target_class": target.__class__.__name__,
    }
    if evaluation_seconds is not None:
        metadata["energy_evaluation_wall_time_s"] = float(
            evaluation_seconds
        )
    if task_metadata is not None:
        metadata.update(task_metadata)
    return metadata


def target_numint_metadata(target) -> dict[str, object]:
    """Return the metadata attached by the target evaluator."""

    metadata = getattr(target, "_dcdft_target_numint_metadata", None)
    if metadata is None:
        return _metadata(
            UNIFORM,
            target,
            mesh=np.asarray(target.cell.mesh, dtype=int),
        )
    return dict(metadata)
