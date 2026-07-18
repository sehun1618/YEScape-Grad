"""Reusable fixed-density target-DFA intermediates.

The DC energy and its analytic gradient use the same HF density and target
functional.  Keeping the expensive target Veff/Fock from the energy pass
avoids a second numerical-integration and Coulomb build in the gradient.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import cupy as cp
import numpy as np


@dataclass(frozen=True)
class TargetEvaluationWorkspace:
    """Device intermediates owned by one converged DC energy evaluation."""

    source_mf: object
    target_ks: object
    with_df: object
    xc: str
    kpts: np.ndarray
    density: cp.ndarray
    hcore: cp.ndarray
    veff: cp.ndarray
    target_fock: cp.ndarray
    target_integrator: str
    target_numint: object | None
    ownership: dict[str, object]
    target_multigrid_mesh: tuple[int, int, int]
    input_signature: str
    source_state: tuple[int, int, int, float]

    @classmethod
    def create(
        cls,
        *,
        source_mf,
        target_ks,
        with_df,
        xc,
        kpts,
        density,
        hcore,
        veff,
        target_integrator="uniform",
        target_numint=None,
        ownership=None,
        target_multigrid_mesh=None,
    ):
        """Capture one internally consistent energy/Fock evaluation."""

        hcore_array = cp.asarray(hcore)
        veff_array = cp.asarray(veff)
        return cls(
            source_mf=source_mf,
            target_ks=target_ks,
            with_df=with_df,
            xc=str(xc),
            kpts=np.asarray(kpts, dtype=float).reshape(-1, 3).copy(),
            density=cp.asarray(density),
            hcore=hcore_array,
            veff=veff_array,
            target_fock=hcore_array + veff_array,
            target_integrator=str(target_integrator),
            target_numint=target_numint,
            ownership={} if ownership is None else dict(ownership),
            target_multigrid_mesh=tuple(
                int(value)
                for value in (
                    np.asarray(source_mf.cell.mesh).reshape(3)
                    if target_multigrid_mesh is None
                    else target_multigrid_mesh
                )
            ),
            input_signature=_input_signature(
                source_mf.cell,
                kpts=kpts,
                xc=xc,
                target_integrator=target_integrator,
                target_multigrid_mesh=target_multigrid_mesh,
            ),
            source_state=_source_state(source_mf),
        )

    def compatible_with(self, dc_mf) -> bool:
        """Return whether ``dc_mf`` still describes this exact workspace."""

        if self.source_mf is not getattr(dc_mf, "source_scf", None):
            return False
        if self.target_ks is not getattr(dc_mf, "target_ks", None):
            return False
        if self.with_df is not getattr(dc_mf, "with_df", None):
            return False
        if self.xc != str(getattr(dc_mf, "xc", "")):
            return False
        if self.target_integrator != str(
            getattr(dc_mf, "target_integrator", "uniform")
        ):
            return False
        if self.source_state != _source_state(self.source_mf):
            return False
        return self.input_signature == _input_signature(
            dc_mf.cell,
            kpts=getattr(dc_mf, "kpts", self.kpts),
            xc=dc_mf.xc,
            target_integrator=getattr(
                dc_mf, "target_integrator", "uniform"
            ),
            target_multigrid_mesh=getattr(
                dc_mf, "target_multigrid_mesh", None
            ),
        )


def compatible_target_workspace(dc_mf):
    """Return the valid energy workspace attached to ``dc_mf``, if any."""

    workspace = getattr(dc_mf, "target_evaluation", None)
    if isinstance(workspace, TargetEvaluationWorkspace):
        if workspace.compatible_with(dc_mf):
            return workspace
    return None


def _source_state(source_mf) -> tuple[int, int, int, float]:
    return (
        id(getattr(source_mf, "mo_coeff", None)),
        id(getattr(source_mf, "mo_occ", None)),
        id(getattr(source_mf, "mo_energy", None)),
        float(getattr(source_mf, "e_tot", 0.0)),
    )


def _input_signature(
    cell,
    *,
    kpts,
    xc,
    target_integrator="uniform",
    target_multigrid_mesh=None,
) -> str:
    payload = (
        tuple(cell.atom_symbol(index) for index in range(cell.natm)),
        np.asarray(cell.atom_coords(unit="Bohr"), dtype=np.float64),
        np.asarray(cell.lattice_vectors(), dtype=np.float64),
        np.asarray(kpts, dtype=np.float64).reshape(-1, 3),
        tuple(int(value) for value in np.asarray(cell.mesh).reshape(3)),
        cell._basis,
        cell._pseudo,
        float(cell.precision),
        float(cell.omega),
        str(xc),
        str(target_integrator),
        (
            None
            if target_multigrid_mesh is None
            else tuple(
                int(value)
                for value in np.asarray(target_multigrid_mesh).reshape(3)
            )
        ),
    )
    return hashlib.sha256(repr(_freeze(payload)).encode("utf-8")).hexdigest()


def _freeze(value):
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return (array.dtype.str, array.shape, array.tobytes().hex())
    if isinstance(value, dict):
        return tuple(
            (repr(key), _freeze(item))
            for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value
