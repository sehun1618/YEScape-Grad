#!/usr/bin/env python
"""DC-DFT gradient with adjoint Ax(S1) and direct contracted DF-HF F1.

This is a standalone implementation of the S1-optimized gradient whose
non-variational correction avoids materialising the large
perturbation-indexed arrays ``B``, ``dm_s``, ``v_resp`` and ``AxS``.

For each nuclear perturbation A, the original expression is

    g_nv[A] = 4 * Z : (F1[A] - S1e[A] - 0.5*AxS[A])

It is evaluated here as three already-contracted vectors:

    g_nv = g_zf1 - 4*z_se - 2*z_ax

``z_ax`` uses the self-adjointness of the real RHF response operator, reducing
3*Natm response builds to one.  ``g_zf1`` uses GPU4PySCF's DF derivative-energy
kernel to contract F1 with Z directly.  The latter is algebraically exact but
can differ from ``Hessian.make_h1`` by a few 1e-9 a.u. because of FP64 summation
order.

The optimized path currently targets real, molecular, density-fitted RHF.  A
non-DF reference automatically falls back to the full implementation included
in this file.

Optional HDF5 checkpoints make the SCF and each expensive gradient stage
restartable.  In particular, an OOM in direct ``Z:F1`` can resume after the
saved skeleton and adjoint ``Z:Ax`` contractions without repeating CPHF.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import threading
import time

import cupy as cp
import h5py
import numpy as np
from pyscf import df
from pyscf import gto
from pyscf.lib import logger

from gpu4pyscf.df.df_jk import _make_factorized_dm
from gpu4pyscf.df.grad import tdrhf as df_tdrhf_grad
from gpu4pyscf.dft import rks as gpu_rks
from gpu4pyscf.grad import rhf as gpu_rhf_grad
from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.scf import cphf as gpu_cphf
from gpu4pyscf.scf import hf as gpu_hf
from gpu4pyscf.scf._response_functions import (
    _gen_rhf_response as _gen_rhf_response_gpu,
)

from ._gpu4pyscf_compat import df_hessian_get_jk_rhf


try:
    from gpu4pyscf.lib.cutensor import contract as _cutensor_contract
except ImportError:
    _cutensor_contract = None


_DF_DERIVATIVE_MEMORY_LOCK = threading.RLock()


def _contract(pattern, a, b):
    """Contract two GPU arrays, using cuTENSOR when it is available."""

    if _cutensor_contract is not None:
        return _cutensor_contract(pattern, a, b)
    return cp.einsum(pattern, a, b)


@contextmanager
def _df_derivative_memory_limit(memory_gb):
    """Temporarily cap memory reported to GPU4PySCF's DF derivative kernel."""

    if memory_gb is None:
        yield
        return
    memory_gb = float(memory_gb)
    if memory_gb <= 0:
        raise ValueError("df_deriv_memory_gb must be positive")

    with _DF_DERIVATIVE_MEMORY_LOCK:
        original = df_tdrhf_grad.get_avail_mem
        limit_bytes = int(memory_gb * 2**30)

        def limited_memory(*args, **kwargs):
            return min(int(original(*args, **kwargs)), limit_bytes)

        df_tdrhf_grad.get_avail_mem = limited_memory
        try:
            yield
        finally:
            df_tdrhf_grad.get_avail_mem = original


def _df_hf_response_mo(dfobj, dm_tag, left_coeff, occ_coeff,
                       *, direct_scf_tol=1.0e-14):
    """Return the requested MO block of ``(J - 0.5 K)[dm]``.

    GPU4PySCF consumes the low-rank tags attached to ``dm_tag`` and performs
    the final AO-to-MO projection inside the DF loop.  This avoids returning a
    full AO response matrix to the CPHF solver.
    """

    vj_mo, vk_mo = df_hessian_get_jk_rhf(
        dfobj, dm_tag, left_coeff, occ_coeff,
        direct_scf_tol=direct_scf_tol,
    )
    nset = dm_tag.reshape(-1, dm_tag.shape[-1], dm_tag.shape[-1]).shape[0]
    out = (vj_mo - 0.5 * vk_mo).reshape(
        int(nset), left_coeff.shape[1], occ_coeff.shape[1]
    )
    del vj_mo, vk_mo
    return out


def _make_mo_direct_ax(dfobj, Cv, Co, *, direct_scf_tol=1.0e-14):
    """Build the CPHF callback that returns only the virtual-occupied block."""

    nvir, nocc = Cv.shape[1], Co.shape[1]
    co2 = Co * 2.0

    def ax_vo(x_vo):
        original_shape = x_vo.shape
        x = cp.asarray(x_vo).reshape(-1, nvir, nocc)
        mo1 = _contract("pa,kai->kpi", Cv, x)
        dm_x = _contract("kpi,qi->kpq", mo1, co2)
        dm_x = dm_x + dm_x.transpose(0, 2, 1)
        dm_x = tag_array(dm_x, mo1=mo1, occ_coeff=Co)
        out = _df_hf_response_mo(
            dfobj, dm_x, Cv, Co, direct_scf_tol=direct_scf_tol
        )
        del mo1, dm_x
        return out.reshape(original_shape)

    return ax_vo


def _release_df_storage(with_df, label, log):
    """Drop a last-use DF tensor and return the released device memory."""

    cp.cuda.runtime.deviceSynchronize()
    free_before, _ = cp.cuda.runtime.memGetInfo()
    with_df.reset()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()
    cp.cuda.runtime.deviceSynchronize()
    free_after, _ = cp.cuda.runtime.memGetInfo()
    released_gib = (free_after - free_before) / 2**30
    log.info("  released %s DF storage: %.2f GiB", label, released_gib)
    return released_gib


class _CheckpointStore:
    """Incremental HDF5 storage for restartable gradient stages."""

    SCHEMA = "rdc_grad_gpu_pure_s1opt_adjoint_checkpoint_v2"
    TARGET_CHUNK_BYTES = 64 * 2**20

    def __init__(self, path, signature, metadata, resume):
        self.path = Path(path).expanduser().resolve()
        self.signature = signature
        self.metadata = metadata
        self.resume = bool(resume)

    def prepare(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.resume:
            if not self.path.is_file():
                raise FileNotFoundError(
                    f"Checkpoint file not found for --resume: {self.path}"
                )
            with h5py.File(self.path, "r") as handle:
                schema = handle.attrs.get("schema", "")
                signature = handle.attrs.get("signature", "")
            if schema != self.SCHEMA:
                raise ValueError(
                    f"Unsupported checkpoint schema in {self.path}: {schema!r}"
                )
            if signature != self.signature:
                raise ValueError(
                    "Checkpoint settings do not match the current molecule, "
                    "basis, functional, grid, or tolerances"
                )
            return

        with h5py.File(self.path, "w") as handle:
            handle.attrs["schema"] = self.SCHEMA
            handle.attrs["signature"] = self.signature
            handle.attrs["metadata_json"] = json.dumps(
                self.metadata, sort_keys=True
            )
            handle.attrs["created_unix_time"] = time.time()
            handle.require_group("stages")
            handle.flush()

    def has(self, stage):
        if not self.path.is_file():
            return False
        with h5py.File(self.path, "r") as handle:
            key = f"stages/{stage}"
            return key in handle and bool(handle[key].attrs.get("complete", 0))

    @classmethod
    def _chunk_rows(cls, shape, itemsize):
        if not shape or shape[0] == 0:
            return 1
        row_items = int(np.prod(shape[1:], dtype=np.int64)) if len(shape) > 1 else 1
        row_bytes = max(1, row_items * itemsize)
        return max(1, min(shape[0], cls.TARGET_CHUNK_BYTES // row_bytes))

    @classmethod
    def _write_array(cls, group, name, value):
        is_gpu = isinstance(value, cp.ndarray)
        shape = tuple(value.shape)
        dtype = np.dtype(value.dtype)
        if not shape:
            dataset = group.create_dataset(name, shape=(), dtype=dtype)
            dataset[()] = value.item()
            return

        rows = cls._chunk_rows(shape, dtype.itemsize)
        chunks = (rows,) + shape[1:]
        dataset = group.create_dataset(
            name, shape=shape, dtype=dtype, chunks=chunks
        )
        for p0 in range(0, shape[0], rows):
            p1 = min(shape[0], p0 + rows)
            block = cp.asnumpy(value[p0:p1]) if is_gpu else np.asarray(value[p0:p1])
            dataset[p0:p1] = block

    def save(self, stage, **values):
        writing = f"{stage}__writing"
        with h5py.File(self.path, "a") as handle:
            stages = handle.require_group("stages")
            if writing in stages:
                del stages[writing]
            group = stages.create_group(writing)
            for name, value in values.items():
                self._write_array(group, name, value)
            group.attrs["complete"] = 1
            group.attrs["saved_unix_time"] = time.time()
            handle.flush()
            if stage in stages:
                del stages[stage]
            stages.move(writing, stage)
            handle.flush()

    def load_gpu(self, stage, *names):
        result = {}
        with h5py.File(self.path, "r") as handle:
            group = handle[f"stages/{stage}"]
            if not bool(group.attrs.get("complete", 0)):
                raise RuntimeError(f"Incomplete checkpoint stage: {stage}")
            for name in names:
                dataset = group[name]
                if dataset.shape == ():
                    result[name] = dataset[()]
                    continue
                out = cp.empty(dataset.shape, dtype=dataset.dtype)
                rows = self._chunk_rows(dataset.shape, dataset.dtype.itemsize)
                for p0 in range(0, dataset.shape[0], rows):
                    p1 = min(dataset.shape[0], p0 + rows)
                    out[p0:p1].set(np.asarray(dataset[p0:p1]))
                result[name] = out
        return result

    def load_numpy(self, stage, *names):
        result = {}
        with h5py.File(self.path, "r") as handle:
            group = handle[f"stages/{stage}"]
            if not bool(group.attrs.get("complete", 0)):
                raise RuntimeError(f"Incomplete checkpoint stage: {stage}")
            for name in names:
                result[name] = np.asarray(group[name][()])
        return result


def _checkpoint_signature(mol, *, auxbasis, xc, atom_grid, grid_response,
                          scf_tol, cphf_tol, response_backend, share_df,
                          factorized_z_f1, df_deriv_memory_gb):
    metadata = {
        "molecule": {
            "atom_charges": np.asarray(mol.atom_charges()).tolist(),
            "atom_coords_bohr": np.asarray(mol.atom_coords()).tolist(),
            "basis": mol._basis,
            "ecp": mol._ecp,
            "pseudo": mol._pseudo,
            "charge": int(mol.charge),
            "spin": int(mol.spin),
            "cart": bool(mol.cart),
        },
        "auxbasis": repr(auxbasis),
        "xc": str(xc),
        "atom_grid": [int(x) for x in atom_grid],
        "grid_response": bool(grid_response),
        "scf_tol": float(scf_tol),
        "cphf_tol": float(cphf_tol),
        "response_backend": str(response_backend),
        "share_df": bool(share_df),
        "factorized_z_f1": bool(factorized_z_f1),
        "df_deriv_memory_gb": df_deriv_memory_gb,
    }
    encoded = json.dumps(metadata, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest(), metadata


def _direct_contracted_z_f1(mol, hf_gpu, C, occ, Co, Cv, Z,
                            *, memory_gb=None):
    """Return ``4 * Z:F1_vo`` without building the full derivative Fock.

    The one-electron derivative is obtained from GPU4PySCF's installed hcore
    derivative implementation.  The DF J/K derivative is evaluated as a cross
    derivative between the symmetrised Z transition density and the closed-
    shell reference density.
    """

    nocc = Co.shape[1]
    nmo = C.shape[1]
    npert = 3 * mol.natm

    # One-electron contribution.  This temporary contains only the hcore
    # derivative, not the much larger DF derivative Fock construction.
    hcore1 = gpu_rhf_grad.get_grad_hcore(
        hf_gpu.nuc_grad_method(), mo_coeff=C, mo_occ=occ
    ).reshape(npert, nmo, nocc)
    g_hcore = 4.0 * _contract("ai,Aai->A", Z, hcore1[:, nocc:, :])
    del hcore1

    # Z transition density and the closed-shell reference density.
    dz = Cv @ Z @ Co.T
    dz_sym = dz + dz.T
    dm0 = 2.0 * Co @ Co.T

    auxmol = hf_gpu.with_df.auxmol
    if auxmol is None:
        auxmol = df.addons.make_auxmol(
            mol, auxbasis=hf_gpu.with_df.auxbasis
        )

    # GPU4PySCF's DF TD-gradient kernel supports different density matrices on
    # the two sides of the derivative ERI.  Factors [4,4] reproduce exactly the
    # fourfold Z contraction of J - 1/2 K in make_h1.
    int3c2e_opt = df_tdrhf_grad.Int3c2eOpt(mol, auxmol).build()
    with _df_derivative_memory_limit(memory_gb):
        g_jk = df_tdrhf_grad._jk_energies_per_atom(
            int3c2e_opt, [[dz_sym, dm0]], j_factor=[4.0], k_factor=[4.0],
            sum_results=True,
        )

    g_zf1 = g_hcore.reshape(mol.natm, 3) + cp.asarray(g_jk)
    del dz, dz_sym, dm0, int3c2e_opt, g_jk
    return g_zf1


def _direct_contracted_z_f1_factorized(mol, hf_gpu, C, occ, Co, Cv, Z,
                                       *, memory_gb=None):
    """Return ``4 * Z:F1_vo`` while preserving known density factors.

    The dense density values are still available to the Coulomb path, but the
    factor tags prevent an unnecessary SVD before the exchange derivative.
    """

    nocc, nmo = Co.shape[1], C.shape[1]
    npert = 3 * mol.natm
    hcore1 = gpu_rhf_grad.get_grad_hcore(
        hf_gpu.nuc_grad_method(), mo_coeff=C, mo_occ=occ
    ).reshape(npert, nmo, nocc)
    g_hcore = 4.0 * _contract("ai,Aai->A", Z, hcore1[:, nocc:, :])
    del hcore1

    z_left = Cv @ Z
    dz_sym = _make_factorized_dm(z_left, Co, symmetrize=1)
    dm0 = _make_factorized_dm(Co * 2.0, Co, symmetrize=0)

    auxmol = hf_gpu.with_df.auxmol
    if auxmol is None:
        auxmol = df.addons.make_auxmol(
            mol, auxbasis=hf_gpu.with_df.auxbasis
        )
    int3c2e_opt = df_tdrhf_grad.Int3c2eOpt(mol, auxmol).build()
    with _df_derivative_memory_limit(memory_gb):
        g_jk = df_tdrhf_grad._jk_energies_per_atom(
            int3c2e_opt, [[dz_sym, dm0]], j_factor=[4.0], k_factor=[4.0],
            sum_results=True,
        )
    out = g_hcore.reshape(mol.natm, 3) + cp.asarray(g_jk)
    del z_left, dz_sym, dm0, int3c2e_opt, g_jk
    return out


def _build_s1_oo_and_z_se(S1_ao, Co, Cv, Z, e_occ, atom_slices):
    """Build only the overlap information consumed after the S1 transform.

    ``S1_oo`` is retained for the skeleton and adjoint contractions.  The
    virtual-occupied block is immediately reduced to ``Z:(S1*e_occ)`` and is
    never stored as a perturbation-indexed tensor.
    """

    npert = 3 * len(atom_slices)
    nocc = Co.shape[1]
    S1_oo = cp.empty((npert, nocc, nocc), dtype=Co.dtype)
    z_se = cp.empty(npert, dtype=Co.dtype)
    z_e = Z * e_occ[None, :]

    for ia, (p0, p1) in enumerate(atom_slices):
        co_atom = Co[p0:p1]
        cv_atom = Cv[p0:p1]
        for ic in range(3):
            ip = 3 * ia + ic
            s_rows = S1_ao[ic, p0:p1]
            s_co = s_rows @ Co
            s_oo = co_atom.T @ s_co + s_co.T @ co_atom
            s_cv = s_rows @ Cv
            s_vo = cv_atom.T @ s_co + s_cv.T @ co_atom
            S1_oo[ip] = s_oo
            z_se[ip] = _contract("ai,ai->", z_e, s_vo)

    del z_e
    return S1_oo, z_se


def _adjoint_z_ax(hf_resp, Z, Co, Cv, S1_oo):
    """Return ``Z:Ax(S1)_vo`` using a single HF response evaluation.

    With ``Dz = Cv Z Co.T`` and the self-adjoint RHF response V,

        Z : Cv.T V(dm_A) Co
          = 2 * S_oo[A] : Co.T V(Dz + Dz.T) Co.

    No perturbation-indexed AO density or AO response is constructed.
    """

    dz = Cv @ Z @ Co.T
    vz = hf_resp(dz + dz.T)
    vz_oo = Co.T @ vz @ Co
    z_ax = 2.0 * _contract("Aij,ij->A", S1_oo, vz_oo)
    del dz, vz, vz_oo
    return z_ax


def _adjoint_z_ax_mo_direct(dfobj, Z, Co, Cv, S1_oo,
                            *, direct_scf_tol=1.0e-14):
    """Return the adjoint overlap response without a full AO J/K output."""

    z_left = Cv @ Z
    dz_sym = z_left @ Co.T
    dz_sym = dz_sym + dz_sym.T
    dz_sym = tag_array(
        dz_sym[None], mo1=(0.5 * z_left)[None], occ_coeff=Co
    )
    vz_oo = _df_hf_response_mo(
        dfobj, dz_sym, Co, Co, direct_scf_tol=direct_scf_tol
    )[0]
    z_ax = 2.0 * _contract("Aij,ij->A", S1_oo, vz_oo)
    del z_left, dz_sym, vz_oo
    return z_ax


class DCGradientGPU:
    """Standalone S1-optimized DC gradient with adjoint contractions."""

    def __init__(self, mf, xc="PBE", cphf_tol=1e-10, cphf_max_cycle=500,
                 atom_grid=(99, 590), grid_response=True, checkpoint=None,
                 resume=False, scf_tol=None, response_backend="mo_direct",
                 share_df=True, factorized_z_f1=True,
                 df_deriv_memory_gb=None):
        self.mf_hf = mf
        self.mol = mf.mol
        self.xc = xc
        self.cphf_tol = cphf_tol
        self.cphf_max_cycle = cphf_max_cycle
        self.atom_grid = atom_grid
        self.grid_response = grid_response
        self.scf_tol = float(mf.conv_tol if scf_tol is None else scf_tol)
        self.checkpoint_path = checkpoint
        self.resume = bool(resume)
        response_backend = response_backend.lower().replace("-", "_")
        if response_backend not in ("mo_direct", "ao"):
            raise ValueError("response_backend must be 'mo_direct' or 'ao'")
        self.response_backend = response_backend
        self.share_df = bool(share_df)
        self.factorized_z_f1 = bool(factorized_z_f1)
        self.df_deriv_memory_gb = (
            None if df_deriv_memory_gb is None else float(df_deriv_memory_gb)
        )
        if self.df_deriv_memory_gb is not None and self.df_deriv_memory_gb <= 0:
            raise ValueError("df_deriv_memory_gb must be positive")
        if self.resume and self.checkpoint_path is None:
            raise ValueError("resume=True requires a checkpoint path")
        self.memory_diagnostics = {}
        self.optimization_diagnostics = {}
        self.checkpoint_diagnostics = {"loaded": [], "saved": []}
        self._checkpoint_store = None
        self._reference_restored = False

    def _get_checkpoint(self, auxbasis):
        if self.checkpoint_path is None:
            return None
        if self._checkpoint_store is None:
            signature, metadata = _checkpoint_signature(
                self.mol,
                auxbasis=auxbasis,
                xc=self.xc,
                atom_grid=self.atom_grid,
                grid_response=self.grid_response,
                scf_tol=self.scf_tol,
                cphf_tol=self.cphf_tol,
                response_backend=self.response_backend,
                share_df=self.share_df,
                factorized_z_f1=self.factorized_z_f1,
                df_deriv_memory_gb=self.df_deriv_memory_gb,
            )
            store = _CheckpointStore(
                self.checkpoint_path,
                signature,
                metadata,
                self.resume,
            )
            store.prepare()
            self._checkpoint_store = store
        return self._checkpoint_store

    def restore_reference(self):
        """Restore converged RHF orbitals from the configured checkpoint."""

        if not self.resume:
            raise ValueError("restore_reference requires resume=True")
        has_df = hasattr(self.mf_hf, "with_df") and self.mf_hf.with_df is not None
        auxbasis = self.mf_hf.with_df.auxbasis if has_df else None
        checkpoint = self._get_checkpoint(auxbasis)
        if checkpoint is None or not checkpoint.has("scf"):
            raise RuntimeError("Checkpoint does not contain a completed SCF stage")
        data = checkpoint.load_gpu(
            "scf", "mo_coeff", "mo_energy", "mo_occ", "e_tot"
        )
        self.mf_hf.mo_coeff = data["mo_coeff"]
        self.mf_hf.mo_energy = data["mo_energy"]
        self.mf_hf.mo_occ = data["mo_occ"]
        self.mf_hf.e_tot = float(data["e_tot"])
        self.mf_hf.converged = True
        self._reference_restored = True
        self.checkpoint_diagnostics["loaded"].append("scf")
        return self.mf_hf.e_tot

    def _save_checkpoint(self, checkpoint, stage, log, **values):
        if checkpoint is None:
            return
        t0 = time.time()
        checkpoint.save(stage, **values)
        self.checkpoint_diagnostics["saved"].append(stage)
        log.info(
            "  checkpoint saved: %s [%s] (%.2f s)",
            checkpoint.path,
            stage,
            time.time() - t0,
        )

    def _load_checkpoint_gpu(self, checkpoint, stage, log, *names):
        t0 = time.time()
        data = checkpoint.load_gpu(stage, *names)
        self.checkpoint_diagnostics["loaded"].append(stage)
        log.info(
            "  checkpoint restored: %s [%s] (%.2f s)",
            checkpoint.path,
            stage,
            time.time() - t0,
        )
        return data

    def kernel(self, verbose=None):
        """Evaluate the complete DC-DFT gradient."""

        mol = self.mol
        if verbose is None:
            verbose = mol.verbose
        log = logger.new_logger(mol, verbose)

        t0 = time.time()
        stage_times = {}
        self._nonvariational_stage_times = {}
        log.info("\n" + "=" * 70)
        log.info("DC-DFT Gradient - 100%% GPU (S1 Optimized, Adjoint)")
        log.info("=" * 70)

        has_df = hasattr(self.mf_hf, "with_df") and self.mf_hf.with_df is not None
        auxbasis = self.mf_hf.with_df.auxbasis if has_df else None
        shared_df = self.mf_hf.with_df if has_df and self.share_df else None
        checkpoint = self._get_checkpoint(auxbasis)
        if self.resume and not self._reference_restored:
            self.restore_reference()
        if not self.mf_hf.converged:
            raise RuntimeError("The reference RHF calculation is not converged")
        reference_e_tot = self.mf_hf.e_tot

        # Keep all MO data and subsequent contractions on the GPU.
        C = cp.asarray(self.mf_hf.mo_coeff)
        e = cp.asarray(self.mf_hf.mo_energy)
        occ = cp.asarray(self.mf_hf.mo_occ)

        if checkpoint is not None and not checkpoint.has("scf"):
            self._save_checkpoint(
                checkpoint,
                "scf",
                log,
                mo_coeff=C,
                mo_energy=e,
                mo_occ=occ,
                e_tot=np.asarray(reference_e_tot),
            )

        if self.resume and checkpoint is not None and checkpoint.has("gradient"):
            data = checkpoint.load_numpy("gradient", "value")
            self.checkpoint_diagnostics["loaded"].append("gradient")
            log.info("  checkpoint restored: %s [gradient]", checkpoint.path)
            return data["value"]

        # The optimized route lets KS and CPHF consume the converged reference
        # CDERI.  The comparison route preserves the older separate-DF layout.
        if has_df and shared_df is None:
            self.memory_diagnostics["released_reference_df_gib"] = (
                _release_df_storage(self.mf_hf.with_df, "reference SCF", log)
            )

        nao, nmo = C.shape
        nocc = int(cp.sum(occ > 0))
        nvir = nmo - nocc
        so = slice(0, nocc)
        sv = slice(nocc, nmo)
        Co = C[:, so]
        Cv = C[:, sv]
        Co2 = Co * 2.0

        ao_slices = mol.aoslice_by_atom()
        atom_slices = [ao_slices[ia, 2:4] for ia in range(mol.natm)]

        # The HF backend is cheap to reconstruct.  Its CDERI is built lazily
        # only if CPHF or the adjoint response must actually be recomputed.
        hf_gpu = gpu_hf.RHF(mol)
        if has_df:
            hf_gpu = hf_gpu.density_fit(auxbasis=auxbasis)
            if shared_df is not None:
                hf_gpu.with_df = shared_df
        hf_gpu.mo_coeff = C
        hf_gpu.mo_energy = e
        hf_gpu.mo_occ = occ
        hf_gpu.converged = True
        hf_gpu.e_tot = reference_e_tot
        hf_resp = None

        def can_restore(stage):
            return (
                self.resume
                and checkpoint is not None
                and checkpoint.has(stage)
            )

        if can_restore("skeleton"):
            for required in ("z_vector", "s1"):
                if not checkpoint.has(required):
                    raise RuntimeError(
                        f"Checkpoint stage 'skeleton' requires '{required}'"
                    )
            log.info("\n[1-4/5] Restoring completed gradient intermediates")
            Z = self._load_checkpoint_gpu(
                checkpoint, "z_vector", log, "Z"
            )["Z"]
            names = ("S1_oo", "z_se") if has_df else (
                "S1_oo", "z_se", "S1_mo_o"
            )
            s1_data = self._load_checkpoint_gpu(
                checkpoint, "s1", log, *names
            )
            S1_oo, z_se = s1_data["S1_oo"], s1_data["z_se"]
            S1_mo_o = s1_data.get("S1_mo_o")
            g_skel = self._load_checkpoint_gpu(
                checkpoint, "skeleton", log, "g_skel"
            )["g_skel"]
            stage_times["restored_intermediates"] = time.time() - t0
        else:
            dm_hf = 2.0 * Co @ Co.T

            # 1) KS Fock evaluated at the HF density.
            log.info("\n[1/5] KS Fock (GPU)")
            t1 = time.time()
            ks_gpu = gpu_rks.RKS(mol, xc=self.xc)
            if has_df:
                ks_gpu = ks_gpu.density_fit(auxbasis=auxbasis)
                if shared_df is not None:
                    ks_gpu.with_df = shared_df
            ks_gpu.grids.atom_grid = self.atom_grid

            if can_restore("ks_fock"):
                data = self._load_checkpoint_gpu(
                    checkpoint,
                    "ks_fock",
                    log,
                    "F_KS",
                    "F_KS_vo",
                    "F_KS_oo",
                )
                F_KS = data["F_KS"]
                F_KS_vo = data["F_KS_vo"]
                F_KS_oo = data["F_KS_oo"]
                if has_df and ks_gpu.with_df.auxmol is None:
                    ks_gpu.with_df.auxmol = df.addons.make_auxmol(
                        mol, auxbasis=auxbasis
                    )
            else:
                F_KS = cp.asarray(ks_gpu.get_fock(dm=dm_hf))
                F_KS_vo = Cv.T @ F_KS @ Co
                F_KS_oo = Co.T @ F_KS @ Co
                self._save_checkpoint(
                    checkpoint,
                    "ks_fock",
                    log,
                    F_KS=F_KS,
                    F_KS_vo=F_KS_vo,
                    F_KS_oo=F_KS_oo,
                )
            stage_times["ks_fock"] = time.time() - t1
            log.info("  %.2f s", stage_times["ks_fock"])

            # 2) Z-vector from GPU CPHF.
            log.info("[2/5] Z-vector (GPU CPHF)")
            t2 = time.time()
            if can_restore("z_vector"):
                Z = self._load_checkpoint_gpu(
                    checkpoint, "z_vector", log, "Z"
                )["Z"]
            else:
                if has_df and self.response_backend == "mo_direct":
                    Ax_vo = _make_mo_direct_ax(
                        hf_gpu.with_df, Cv, Co,
                        direct_scf_tol=hf_gpu.direct_scf_tol,
                    )
                else:
                    hf_resp = _gen_rhf_response_gpu(
                        hf_gpu, mo_coeff=C, mo_occ=occ, hermi=1
                    )

                    def Ax_vo(x_vo):
                        original_shape = x_vo.shape
                        x = cp.asarray(x_vo).reshape(-1, nvir, nocc)
                        mo1 = _contract("pa,kai->kpi", Cv, x)
                        dm_x = _contract("kpi,qi->kpq", mo1, Co2)
                        dm_x = dm_x + dm_x.transpose(0, 2, 1)
                        dm_x = tag_array(
                            dm_x, mo1=mo1, occ_coeff=Co, mo_occ=occ
                        )
                        v_dm = hf_resp(dm_x)
                        tmp = _contract("ap,kpq->kaq", Cv.T, v_dm)
                        out = _contract("kaq,qi->kai", tmp, Co)
                        del mo1, dm_x, v_dm, tmp
                        return out.reshape(original_shape)

                Z, converged = gpu_cphf.solve(
                    Ax_vo,
                    e,
                    occ,
                    F_KS_vo,
                    max_cycle=self.cphf_max_cycle,
                    tol=self.cphf_tol,
                )
                if converged is not None and not converged:
                    log.warn("GPU CPHF not converged")
                self._save_checkpoint(checkpoint, "z_vector", log, Z=Z)
            stage_times["z_vector"] = time.time() - t2
            log.info("  %.2f s", stage_times["z_vector"])

            # 3) Occupied-column-only overlap derivative in the MO basis.
            log.info("[3/5] S1 AO->MO (GPU, occ-only)")
            t3 = time.time()
            ks_gpu.mo_coeff = C
            ks_gpu.mo_occ = occ
            ks_gpu.mo_energy = e
            ks_gpu.converged = True

            h1e = cp.asarray(ks_gpu.get_hcore(mol))
            e_tot = float(cp.einsum("ij,ji->", dm_hf, h1e + F_KS) * 0.5)
            ks_gpu.e_tot = e_tot + ks_gpu.energy_nuc()
            ks_grad_gpu = ks_gpu.nuc_grad_method()
            ks_grad_gpu.grid_response = self.grid_response
            S1_ao = None

            if can_restore("s1"):
                names = ("S1_oo", "z_se") if has_df else (
                    "S1_oo", "z_se", "S1_mo_o"
                )
                s1_data = self._load_checkpoint_gpu(
                    checkpoint, "s1", log, *names
                )
                S1_oo, z_se = s1_data["S1_oo"], s1_data["z_se"]
                S1_mo_o = s1_data.get("S1_mo_o")
            else:
                S1_ao = cp.asarray(ks_grad_gpu.get_ovlp(mol))
                S1_oo, z_se = _build_s1_oo_and_z_se(
                    S1_ao, Co, Cv, Z, e[so], atom_slices
                )
                S1_mo_o = None
                save_values = {"S1_oo": S1_oo, "z_se": z_se}
                if not has_df:
                    npert = 3 * mol.natm
                    S1_mo_o = cp.empty(
                        (npert, nmo, nocc), dtype=C.dtype
                    )
                    for ia, (p0, p1) in enumerate(atom_slices):
                        for ic in range(3):
                            s1_block = cp.zeros((nao, nao), dtype=C.dtype)
                            s1_block[p0:p1] = S1_ao[ic, p0:p1]
                            s1_block = s1_block + s1_block.T
                            S1_mo_o[3 * ia + ic] = C.T @ s1_block @ Co
                    save_values["S1_mo_o"] = S1_mo_o
                self._save_checkpoint(checkpoint, "s1", log, **save_values)
            stage_times["s1_streaming"] = time.time() - t3
            log.info("  %.2f s", stage_times["s1_streaming"])

            # 4) Variational/skeleton contribution.
            log.info("[4/5] Skeleton (GPU)")
            t4 = time.time()
            g_skel = self._skeleton_gpu(
                mol,
                ks_grad_gpu,
                F_KS_oo,
                S1_oo,
                log,
                e[so],
            )
            self._save_checkpoint(
                checkpoint, "skeleton", log, g_skel=g_skel
            )
            stage_times["skeleton"] = time.time() - t4
            log.info("  %.2f s", stage_times["skeleton"])

            # The KS gradient is complete.  Its DF tensor and stage-local
            # arrays are not needed by the non-variational correction.
            ks_df = ks_gpu.with_df if has_df else None
            ks_df_is_separate = (
                ks_df is not None and ks_df is not hf_gpu.with_df
            )
            del ks_grad_gpu, ks_gpu, F_KS, F_KS_vo, F_KS_oo
            del dm_hf, h1e, S1_ao
            if ks_df_is_separate:
                self.memory_diagnostics["released_ks_df_gib"] = (
                    _release_df_storage(ks_df, "KS gradient", log)
                )
            del ks_df

        needs_ao_response = not has_df or self.response_backend == "ao"
        if hf_resp is None and needs_ao_response and not can_restore("z_ax"):
            hf_resp = _gen_rhf_response_gpu(
                hf_gpu, mo_coeff=C, mo_occ=occ, hermi=1
            )

        # 5) Non-variational contribution.
        log.info("[5/5] Non-variational (GPU)")
        t5 = time.time()
        g_nv = self._nonvariational_gpu(
            mol, hf_gpu, hf_resp, C, e, occ, so, sv, Co, Cv, Co2, Z,
            S1_oo, z_se, S1_mo_o, atom_slices, log, checkpoint=checkpoint,
            resume=self.resume,
        )
        stage_times["nonvariational"] = time.time() - t5
        log.info("  %.2f s", stage_times["nonvariational"])

        g_np = cp.asnumpy(g_skel + g_nv)
        self._save_checkpoint(
            checkpoint, "gradient", log, value=g_np
        )
        self.optimization_diagnostics = {
            "response_backend": self.response_backend,
            "shared_df_identity": bool(
                shared_df is not None and hf_gpu.with_df is shared_df
            ),
            "direct_z_f1_backend": (
                "factorized" if self.factorized_z_f1 else "dense"
            ),
            "df_deriv_memory_gb": self.df_deriv_memory_gb,
            "stage_times_seconds": {
                **stage_times,
                **self._nonvariational_stage_times,
            },
            "total_seconds": time.time() - t0,
        }
        log.info("\n" + "=" * 70)
        log.info(
            "Total: %.2f s | Max |grad|: %.6e",
            time.time() - t0,
            np.max(np.abs(g_np)),
        )
        log.info("=" * 70 + "\n")
        return g_np

    def _skeleton_gpu(self, mol, ks_grad_gpu, F_KS_oo, S1_oo, log, e_occ):
        """Return the KS skeleton gradient with the DC overlap correction."""

        log.debug("  ks_grad.kernel (GPU)...")
        g_ks = cp.asarray(ks_grad_gpu.kernel())
        log.debug("  overlap correction...")
        dF = F_KS_oo - cp.diag(e_occ)
        g_corr = -2.0 * _contract("Aij,ij->A", S1_oo, dF).reshape(
            mol.natm, 3
        )
        return g_ks + g_corr

    def _nonvariational_gpu(
        self, mol, hf_gpu, hf_resp, C, e, occ, so, sv, Co, Cv, Co2, Z,
        S1_oo, z_se, S1_mo_o, atom_slices, log, *, checkpoint=None,
        resume=False,
    ):
        # The direct derivative-energy backend below is a DF implementation.
        # Preserve the original behavior for a non-DF reference.
        if not hasattr(hf_gpu, "with_df") or hf_gpu.with_df is None:
            log.warn(
                "Adjoint/direct-ZF1 path requires density fitting; "
                "falling back to the full local non-variational implementation"
            )
            if S1_mo_o is None:
                raise RuntimeError("The non-DF fallback requires full S1 columns")
            return self._nonvariational_full_gpu(
                mol, hf_gpu, hf_resp, C, e, occ, so, sv, Co, Cv, Co2, Z,
                S1_mo_o, atom_slices, log,
            )

        natm = mol.natm

        def can_restore(stage):
            return resume and checkpoint is not None and checkpoint.has(stage)

        # ---------------------------------------------------------------
        # 1) Z:Ax(S1)_vo while the CPHF DF tensor is still available
        # ---------------------------------------------------------------
        if can_restore("z_ax"):
            z_ax = self._load_checkpoint_gpu(
                checkpoint, "z_ax", log, "value"
            )["value"]
        else:
            log.debug("  adjoint Z:Ax(S1)_vo (one HF response)...")
            t_ax = time.time()
            if self.response_backend == "mo_direct":
                z_ax = _adjoint_z_ax_mo_direct(
                    hf_gpu.with_df, Z, Co, Cv, S1_oo,
                    direct_scf_tol=hf_gpu.direct_scf_tol,
                )
            else:
                z_ax = _adjoint_z_ax(hf_resp, Z, Co, Cv, S1_oo)
            self._nonvariational_stage_times["adjoint_response"] = (
                time.time() - t_ax
            )
            log.debug(
                "  adjoint Z:Ax(S1)_vo: %.2f s",
                self._nonvariational_stage_times["adjoint_response"],
            )
            self._save_checkpoint(
                checkpoint, "z_ax", log, value=z_ax
            )

        # This is the final use of hf_resp and its CDERI.  Release it before
        # entering the high-water-mark direct DF derivative contraction.
        self.memory_diagnostics["released_cphf_df_gib"] = (
            _release_df_storage(hf_gpu.with_df, "CPHF", log)
        )

        # ---------------------------------------------------------------
        # 2) 4 * Z:F1_vo after releasing the earlier DF storage
        # ---------------------------------------------------------------
        if can_restore("g_zf1"):
            g_zf1 = self._load_checkpoint_gpu(
                checkpoint, "g_zf1", log, "value"
            )["value"]
        else:
            log.debug("  direct 4*Z:F1_vo (GPU DF derivative)...")
            t_f1 = time.time()
            direct_z_f1 = (
                _direct_contracted_z_f1_factorized
                if self.factorized_z_f1 else _direct_contracted_z_f1
            )
            g_zf1 = direct_z_f1(
                mol, hf_gpu, C, occ, Co, Cv, Z,
                memory_gb=self.df_deriv_memory_gb,
            )
            self._nonvariational_stage_times["direct_z_f1"] = (
                time.time() - t_f1
            )
            log.debug(
                "  direct 4*Z:F1_vo: %.2f s",
                self._nonvariational_stage_times["direct_z_f1"],
            )
            self._save_checkpoint(
                checkpoint, "g_zf1", log, value=g_zf1
            )

        # Original expression:
        #   4 Z:(F1 - Se - 0.5 AxS)
        g_nv = g_zf1.reshape(-1) - 4.0 * z_se - 2.0 * z_ax
        del g_zf1, z_se, z_ax
        return g_nv.reshape(natm, 3)

    def _nonvariational_full_gpu(
        self, mol, hf_gpu, hf_resp, C, e, occ, so, sv, Co, Cv, Co2, Z,
        S1_mo_o, atom_slices, log,
    ):
        """Non-DF fallback using the full derivative-Fock construction."""

        natm = mol.natm
        npert = 3 * natm
        nocc = Co.shape[1]

        log.debug("  make_h1 (GPU fallback)...")
        t_h1 = time.time()
        hf_hess = hf_gpu.Hessian()
        F_HF_1 = hf_hess.make_h1(C, occ).reshape(npert, -1, nocc)
        log.debug(
            "  make_h1: %.2f s, shape=%s",
            time.time() - t_h1,
            F_HF_1.shape,
        )

        log.debug("  Ax(S1) (GPU fallback)...")
        t_ax = time.time()
        S_oo = S1_mo_o[:, so, :]
        mo1_s = _contract("pi,Aij->Apj", Co, S_oo)
        dm_s = _contract("Apj,qj->Apq", mo1_s, Co2)
        dm_s = dm_s + dm_s.transpose(0, 2, 1)
        dm_s = tag_array(
            dm_s, mo1=mo1_s, occ_coeff=Co, mo_occ=occ
        )
        v_resp = hf_resp(dm_s)
        tmp = _contract("up,Apq->Auq", C.T, v_resp)
        AxS = _contract("Auq,qi->Aui", tmp, Co)
        del mo1_s, dm_s, v_resp, tmp
        log.debug("  Ax(S1): %.2f s", time.time() - t_ax)

        Se = S1_mo_o * e[so][None, None, :]
        B_vo = (F_HF_1 - Se - 0.5 * AxS)[:, sv, :]
        g_nv = 4.0 * _contract("ai,Aai->A", Z, B_vo)
        del F_HF_1, Se, AxS, B_vo
        return g_nv.reshape(natm, 3)


def kernel(mf, xc="PBE", verbose=None, **kwargs):
    """Compute the adjoint/direct-ZF1 DC-DFT gradient on the GPU."""

    return DCGradientGPU(mf, xc=xc, **kwargs).kernel(verbose=verbose)


def _read_xyz(path: str | Path) -> str:
    """Return a PySCF atom string from a standard XYZ or coordinate file."""

    xyz_path = Path(path).expanduser().resolve()
    if not xyz_path.is_file():
        raise FileNotFoundError(f"XYZ file not found: {xyz_path}")

    lines = xyz_path.read_text().splitlines()
    try:
        natm = int(lines[0].strip())
    except (ValueError, IndexError):
        atoms = [line for line in lines if line.strip()]
    else:
        atoms = [line for line in lines[2 : 2 + natm] if line.strip()]
        if len(atoms) != natm:
            raise ValueError(
                f"Malformed XYZ file {xyz_path}: expected {natm} atoms, "
                f"found {len(atoms)}"
            )
    return "\n".join(atoms)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a density-fitted RHF calculation and the adjoint/direct-ZF1 "
            "DC-DFT gradient for an XYZ file."
        )
    )
    parser.add_argument("xyz", type=Path, help="Input XYZ file")
    parser.add_argument("--basis", default="def2-svpd")
    parser.add_argument(
        "--auxbasis",
        default="auto",
        help="DF auxiliary basis; 'auto' uses GPU4PySCF/PySCF selection",
    )
    parser.add_argument("--xc", default="PBE")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--spin", type=int, default=0)
    parser.add_argument("--unit", default="Angstrom")
    parser.add_argument("--max-memory", type=int, default=128000, help="MB")
    parser.add_argument("--verbose", type=int, default=4)
    parser.add_argument("--scf-tol", type=float, default=1e-8)
    parser.add_argument("--scf-max-cycle", type=int, default=100)
    parser.add_argument("--cphf-tol", type=float, default=1e-10)
    parser.add_argument("--cphf-max-cycle", type=int, default=500)
    parser.add_argument(
        "--response-backend", choices=("mo-direct", "ao"),
        default="mo-direct", help="CPHF/adjoint response representation",
    )
    parser.add_argument(
        "--separate-df", action="store_true",
        help="Use separate KS and CPHF DF objects for regression comparison",
    )
    parser.add_argument(
        "--dense-z-f1", action="store_true",
        help="Discard known density factors before direct Z:F1 (comparison)",
    )
    parser.add_argument(
        "--df-deriv-memory-gb", type=float,
        help="Cap memory reported to the direct DF derivative kernel",
    )
    parser.add_argument(
        "--atom-grid",
        type=int,
        nargs=2,
        metavar=("NRAD", "NANG"),
        default=(99, 590),
    )
    parser.add_argument(
        "--no-grid-response",
        action="store_true",
        help="Disable DFT integration-grid response in the gradient",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional text file for the Natm x 3 gradient in a.u.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "HDF5 checkpoint file for SCF and gradient-stage intermediates"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume completed stages from --checkpoint",
    )
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    atom = _read_xyz(args.xyz)
    mol = gto.M(
        atom=atom,
        basis=args.basis,
        charge=args.charge,
        spin=args.spin,
        unit=args.unit,
        verbose=args.verbose,
        max_memory=args.max_memory,
    )

    auxbasis = None if args.auxbasis.lower() == "auto" else args.auxbasis
    mf = gpu_hf.RHF(mol).density_fit(auxbasis=auxbasis)
    mf.conv_tol = args.scf_tol
    mf.max_cycle = args.scf_max_cycle
    driver = DCGradientGPU(
        mf,
        xc=args.xc,
        cphf_tol=args.cphf_tol,
        cphf_max_cycle=args.cphf_max_cycle,
        atom_grid=tuple(args.atom_grid),
        grid_response=not args.no_grid_response,
        checkpoint=args.checkpoint,
        resume=args.resume,
        scf_tol=args.scf_tol,
        response_backend=args.response_backend,
        share_df=not args.separate_df,
        factorized_z_f1=not args.dense_z_f1,
        df_deriv_memory_gb=args.df_deriv_memory_gb,
    )

    if args.resume:
        energy = driver.restore_reference()
        print(f"Restored converged DF-RHF reference from: {args.checkpoint}")
    else:
        energy = mf.kernel()
        if not mf.converged:
            raise RuntimeError(
                f"DF-RHF did not converge in {args.scf_max_cycle} cycles"
            )

    gradient = driver.kernel(verbose=args.verbose)

    print(f"\nConverged DF-RHF energy: {energy:.15f} Eh")
    print(f"DC-{args.xc} gradient (Eh/Bohr):")
    print(np.array2string(gradient, precision=12, suppress_small=False))

    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            output,
            gradient,
            fmt="%.16e",
            header=f"DC-{args.xc} gradient (Eh/Bohr); source={args.xyz}",
        )
        print(f"Gradient written to: {output}")
    return 0


__all__ = ["DCGradientGPU", "kernel"]


if __name__ == "__main__":
    raise SystemExit(main())
