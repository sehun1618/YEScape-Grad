# Copyright 2014-2018,2021 The PySCF Developers. All Rights Reserved.
# Modifications Copyright 2026 Sehun.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from PySCF's periodic Fourier AO kernel loader for separate AO-center
# derivatives required by the DC-DFT GDF response implementation.

"""Python loader for the package-owned Fourier AO derivative callbacks."""

from __future__ import annotations

import ctypes
import importlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df.incore import libpbc
from pyscf.pbc.lib.kpts_helper import is_zero


def load_extension() -> ctypes.CDLL:
    """Load PySCF dependencies globally, then return this extension's CDLL."""

    pyscf_lib = Path(lib.__file__).resolve().parent
    for filename in ("libcgto.so", "libpbc.so"):
        ctypes.CDLL(str(pyscf_lib / filename), mode=ctypes.RTLD_GLOBAL)
    module = importlib.import_module("pyscf_pbc_dcdft.df.gdf._ft_ip_ovlp")
    return ctypes.CDLL(str(Path(module.__file__).resolve()), mode=ctypes.RTLD_GLOBAL)


def gen_ft_center_derivative_kernel(
    supmol,
    *,
    center: str,
    aosym: str = "s1",
    return_complex: bool = False,
    kpts=None,
    verbose=None,
) -> Callable:
    """Return a PySCF PBC Fourier kernel differentiated on AO center i or j.

    The call signature intentionally matches ``ft_ao._ExtendedMole`` kernels.
    Output has a Cartesian component axis of length three.
    """

    if center not in {"i", "j"}:
        raise ValueError("center must be 'i' or 'j'")
    log = logger.new_logger(supmol, verbose)
    rs_cell = supmol.rs_cell
    if not isinstance(rs_cell, ft_ao._RangeSeparatedCell):
        raise TypeError("supmol.rs_cell must be a PySCF _RangeSeparatedCell")

    extension = load_extension()
    nbasp = rs_cell.ref_cell.nbas
    cell0_ao_loc = rs_cell.ref_cell.ao_loc
    bvk_ncells, _rs_nbas, nimgs = supmol.bas_mask.shape
    overlap_mask = supmol.get_ovlp_mask()
    bvk_overlap_mask = lib.condense(
        "np.any", overlap_mask, rs_cell.sh_loc, supmol.sh_loc
    )
    cell0_overlap_mask = (
        bvk_overlap_mask.reshape(nbasp, bvk_ncells, nbasp).any(axis=1)
    )
    overlap_mask = overlap_mask.astype(np.int8)
    cell0_overlap_mask = cell0_overlap_mask.astype(np.int8)

    stored_phase = None
    if kpts is not None:
        phase = np.exp(1j * supmol.bvkmesh_Ls.dot(np.asarray(kpts).T))
        stored_phase = (
            np.asarray(phase.real, order="C"),
            np.asarray(phase.imag, order="C"),
        )

    reciprocal = rs_cell.reciprocal_vectors()
    orthogonal = abs(reciprocal - np.diag(reciprocal.diagonal())).sum() < 1e-8
    default_eval_gz = "GTO_Gv_orth" if orthogonal else "GTO_Gv_nonorth"
    driver = libpbc.PBC_ft_bvk_drv
    intor_name = rs_cell._add_suffix(f"GTO_ft_ipovlp_{center}")
    callback = getattr(extension, intor_name)

    def kernel(
        Gv,
        gxyz=None,
        Gvbase=None,
        q=np.zeros(3),
        kptjs=None,
        shls_slice=None,
        aosym=aosym,
        out=None,
    ):
        q_vector = np.asarray(q)
        if q_vector.shape != (3,):
            raise ValueError(f"q must have shape (3,), got {q_vector.shape}")
        phase_r, phase_i = _phase_arrays(
            supmol, nimgs, stored_phase, kptjs
        )
        nkpts = phase_r.shape[1]
        gv_transposed = np.asarray(np.asarray(Gv).T + q_vector[:, None], order="C")
        ngrids = gv_transposed.shape[1]
        if shls_slice is None:
            shls_slice = (0, nbasp, 0, nbasp)
        ni = cell0_ao_loc[shls_slice[1]] - cell0_ao_loc[shls_slice[0]]
        nj = cell0_ao_loc[shls_slice[3]] - cell0_ao_loc[shls_slice[2]]
        shape = (nkpts, 3, ni, nj, ngrids)
        aosym2 = aosym[:2]
        if aosym2 == "s1hermi":
            if not (is_zero(q_vector) and is_zero(kptjs) and ni == nj):
                raise ValueError("s1hermi requires Gamma and square AO slices")
        elif aosym2 == "s2":
            i0 = cell0_ao_loc[shls_slice[0]]
            i1 = cell0_ao_loc[shls_slice[1]]
            packed = i1 * (i1 + 1) // 2 - i0 * (i0 + 1) // 2
            shape = (nkpts, 3, packed, ngrids)

        pointers = _grid_pointers(
            rs_cell,
            reciprocal,
            q_vector,
            gxyz,
            Gvbase,
            default_eval_gz,
        )
        eval_gz_name, p_gxyz, p_mesh, p_b, grid_owners = pointers
        eval_gz = getattr(libpbc, eval_gz_name)
        fill = getattr(
            libpbc, ("PBC_ft_bvk_nk1" if nkpts == 1 else "PBC_ft_bvk_k") + aosym2
        )
        if return_complex:
            sorter = getattr(libpbc, "PBC_ft_zsort_" + aosym2)
            output = np.ndarray(shape, dtype=np.complex128, buffer=out)
        else:
            sorter = getattr(libpbc, "PBC_ft_dsort_" + aosym2)
            output = np.ndarray((2,) + shape, buffer=out)
        if ngrids:
            driver(
                callback,
                eval_gz,
                fill,
                sorter,
                output.ctypes.data_as(ctypes.c_void_p),
                phase_r.ctypes.data_as(ctypes.c_void_p),
                phase_i.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(bvk_ncells),
                ctypes.c_int(nimgs),
                ctypes.c_int(nkpts),
                ctypes.c_int(nbasp),
                ctypes.c_int(3),
                supmol.seg_loc.ctypes.data_as(ctypes.c_void_p),
                supmol.seg2sh.ctypes.data_as(ctypes.c_void_p),
                cell0_ao_loc.ctypes.data_as(ctypes.c_void_p),
                (ctypes.c_int * 4)(*shls_slice),
                overlap_mask.ctypes.data_as(ctypes.c_void_p),
                cell0_overlap_mask.ctypes.data_as(ctypes.c_void_p),
                gv_transposed.ctypes.data_as(ctypes.c_void_p),
                p_b,
                p_gxyz,
                p_mesh,
                ctypes.c_int(ngrids),
                supmol._atm.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(supmol.natm),
                supmol._bas.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(supmol.nbas),
                supmol._env.ctypes.data_as(ctypes.c_void_p),
            )
            # ``grid_owners`` keeps the NumPy buffers behind ``p_gxyz`` and
            # ``p_b`` alive until the C driver has consumed both pointers.
            # Merely returning ``ctypes.c_void_p`` objects is insufficient:
            # those pointers do not retain ownership of their source arrays.
            del grid_owners
            log.debug("evaluated Fourier AO derivative on center %s", center)
        return np.rollaxis(output, -1, 2 if return_complex else 3)

    return kernel


def _phase_arrays(supmol, nimgs, stored_phase, kptjs):
    if kptjs is None:
        if stored_phase is None:
            return np.ones((nimgs, 1)), np.zeros((nimgs, 1))
        return stored_phase
    kpoint_array = np.asarray(kptjs, order="C").reshape(-1, 3)
    phase = np.exp(1j * supmol.bvkmesh_Ls.dot(kpoint_array.T))
    return np.asarray(phase.real, order="C"), np.asarray(phase.imag, order="C")


def _grid_pointers(rs_cell, reciprocal, q, gxyz, gvbase, default_eval_gz):
    if gxyz is None or gvbase is None or abs(q).sum() > 1e-9:
        return (
            "GTO_Gv_general",
            lib.c_null_ptr(),
            (ctypes.c_int * 3)(0, 0, 0),
            (ctypes.c_double * 1)(0),
            (),
        )
    gxyz_t = np.asarray(np.asarray(gxyz).T, order="C", dtype=np.int32)
    bqgv = np.hstack((reciprocal.ravel(), q) + tuple(gvbase))
    return (
        default_eval_gz,
        gxyz_t.ctypes.data_as(ctypes.c_void_p),
        (ctypes.c_int * 3)(*[len(values) for values in gvbase]),
        bqgv.ctypes.data_as(ctypes.c_void_p),
        (gxyz_t, bqgv),
    )
