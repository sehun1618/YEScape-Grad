'''Fourier AO-pair derivatives resolved by nuclear center.'''

import ctypes
import importlib
from pathlib import Path

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.df import ft_ao
from pyscf.pbc.df.incore import libpbc
from pyscf.pbc.lib.kpts_helper import is_zero


def load_extension():
    '''Return the package-owned Fourier AO derivative library.

    PySCF's periodic Fourier driver and Gaussian recurrences live in
    ``libpbc`` and ``libcgto``.  Loading both with ``RTLD_GLOBAL`` leaves the
    small DC-Forge callback extension independent of an environment-specific
    runtime search path.

    Returns:
        ctypes.CDLL
            Library exporting the Cartesian and spherical i/j-center
            derivative callbacks used by ``PBC_ft_bvk_drv``.
    '''
    pyscf_lib = Path(lib.__file__).resolve().parent
    for filename in ('libcgto.so', 'libpbc.so'):
        ctypes.CDLL(str(pyscf_lib / filename), mode=ctypes.RTLD_GLOBAL)
    module = importlib.import_module(
        'dcforge.cpu.pbc.df.gaussian._ft_ip_ovlp')
    return ctypes.CDLL(
        str(Path(module.__file__).resolve()), mode=ctypes.RTLD_GLOBAL)


def gen_ft_center_derivative_kernel(
        supmol, center, aosym='s1', return_complex=False, kpts=None,
        verbose=None):
    r'''Return a Fourier AO-pair kernel differentiated on center i or j.

    For the AO pair transform

    .. math::

        F_{\mu\nu}(\mathbf G)
        = \int \chi_\mu(\mathbf r)\chi_\nu(\mathbf r)
          e^{-i\mathbf G\cdot\mathbf r}\,d\mathbf r,

    ``center='i'`` returns the electron-coordinate derivative acting on
    :math:`\chi_\mu`, while ``center='j'`` acts on :math:`\chi_\nu`.
    A nuclear-center derivative later enters with the opposite sign.  Keeping
    the centers separate is required to assemble each atom's contribution to
    the GDF three-center source :math:`B^\lambda`.

    Args:
        supmol : pyscf.pbc.df.ft_ao._ExtendedMole
            BvK Fourier supermolecule constructed by PySCF.
        center : {'i', 'j'}
            AO center on which the derivative recurrence acts.
        aosym : str
            PySCF AO-pair storage convention.  The production GDF path uses
            ``'s1'``.
        return_complex : bool
            If true, return a complex array.  Otherwise preserve PySCF's
            separate real/imaginary leading axis.
        kpts : ndarray, optional
            Stored k-points used to precompute BvK phases.
        verbose : int or logger.Logger, optional
            PySCF verbosity setting.

    Returns:
        callable
            Function with the same grid argument contract as PySCF's
            ``_ExtendedMole.gen_ft_kernel`` and one component axis of length
            three.

    Notes:
        Grid construction, lattice-image phases, AO sorting, screening, and
        output layout remain owned by PySCF's ``PBC_ft_bvk_drv``.  DC-Forge
        only supplies the missing i/j-center callback recurrence.
    '''
    if center not in ('i', 'j'):
        raise ValueError("center must be 'i' or 'j'")

    log = logger.new_logger(supmol, verbose)
    rs_cell = supmol.rs_cell
    if not isinstance(rs_cell, ft_ao._RangeSeparatedCell):
        raise TypeError('supmol.rs_cell must be a PySCF _RangeSeparatedCell')

    extension = load_extension()
    nbasp = rs_cell.ref_cell.nbas
    cell0_ao_loc = rs_cell.ref_cell.ao_loc
    bvk_ncells, _rs_nbas, nimgs = supmol.bas_mask.shape
    overlap_mask = supmol.get_ovlp_mask()
    bvk_overlap_mask = lib.condense(
        'np.any', overlap_mask, rs_cell.sh_loc, supmol.sh_loc)
    cell0_overlap_mask = bvk_overlap_mask.reshape(
        nbasp, bvk_ncells, nbasp).any(axis=1)
    overlap_mask = overlap_mask.astype(np.int8)
    cell0_overlap_mask = cell0_overlap_mask.astype(np.int8)

    stored_phase = None
    if kpts is not None:
        phase = np.exp(1j * supmol.bvkmesh_Ls.dot(np.asarray(kpts).T))
        stored_phase = (
            np.asarray(phase.real, order='C'),
            np.asarray(phase.imag, order='C'),
        )

    reciprocal = rs_cell.reciprocal_vectors()
    orthogonal = abs(
        reciprocal - np.diag(reciprocal.diagonal())).sum() < 1e-8
    default_eval_gz = 'GTO_Gv_orth' if orthogonal else 'GTO_Gv_nonorth'
    driver = libpbc.PBC_ft_bvk_drv
    callback = getattr(
        extension, rs_cell._add_suffix(f'GTO_ft_ipovlp_{center}'))

    def kernel(
            Gv, gxyz=None, Gvbase=None, q=None, kptjs=None,
            shls_slice=None, aosym=aosym, out=None):
        '''Evaluate the selected center derivative on reciprocal vectors.'''
        q = np.zeros(3) if q is None else np.asarray(q)
        if q.shape != (3,):
            raise ValueError(f'q must have shape (3,), got {q.shape}')
        phase_r, phase_i = _phase_arrays(supmol, nimgs, stored_phase, kptjs)
        nkpts = phase_r.shape[1]
        GvT = np.asarray(np.asarray(Gv).T + q[:, None], order='C')
        ngrids = GvT.shape[1]

        if shls_slice is None:
            shls_slice = (0, nbasp, 0, nbasp)
        ni = cell0_ao_loc[shls_slice[1]] - cell0_ao_loc[shls_slice[0]]
        nj = cell0_ao_loc[shls_slice[3]] - cell0_ao_loc[shls_slice[2]]
        shape = (nkpts, 3, ni, nj, ngrids)
        aosym2 = aosym[:2]
        if aosym2 == 's1hermi':
            if not (is_zero(q) and is_zero(kptjs) and ni == nj):
                raise ValueError(
                    's1hermi requires Gamma and square AO slices')
        elif aosym2 == 's2':
            i0 = cell0_ao_loc[shls_slice[0]]
            i1 = cell0_ao_loc[shls_slice[1]]
            nao_pair = i1 * (i1 + 1) // 2 - i0 * (i0 + 1) // 2
            shape = (nkpts, 3, nao_pair, ngrids)

        pointers = _grid_pointers(
            rs_cell, reciprocal, q, gxyz, Gvbase, default_eval_gz)
        eval_gz_name, p_gxyz, p_mesh, p_b, grid_owners = pointers
        eval_gz = getattr(libpbc, eval_gz_name)
        fill = getattr(
            libpbc, ('PBC_ft_bvk_nk1' if nkpts == 1
                     else 'PBC_ft_bvk_k') + aosym2)
        if return_complex:
            sorter = getattr(libpbc, 'PBC_ft_zsort_' + aosym2)
            output = np.ndarray(shape, dtype=np.complex128, buffer=out)
        else:
            sorter = getattr(libpbc, 'PBC_ft_dsort_' + aosym2)
            output = np.ndarray((2,) + shape, buffer=out)

        if ngrids:
            driver(
                callback, eval_gz, fill, sorter,
                output.ctypes.data_as(ctypes.c_void_p),
                phase_r.ctypes.data_as(ctypes.c_void_p),
                phase_i.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(bvk_ncells), ctypes.c_int(nimgs),
                ctypes.c_int(nkpts), ctypes.c_int(nbasp), ctypes.c_int(3),
                supmol.seg_loc.ctypes.data_as(ctypes.c_void_p),
                supmol.seg2sh.ctypes.data_as(ctypes.c_void_p),
                cell0_ao_loc.ctypes.data_as(ctypes.c_void_p),
                (ctypes.c_int * 4)(*shls_slice),
                overlap_mask.ctypes.data_as(ctypes.c_void_p),
                cell0_overlap_mask.ctypes.data_as(ctypes.c_void_p),
                GvT.ctypes.data_as(ctypes.c_void_p), p_b, p_gxyz, p_mesh,
                ctypes.c_int(ngrids),
                supmol._atm.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(supmol.natm),
                supmol._bas.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(supmol.nbas),
                supmol._env.ctypes.data_as(ctypes.c_void_p),
            )
            # ctypes pointers do not retain the NumPy buffers they address.
            # Keep both grid buffers alive until the C driver returns.
            del grid_owners
            log.debug('evaluated Fourier AO derivative on center %s', center)
        return np.rollaxis(output, -1, 2 if return_complex else 3)

    return kernel


def _phase_arrays(supmol, nimgs, stored_phase, kptjs):
    if kptjs is None:
        if stored_phase is None:
            return np.ones((nimgs, 1)), np.zeros((nimgs, 1))
        return stored_phase
    kpts = np.asarray(kptjs, order='C').reshape(-1, 3)
    phase = np.exp(1j * supmol.bvkmesh_Ls.dot(kpts.T))
    return np.asarray(phase.real, order='C'), np.asarray(
        phase.imag, order='C')


def _grid_pointers(rs_cell, reciprocal, q, gxyz, Gvbase, default_eval_gz):
    if gxyz is None or Gvbase is None or abs(q).sum() > 1e-9:
        return (
            'GTO_Gv_general', lib.c_null_ptr(),
            (ctypes.c_int * 3)(0, 0, 0),
            (ctypes.c_double * 1)(0), (),
        )
    gxyzT = np.asarray(np.asarray(gxyz).T, order='C', dtype=np.int32)
    bqGv = np.hstack((reciprocal.ravel(), q) + tuple(Gvbase))
    return (
        default_eval_gz,
        gxyzT.ctypes.data_as(ctypes.c_void_p),
        (ctypes.c_int * 3)(*[len(values) for values in Gvbase]),
        bqGv.ctypes.data_as(ctypes.c_void_p),
        (gxyzT, bqGv),
    )
