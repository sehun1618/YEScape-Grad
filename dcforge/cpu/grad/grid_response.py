"""CPU molecular moving-grid response, selectable without global patches.

The native kernel evaluates p_A / sum_B p_B and its nuclear derivative in
two atom-pair passes per grid point. It avoids PySCF's atom-pair-by-grid
derivative tensor; XC/AO integration and DF J/K/auxiliary derivatives remain
upstream. Becke, J. Chem. Phys. 88, 2547 (1988), doi:10.1063/1.454033.
Equations follow pyscf.grad.rks.grids_response_becke (PySCF Apache-2.0).
"""
import ctypes
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
from pyscf import lib
from pyscf.dft import gen_grid, radi
from pyscf.grad import rks, uks

from dcforge.tools.cphf import _bind_globals


@lru_cache(maxsize=1)
def native():
    """Load the optional build-time extension; never compile during a run."""
    try:
        from dcforge.cpu.grad import _grid_response_becke
    except ImportError as err:
        raise RuntimeError(
            'becke_omp extension unavailable; install/build DC-Forge with '
            'OpenMP support or select pyscf_default') from err
    handle = ctypes.CDLL(_grid_response_becke.__file__)
    arr = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
    handle.partition.argtypes = [ctypes.c_int]*4 + [arr]*7
    handle.partition.restype = ctypes.c_int
    return handle


def parameters(grids):
    """Resolve PySCF's original-Becke radii correction and inverse distances."""
    if grids.becke_scheme is not gen_grid.original_becke:
        raise NotImplementedError('becke_omp supports original_becke only')
    if grids.radii_adjust not in (None, radi.treutler_atomic_radii_adjust,
                                 radi.becke_atomic_radii_adjust):
        raise NotImplementedError('Unsupported radii adjustment')
    atoms = np.ascontiguousarray(grids.mol.atom_coords(), dtype=np.float64)
    distance = np.linalg.norm(atoms[:, None, :]-atoms[None, :, :], axis=2)
    offdiag = ~np.eye(len(atoms), dtype=bool)
    if not np.isfinite(atoms).all() or np.any(distance[offdiag] < 1e-12):
        raise ValueError('Invalid or coincident nuclear coordinates')
    inv = np.zeros_like(distance)
    inv[offdiag] = 1/distance[offdiag]
    adjust = np.zeros_like(distance)
    if grids.radii_adjust is not None:
        rad = grids.atomic_radii[grids.mol.atom_charges()]
        if grids.radii_adjust is radi.treutler_atomic_radii_adjust:
            rad = np.sqrt(rad)
        rad = rad+1e-200
        rr = rad[:, None]*(1/rad)
        adjust = np.clip(.25*(rr.T-rr), -.5, .5)
    return atoms, np.ascontiguousarray(adjust), np.ascontiguousarray(inv)


def partition(coords, vol, owner, params, threads):
    """Return weights (ngrid,) and nuclear derivatives (natm, 3, ngrid).

    coords and vol are the owner atom's full grid and radial/angular weights.
    params is the output of parameters(); all native arrays use FP64.
    """
    atoms, adjust, inv = params
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    vol = np.ascontiguousarray(vol, dtype=np.float64)
    n, ng = len(atoms), len(coords)
    # Validate the native boundary even when partition() is called directly.
    for array, shape in ((atoms, (n, 3)), (adjust, (n, n)), (inv, (n, n))):
        if (array.shape != shape or array.dtype != np.float64 or
                not array.flags.c_contiguous or not np.isfinite(array).all()):
            raise ValueError('Invalid partition parameters')
    if (coords.shape != (ng, 3) or vol.shape != (ng,) or
            not 0 <= owner < n or not 1 <= threads <= 1024):
        raise ValueError('Invalid coordinates, weights, owner or thread count')
    if (ng > np.iinfo(np.int32).max or not np.isfinite(coords).all() or
            not np.isfinite(vol).all()):
        raise ValueError('Invalid grid')
    w0 = np.empty(ng)
    w1 = np.empty((n, 3, ng))
    rc = native().partition(n, ng, owner, threads, atoms.ravel(), coords.ravel(),
                            vol, adjust.ravel(), inv.ravel(), w0, w1.ravel())
    if rc:
        raise MemoryError('Native scratch allocation failed')
    if not np.isfinite(w0).all() or not np.isfinite(w1).all():
        raise FloatingPointError('Nonfinite partition result')
    return w0, w1


def grids_response_becke(grids, threads=None):
    """Yield PySCF-compatible atom grids, preserving pruning and atom order.

    Kwargs:
        threads : int or None
            None uses pyscf.lib.num_threads(). Intended for component tests;
            gradient drivers always use the PySCF thread setting.
    """
    params = parameters(grids)
    tab = grids.gen_atomic_grids(grids.mol, grids.atom_grid, grids.radi_method,
                                grids.level, grids.prune)
    threads = lib.num_threads() if threads is None else threads
    for owner, center in enumerate(params[0]):
        relative, vol = tab[grids.mol.atom_symbol(owner)]
        coords = relative+center
        w0, w1 = partition(coords, vol, owner, params, threads)
        yield coords, w0, w1


def get_veff(mf_grad, upstream, mol=None, dm=None):
    """Evaluate the original provider with a call-local grid derivative.

    Args:
        mf_grad: DC-Forge molecular gradient object.
        upstream: Its original PySCF conventional or DF get_veff function.
        mol, dm: Unchanged PySCF arguments.

    The explicit provider matters: DF must retain its .aux tag and J/K
    derivatives. Only private function-global dictionaries are rebound.
    No module is modified, including on exceptions or nested calls.
    """
    backend = mf_grad.grid_response_backend
    if backend not in ('pyscf_default', 'becke_omp'):
        raise ValueError("grid_response_backend must be 'pyscf_default' or 'becke_omp'")
    active = mf_grad.grid_response and mf_grad.base._numint._xc_type(mf_grad.base.xc) != 'HF'
    mf_grad.grid_response_backend_used = backend if active else 'inactive'
    lib.logger.info(mf_grad, 'grid_response_backend = %s (used: %s, threads: %d)',
                    backend, mf_grad.grid_response_backend_used, lib.num_threads())
    if backend == 'pyscf_default' or not active:
        return upstream(mf_grad, mol, dm)

    native()
    rks_names = dict(vars(rks), grids_response_cc=grids_response_becke)
    rks_full = _bind_globals(rks.get_vxc_full_response, rks_names)
    rks_proxy = SimpleNamespace(**dict(rks_names, get_vxc_full_response=rks_full))
    uks_names = dict(vars(uks), rks_grad=rks_proxy)
    uks_full = _bind_globals(uks.get_vxc_full_response, uks_names)
    uks_proxy = SimpleNamespace(**dict(uks_names, get_vxc_full_response=uks_full))
    names = dict(upstream.__globals__, rks_grad=rks_proxy, uks_grad=uks_proxy,
                 get_vxc_full_response=(uks_full if upstream.__module__.endswith('uks')
                                        else rks_full))
    return _bind_globals(upstream, names)(mf_grad, mol, dm)
