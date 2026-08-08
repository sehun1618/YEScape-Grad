'''Shared GPU4PySCF sampled-k Gaussian density-response contraction.'''

from contextlib import contextmanager
from threading import RLock
from time import perf_counter

import cupy as cp
import numpy as np
from gpu4pyscf.pbc.df.grad import krhf as gpu_gdf_grad
from gpu4pyscf.pbc.df.grad.krhf import _jk_energy_per_atom
from gpu4pyscf.pbc.df.int3c2e import SRInt3c2eOpt

from dcforge.cpu.pbc.kpts import check_kpts_equal
from dcforge.gpu.pbc._compat import contiguous_int1e_hermitian

_RESPONSE_LAYOUT_LOCK = RLock()


@contextmanager
def _contiguous_complex_view(provider):
    '''Localize the GPU4PySCF 1.8 sampled-k complex-view workaround.

    GPU4PySCF contracts a real lattice-phase tensor and reinterprets its
    final size-two axis as ``complex128``.  CuPy requires that axis to be
    contiguous.  Only the affected tensor is copied and the process-global
    symbol is restored under a lock before this context exits.
    '''
    with _RESPONSE_LAYOUT_LOCK:
        original = gpu_gdf_grad.contract

        def contract_contiguous(expression, *args, **kwargs):
            value = original(expression, *args, **kwargs)
            if (expression == 'tLr,LKz->trKz'
                    and not value.flags.c_contiguous):
                provider.sampled_k_response_layout_copy_count += 1
                provider.sampled_k_response_layout_copy_bytes += int(
                    value.nbytes)
                value = cp.ascontiguousarray(value)
            return value

        gpu_gdf_grad.contract = contract_contiguous
        try:
            yield
        finally:
            gpu_gdf_grad.contract = original


def make_native_gradient_response(
        mf_hf, dm0, dz, omega, kmesh, atmlst=None, *, backend_name):
    r'''Return density-contracted sampled-k J and source-Fock derivatives.

    For the quadratic Gaussian-DF functional

    ``Q[D] = Tr[D (J[D] - K[D]/2)] / 2``,

    polarization gives the two observables needed by DC-DFT,

    ``E_J^lambda[P] = d Tr[P J[P]] / (2 dR_lambda)``

    and

    ``E_F^lambda[P,D_Z] = -(Q[P+D_Z]^lambda-Q[P-D_Z]^lambda)``.

    The three native calls share one immutable ``SRInt3c2eOpt``.  Large
    source, metric, and density contractions stay on the device; the native
    GPU4PySCF endpoint returns only three small per-atom arrays to the host.
    It does not expose a source-versus-metric split, so the result is labelled
    ``native_combined`` rather than assigning invented component terms.

    Args:
        mf_hf : gpu4pyscf.pbc.scf.khf.KRHF
            Converged source KRHF object.
        dm0, dz : cupy.ndarray of shape (nkpts,nao,nao)
            Hermitian source and Z-transition densities.
        omega : float
            Short-/long-range partition used by the native integral
            optimizer.  The complete ``omega=0`` Coulomb operator is still
            evaluated by each observable call.
        kmesh : array_like of shape (3,)
            BvK mesh corresponding to the explicit full-BZ k-point list.

    Kwargs:
        atmlst : sequence of int or None
            Requested nuclear centers.
        backend_name : str
            Provider name recorded in the response provenance.

    Returns:
        tuple
            ``(terms, metadata)`` where ``terms`` contains device ``de_j``
            and ``de_fock`` arrays in Ha/Bohr.
    '''
    provider = mf_hf.with_df
    kpts = check_kpts_equal(
        mf_hf.kpts, provider.kpts,
        f'GPU sampled-k {backend_name.upper()} response kpts')
    if len(kpts) < 2 and not np.any(kpts):
        raise NotImplementedError(
            'sampled-k Gaussian response excludes the native Gamma route')
    dm0 = _validate_density(provider, dm0, 'dm0')
    dz = _validate_density(provider, dz, 'dz')
    atoms = (
        np.arange(provider.cell.natm, dtype=int)
        if atmlst is None else np.asarray(atmlst, dtype=int))
    omega = float(omega)
    kmesh = np.asarray(kmesh, dtype=int)
    if omega <= 0:
        raise ValueError('native sampled-k response omega must be positive')
    if kmesh.shape != (3,) or np.any(kmesh < 1):
        raise ValueError('native sampled-k response kmesh must be positive (3,)')

    call_seconds = {}
    with (contiguous_int1e_hermitian(mf_hf),
          _contiguous_complex_view(provider)):
        start = perf_counter()
        int3c2e_opt = SRInt3c2eOpt(
            provider.cell, provider.auxcell, omega, kmesh).build()
        optimizer_build_seconds = perf_counter() - start

        start = perf_counter()
        de_j_host = np.asarray(_jk_energy_per_atom(
            int3c2e_opt, dm0, kpts=kpts, hermi=1,
            j_factor=1, k_factor=0, exxdiv=None, omega=0))
        call_seconds['target_j'] = perf_counter() - start
        start = perf_counter()
        plus_host = np.asarray(_jk_energy_per_atom(
            int3c2e_opt, dm0 + dz, kpts=kpts, hermi=1,
            j_factor=1, k_factor=1, exxdiv=mf_hf.exxdiv, omega=0))
        call_seconds['source_plus'] = perf_counter() - start
        start = perf_counter()
        minus_host = np.asarray(_jk_energy_per_atom(
            int3c2e_opt, dm0 - dz, kpts=kpts, hermi=1,
            j_factor=1, k_factor=1, exxdiv=mf_hf.exxdiv, omega=0))
        call_seconds['source_minus'] = perf_counter() - start

    de_fock_host = -(plus_host - minus_host)
    host_bytes = int(
        de_j_host.nbytes + plus_host.nbytes + minus_host.nbytes)
    terms = {
        'de_j': cp.asarray(de_j_host[atoms]),
        'de_fock': cp.asarray(de_fock_host[atoms]),
    }
    metadata = {
        'response_backend': f'gpu4pyscf_{backend_name}_native_combined',
        'term_resolution': 'source_plus_metric_plus_ewald_combined',
        'quadratic_native_calls': 3,
        'native_optimizer_builds': 1,
        'native_optimizer_build_seconds': optimizer_build_seconds,
        'native_call_seconds': call_seconds,
        'native_total_seconds': (
            optimizer_build_seconds + sum(call_seconds.values())),
        'native_optimizer_reuse': True,
        'native_partition_omega': omega,
        'native_bvk_kmesh': kmesh.tolist(),
        'private_api': (
            'gpu4pyscf.pbc.df.grad.krhf._jk_energy_per_atom',
            'gpu4pyscf.pbc.df.int3c2e.SRInt3c2eOpt',
        ),
        'native_batching_owner': (
            'GPU4PySCF dynamic auxiliary and reciprocal blocks'),
        'native_host_result_bytes': host_bytes,
        'layout_copy_count': int(
            provider.sampled_k_response_layout_copy_count),
        'layout_copy_bytes': int(
            provider.sampled_k_response_layout_copy_bytes),
        'scientific_host_fallback': False,
    }
    return terms, metadata


def _validate_density(provider, density, name):
    '''Return one finite Hermitian device sampled-k density.'''
    density = cp.asarray(density, order='C')
    shape = (
        len(provider.kpts), provider.cell.nao_nr(), provider.cell.nao_nr())
    if density.shape != shape:
        raise ValueError(f'{name} must have shape {shape}')
    if not bool(cp.all(cp.isfinite(density)).get()):
        raise ValueError(f'{name} contains nonfinite values')
    error = float(cp.max(cp.abs(
        density - density.conj().transpose(0, 2, 1))).get())
    if error > 1e-9:
        raise ValueError(f'{name} must be Hermitian')
    return density
