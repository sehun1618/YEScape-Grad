'''Density-contracted sampled-k GDF response through GPU4PySCF kernels.'''

import numpy as np
from gpu4pyscf.pbc.df.rsdf_builder import _guess_omega
from pyscf.pbc.tools.k2gamma import kpts_to_kmesh

from dcforge.gpu.pbc.df.gaussian.kpts_response import (
    make_native_gradient_response,
)
from dcforge.gpu.pbc.df.gdf.gdf import GDF


def make_gradient_response(mf_hf, dm0, dz, atmlst=None):
    r'''Return sampled-k target-J and source-Fock scalar derivatives.

    The native quadratic functional is

    ``Q[D] = Tr[D (J[D]-K[D]/2)] / 2``.

    Its polarization gives the DC-DFT response directly,

    ``de_fock = -(Q[P+D_Z]^lambda-Q[P-D_Z]^lambda)``.

    The target Hartree term is evaluated by the same native GDF derivative
    with exchange factors set to zero.  GPU4PySCF returns only the combined
    per-atom source-plus-metric result; DC-Forge therefore labels these terms
    ``native_combined`` and does not invent an unavailable source/metric
    split.

    Args:
        mf_hf : dcforge.gpu.pbc.energy.krhf_dft.DFKRHF
            Converged sampled-k source KRHF with a DC-Forge GDF provider.
        dm0, dz : cupy.ndarray of shape (nkpts,nao,nao)
            Source density and Hermitian Z-transition density.

    Kwargs:
        atmlst : sequence of int or None
            Requested nuclear centers.

    Returns:
        tuple
            ``(terms,metadata)`` with device ``de_j`` and ``de_fock`` arrays
            in Ha/Bohr and explicit native host-result transfer accounting.
    '''
    provider = mf_hf.with_df
    if not isinstance(provider, GDF):
        raise TypeError('sampled-k GDF response requires a GPU GDF provider')
    if provider.auxcell is None:
        provider.build(j_only=False)
    kpts = np.asarray(provider.kpts, dtype=float).reshape(-1, 3)
    omega = _guess_omega(provider.cell)
    kmesh = kpts_to_kmesh(
        provider.cell, kpts, rcut=provider.cell.rcut + 10,
        bound_by_supmol=False)
    terms, metadata = make_native_gradient_response(
        mf_hf, dm0, dz, omega, kmesh, atmlst=atmlst,
        backend_name='gdf')
    # Preserve the established public GDF provenance label while the shared
    # implementation records the provider-specific name for new backends.
    metadata['response_backend'] = 'gpu4pyscf_native_combined'
    metadata['private_api'] += (
        'gpu4pyscf.pbc.df.rsdf_builder._guess_omega',)
    return terms, metadata
