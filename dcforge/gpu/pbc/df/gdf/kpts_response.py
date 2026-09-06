'''Density-contracted sampled-k GDF response through GPU4PySCF kernels.'''

import numpy as np
from pyscf.pbc.tools.k2gamma import kpts_to_kmesh

from dcforge.gpu.pbc.df.gaussian.kpts_response import (
    make_native_gradient_response,
)
from dcforge.gpu.pbc.df.gdf.configuration import resolved_response_settings
from dcforge.gpu.pbc.df.gdf.gdf import GDF


def make_gradient_response(mf_hf, dm0, dz, atmlst=None):
    r'''Return sampled-k target-J and source-Fock scalar derivatives.

    The native quadratic functional

    ``Q[D] = Tr[D (J[D]-K[D]/2)] / 2``

    yields the DC-DFT source-Fock response through

    ``de_fock = -(Q[P+D_Z]^lambda-Q[P-D_Z]^lambda)``.

    GPU4PySCF returns only the combined source-plus-metric result.  DC-Forge
    therefore records that boundary instead of inventing unavailable terms.

    Returns:
        tuple
            Device ``de_j`` and ``de_fock`` arrays plus provenance metadata.
    '''
    provider = mf_hf.with_df
    if not isinstance(provider, GDF):
        raise TypeError('sampled-k GDF response requires a GPU GDF provider')
    if provider.auxcell is None:
        provider.build(j_only=False)
    kpts = np.asarray(provider.kpts, dtype=float).reshape(-1, 3)
    omega, _source_mesh, _metric_mesh = resolved_response_settings(provider)
    kmesh = kpts_to_kmesh(
        provider.cell, kpts, rcut=provider.cell.rcut + 10,
        bound_by_supmol=False)
    terms, metadata = make_native_gradient_response(
        mf_hf, dm0, dz, omega, kmesh, atmlst=atmlst,
        backend_name='gdf')
    metadata['response_backend'] = 'gpu4pyscf_native_combined'
    metadata['partition_owner'] = 'provider.resolved_omega from source CDERI'
    return terms, metadata
