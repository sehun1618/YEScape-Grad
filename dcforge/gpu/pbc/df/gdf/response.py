'''Native density-contracted Gamma-point GDF nuclear-response contractions.

GPU4PySCF exposes periodic Gaussian-DF derivative kernels through its RHF
gradient dispatcher rather than public ``GDF.get_j_e1``/``get_k_e1`` matrix
APIs.  That dispatcher evaluates short-range, reciprocal, auxiliary-metric,
and Ewald response on the device before returning its small per-atom result.
'''

import cupy as cp

from dcforge.gpu.pbc.df.gdf.native_gamma import (
    coulomb_energy_derivative as native_coulomb_energy_derivative,
)
from dcforge.gpu.pbc.df.gdf.native_gamma import (
    fock_bilinear_derivative,
    signed_hf_bilinear_derivative,
    target_j_and_fock_derivative,
    target_j_fock_derivative,
    target_j_fock_signed_derivative,
)


def coulomb_energy_derivative(mf_hf, dm, atmlst=None):
    r'''Return ``d[Tr(P J[P])/2]/dR`` from the active GPU GDF provider.

    Args:
        mf_hf : gpu4pyscf.pbc.scf.hf.RHF
            Converged Gamma-point source RHF that owns the active GDF object.
        dm : cupy.ndarray
            Closed-shell AO density ``P`` including the occupation factor.

    Kwargs:
        atmlst : sequence of int or None
            Centers to retain.  ``None`` requests every center.

    Returns:
        cupy.ndarray
            Coulomb energy derivative of shape ``(len(atmlst), 3)`` in
            Ha/Bohr.

    Notes:
        ``jk_energy_per_atom`` is the native GDF scalar derivative primitive.
        Setting both exchange factors to zero leaves exactly the target
        ``E_J[P] = Tr[P J[P]] / 2`` term.  Its terminal per-atom reduction is
        host-visible in GPU4PySCF; the small result is immediately returned to
        CuPy for the common DC-DFT term assembly.
    '''
    return native_coulomb_energy_derivative(mf_hf, dm, atmlst)


def restricted_hf_bilinear_derivative(mf_hf, dm0, dz, atmlst=None):
    r'''Return source-HF cross derivative through signed polarization.

    For

    .. math::

       Q[D] = \frac12\operatorname{Tr}\{D[J[D]-K[D]/2]\},

    the polarization identity yields

    .. math::

       Q[P+D_Z]^{\lambda} - Q[P-D_Z]^{\lambda}
       = 2\operatorname{Tr}\{D_Z[J[P]-K[P]/2]^{\lambda}\}.

    ``P\pm D_Z`` are indefinite even though they are Hermitian.  The
    dedicated Gamma primitive kernel supplies both signed low-rank states to
    every derivative contraction in one traversal.  It avoids the mutable
    density-factor cache in GPU4PySCF's quadratic public endpoint and never
    materialises ``dCDERI``, ``dJ``, or ``dK`` tensors.

    Args:
        mf_hf : gpu4pyscf.pbc.scf.hf.RHF
            Converged Gamma-point source RHF with active GDF.
        dm0 : cupy.ndarray
            Reference HF AO density ``P``.
        dz : cupy.ndarray
            Hermitian adjoint transition density ``D_Z``.

    Kwargs:
        atmlst : sequence of int or None
            Centers to retain.  ``None`` requests every center.

    Returns:
        cupy.ndarray
            Cross-density derivative with shape ``(len(atmlst), 3)`` in
            Ha/Bohr.

    Notes:
        GPU4PySCF exposes this GDF result only after source, metric and Ewald
        pieces are summed.  Callers must record it as ``native_combined`` and
        must not mislabel it as a source-only contribution.
    '''
    dm0 = cp.asarray(dm0)
    dz = cp.asarray(dz)
    if dm0.shape != dz.shape:
        raise ValueError(
            'reference and adjoint densities must have identical AO shape')
    return signed_hf_bilinear_derivative(mf_hf, dm0, dz, atmlst)


def restricted_hf_bilinear_derivative_direct(
        mf_hf, dm0, dz, atmlst=None, return_metadata=False,
        profile_detail=False, occupied_coefficients=None,
        response_orbitals=None, adjoint_backend='auto'):
    r'''Return the source-HF cross derivative through one direct adjoint.

    This has the same value and sign as
    :func:`restricted_hf_bilinear_derivative`; it avoids constructing the
    two signed full-rank primitive tensors and exposes source/metric/Ewald
    metadata for the MG target-J partition.  ``profile_detail=True`` adds
    validation-only metric/adjoint timers without changing the contraction.
    '''
    dm0 = cp.asarray(dm0)
    dz = cp.asarray(dz)
    if dm0.shape != dz.shape:
        raise ValueError(
            'reference and adjoint densities must have identical AO shape')
    return fock_bilinear_derivative(
        mf_hf, dm0, dz, atmlst, return_metadata=return_metadata,
        profile_detail=profile_detail,
        occupied_coefficients=occupied_coefficients,
        response_orbitals=response_orbitals,
        adjoint_backend=adjoint_backend)


def target_j_fock_energy_derivative(mf_hf, dm0, dz, atmlst=None,
                                    return_metadata=False,
                                    profile_detail=False,
                                    occupied_coefficients=None,
                                    response_orbitals=None,
                                    adjoint_backend='auto'):
    '''Return fused ``de_j + de_fock_native`` from one GDF traversal.

    See :func:`dcforge.gpu.pbc.df.gdf.native_gamma.target_j_fock_derivative`
    for the exact weighted-state expression.  This is the default GPU GDF
    route because it avoids an unsafe sequence of independent native scalar
    derivative dispatches.  With ``return_metadata=True``,
    ``metadata['term_components']`` contains the exact source-primitive,
    auxiliary-metric, and Ewald components of this *fused* observable.  It
    does not imply that target-J and Z-Fock were evaluated in separate native
    dispatches.  ``profile_detail=True`` only decomposes the existing
    metric/adjoint timing stage.
    '''
    return target_j_fock_derivative(
        mf_hf, dm0, dz, atmlst, return_metadata=return_metadata,
        profile_detail=profile_detail,
        occupied_coefficients=occupied_coefficients,
        response_orbitals=response_orbitals,
        adjoint_backend=adjoint_backend)


def target_j_fock_signed_energy_derivative(
        mf_hf, dm0, dz, atmlst=None, return_metadata=False):
    '''Return the signed-state fused target-J/Z-Fock GDF derivative.'''
    return target_j_fock_signed_derivative(
        mf_hf, dm0, dz, atmlst, return_metadata=return_metadata)


def target_j_and_fock_energy_derivative(mf_hf, dm0, dz, atmlst=None):
    r'''Return separate GDF target-J and source-Fock response derivatives.

    This is the opt-in multi-observable counterpart of
    :func:`target_j_fock_energy_derivative`.  It requires an explicitly
    configured ``with_df.native_multi_observable_library`` containing the
    audited GPU4PySCF 1.8.0 two-channel short-range CUDA ABI.  Without that
    library the ordinary fused function is the only safe production route.

    Args:
        mf_hf : gpu4pyscf.pbc.scf.hf.RHF
            Source RHF with the active Gamma GDF provider.
        dm0, dz : cupy.ndarray
            Reference and Z transition densities in physical AO order.

    Kwargs:
        atmlst : sequence of int or None
            Requested centers.  ``None`` selects every atom.

    Returns:
        tuple
            ``(terms, metadata)`` with separate ``de_j`` and ``de_fock``
            device arrays and source/metric/Ewald ledgers.
    '''
    return target_j_and_fock_derivative(mf_hf, dm0, dz, atmlst)
