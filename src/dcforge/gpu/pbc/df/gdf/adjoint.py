'''Direct real-Gamma GDF primitive adjoints for the DFA@RHF Z response.

The native Gamma derivative is naturally expressed in terms of an
unorthogonalized three-center tensor ``B_P`` and auxiliary metric ``M``.  For
the metric-solved tensor ``W = M^{-1} B``, this module builds the reverse
sources which contract directly with ``dB`` and ``dM``.  It is therefore an
algebraic replacement for the polarization calculation with ``P + D_Z`` and
``P - D_Z`` densities; it never constructs either full-rank signed density.
'''

import cupy as cp
from gpu4pyscf.lib.cupy_helper import contract


def build_rhf_primitive_adjoint(solved_three_center, reference_density,
                                response_density, include_target_j=False,
                                response_sign=1.,
                                auxiliary_block_size=32):
    r'''Build reverse ``dB`` and ``dM`` sources at real Gamma.

    Let ``W_P = M^{-1} B_P``, ``P`` be the source RHF density, and ``T`` be
    the Hermitian Z-vector transition density.  The returned ``source[P]``
    obeys

    .. math::

       \sum_P \mathrm{Tr}[source_P^T dB_P]
       - \sum_{PQ} metric_{PQ} dM_{PQ}
       = E_J[P]^\lambda
       + s\{Q[P+T]^\lambda - Q[P-T]^\lambda\},

    where ``Q[D] = Tr[D(J[D]-K[D]/2)]/2`` and ``s=response_sign``.
    ``include_target_j=True`` adds the first target-Hartree contribution;
    ``s=-1`` gives the DC-Forge ``de_j_fock_native`` convention.  The formula
    is not a numerical finite difference or an approximation to the
    signed-state result.

    Args:
        solved_three_center : cupy.ndarray of shape (naux, nao, nao)
            Real, AO-symmetric ``W = M^{-1} B`` in the primitive sorted-AO
            representation.
        reference_density : cupy.ndarray of shape (nao, nao)
            Real symmetric source-RHF density ``P`` in the same AO basis.
        response_density : cupy.ndarray of shape (nao, nao)
            Real symmetric adjoint transition density ``T`` in the same AO
            basis.

    Kwargs:
        include_target_j : bool
            Include ``E_J[P]^lambda`` in addition to the Z-vector response.
        response_sign : float
            Sign of the polarization difference.  Only ``+1`` and ``-1``
            are accepted.  ``-1`` corresponds to
            ``E_J[P] - {Q[P+T]-Q[P-T]}``.
        auxiliary_block_size : int
            Number of auxiliary functions used to build temporary AO source
            blocks.  Persistent ``source`` storage is unavoidable, but this
            cap prevents multiple full ``(naux, nao, nao)`` temporaries.

    Returns:
        tuple
            ``(source, metric, metadata)``.  ``source`` has shape
            ``(naux, nao, nao)`` and contracts with ``dB``; ``metric`` has
            shape ``(naux, naux)`` and contracts with ``dM``.  ``metadata``
            documents the bounded auxiliary blocking used to assemble them.
    '''
    solved = _as_real_matrix_stack(solved_three_center, 'solved_three_center')
    reference = _as_real_square(reference_density, 'reference_density')
    response = _as_real_square(response_density, 'response_density')
    if reference.shape != response.shape:
        raise ValueError(
            'reference_density and response_density must have one shape, '
            f'got {reference.shape} and {response.shape}')
    if tuple(solved.shape[1:]) != tuple(reference.shape):
        raise ValueError(
            'solved_three_center AO shape does not match density shape: '
            f'{solved.shape[1:]} != {reference.shape}')
    if auxiliary_block_size < 1:
        raise ValueError('auxiliary_block_size must be positive')
    response_sign = float(response_sign)
    if response_sign not in (-1., 1.):
        raise ValueError('response_sign must be +1 or -1')

    naux = int(solved.shape[0])
    auxiliary_reference = contract('pij,ji->p', solved, reference)
    auxiliary_response = contract('pij,ji->p', solved, response)
    source = cp.empty_like(solved)
    maximum = min(naux, int(auxiliary_block_size))
    block_count = 0
    for p0 in range(0, naux, maximum):
        p1 = min(naux, p0 + maximum)
        _build_source_block(
            source[p0:p1], solved[p0:p1], reference, response,
            auxiliary_reference[p0:p1], auxiliary_response[p0:p1],
            include_target_j=include_target_j, response_sign=response_sign)
        block_count += 1

    # Differentiating ``W=M^{-1}B`` gives ``-source:W:dM``.  The native
    # metric kernel therefore receives this positive contraction and applies
    # the leading minus sign in the caller.
    metric = contract('pij,qij->pq', source, solved)
    metadata = {
        'route': 'direct_real_gamma_primitive_adjoint',
        'include_target_j': bool(include_target_j),
        'response_sign': response_sign,
        'persistent_source_bytes': int(source.nbytes),
        'metric_bytes': int(metric.nbytes),
        'auxiliary_block_size': maximum,
        'auxiliary_block_count': block_count,
    }
    return source, metric, metadata


def build_rhf_primitive_adjoint_channels(
        solved_three_center, reference_density, response_density,
        response_sign=-1., auxiliary_block_size=32):
    r'''Build separate target-J and Z-Fock primitive adjoint channels.

    The production fused GDF response is the sum of two physical
    observables,

    .. math::

       E_J[P]^\lambda
       + s\{Q[P+T]^\lambda - Q[P-T]^\lambda\},

    where ``s=-1`` for the DC-Forge source-Fock convention.  This helper
    returns the two corresponding reverse sources rather than treating the
    fused source/metric decomposition as a physical J/Fock decomposition.
    In particular, each returned channel has its own pair
    ``(source, metric)`` for

    .. math::

       source : dB - metric : dM.

    ``PBCsr_ejk_int3c2e_ip1`` historically accepts only one ``source``
    channel.  The companion two-channel CUDA ABI is therefore required to
    consume both results in one short-range traversal.  Keeping this
    construction separate from that ABI makes the algebra independently
    testable and keeps the ordinary fused route unchanged.

    Args:
        solved_three_center : cupy.ndarray
            Real Gamma ``W=M^{-1}B`` with shape ``(naux, nao, nao)``.
        reference_density : cupy.ndarray
            Real symmetric RHF density ``P`` in the primitive AO order.
        response_density : cupy.ndarray
            Real symmetric Z transition density ``T`` in the same order.

    Kwargs:
        response_sign : float
            Sign of the Z-Fock polarization term.  DC-Forge uses ``-1.``.
        auxiliary_block_size : int
            Maximum auxiliary block used while building each source.

    Returns:
        tuple
            ``(target, fock, metadata)``.  ``target`` and ``fock`` are
            dictionaries with ``source`` and ``metric`` device arrays.
    '''
    fock_source, fock_metric, fock_metadata = (
        build_rhf_primitive_adjoint(
            solved_three_center, reference_density, response_density,
            include_target_j=False, response_sign=response_sign,
            auxiliary_block_size=auxiliary_block_size))
    combined_source, combined_metric, combined_metadata = (
        build_rhf_primitive_adjoint(
            solved_three_center, reference_density, response_density,
            include_target_j=True, response_sign=response_sign,
            auxiliary_block_size=auxiliary_block_size))

    # The J source is recovered from the exact linear identity
    # ``combined = target + fock``.  This keeps `_build_source_block` as the
    # one authoritative definition of the RHF direct-adjoint coefficients.
    target_source = combined_source
    target_source -= fock_source
    target_metric = combined_metric
    target_metric -= fock_metric
    metadata = {
        'route': 'direct_real_gamma_primitive_adjoint_channels',
        'response_sign': float(response_sign),
        'target': {
            'persistent_source_bytes': int(target_source.nbytes),
            'metric_bytes': int(target_metric.nbytes),
            'construction': 'combined_minus_fock',
        },
        'fock': dict(fock_metadata),
        'combined': dict(combined_metadata),
    }
    return ({'source': target_source, 'metric': target_metric},
            {'source': fock_source, 'metric': fock_metric}, metadata)


def _build_source_block(source, solved, reference, response,
                        auxiliary_reference, auxiliary_response,
                        include_target_j, response_sign):
    '''Fill one bounded auxiliary block of the exact cross-density source.'''
    temporary = cp.empty_like(source)
    exchange = cp.empty_like(source)

    cp.multiply(auxiliary_response[:, None, None], reference[None], out=source)
    cp.multiply(auxiliary_reference[:, None, None], response[None], out=temporary)
    source += temporary
    source *= 2.

    contract('ij,pjk->pik', reference, solved, out=temporary)
    contract('pik,kl->pil', temporary, response, out=exchange)
    source -= exchange
    contract('ij,pjk->pik', response, solved, out=temporary)
    contract('pik,kl->pil', temporary, reference, out=exchange)
    source -= exchange
    source *= response_sign

    if include_target_j:
        cp.multiply(auxiliary_reference[:, None, None], reference[None],
                    out=temporary)
        source += temporary


def _as_real_matrix_stack(value, name):
    '''Validate a real ``(naux, nao, nao)`` primitive tensor.'''
    array = cp.asarray(value)
    if array.ndim != 3 or array.shape[1] != array.shape[2]:
        raise ValueError(f'{name} must have shape (naux, nao, nao)')
    if cp.iscomplexobj(array):
        imaginary = float(cp.max(cp.abs(array.imag)).get())
        if imaginary > 1e-10:
            raise NotImplementedError(f'{name} must be real at Gamma')
        array = array.real
    return cp.asarray(array, dtype=cp.float64)


def _as_real_square(value, name):
    '''Validate a real square Gamma AO density without symmetrizing it.'''
    array = cp.asarray(value)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f'{name} must be a square Gamma AO matrix')
    if cp.iscomplexobj(array):
        imaginary = float(cp.max(cp.abs(array.imag)).get())
        if imaginary > 1e-10:
            raise NotImplementedError(f'{name} must be real at Gamma')
        array = array.real
    asymmetry = float(cp.max(cp.abs(array - array.T)).get())
    if asymmetry > 1e-9:
        raise ValueError(f'{name} must be symmetric at Gamma')
    return cp.asarray(array, dtype=cp.float64)


# Compatibility aliases for the pre-R2SCAN internal spelling.  The reverse
# adjoint depends on the source RHF density and Z transition density, not on
# whether the target semilocal functional is PBE or R2SCAN.
build_rhf_pbe_primitive_adjoint = build_rhf_primitive_adjoint
build_rhf_pbe_primitive_adjoint_channels = build_rhf_primitive_adjoint_channels
