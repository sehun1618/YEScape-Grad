'''Full-Brillouin-zone k-point contracts for CPU periodic workflows.'''

import hashlib

import numpy as np
from pyscf import lib
from pyscf.pbc.lib.kpts import KPoints

IMAGINARY_RESIDUAL_TOL = 1e-10
DENSE_RESPONSE_RESIDUAL_TOL = 1e-10
PRODUCTION_RESPONSE_RESIDUAL_TOL = 1e-8


def normalize_kpts(kpts, *, required=True):
    '''Return an explicit full-BZ k-point array in reciprocal Bohr.

    Args:
        kpts : array_like or None
            Cartesian k-points with shape ``(nkpts, 3)``.  A one-dimensional
            length-three vector is deliberately not reshaped because the
            public sampled-k contract requires the k-point axis to remain
            visible.

    Kwargs:
        required : bool
            If ``True``, reject ``None``.  If ``False``, ``None`` selects the
            existing one-Gamma provider default.

    Returns:
        numpy.ndarray
            A finite C-contiguous ``float64`` copy with shape ``(nkpts, 3)``.

    Notes:
        ``pyscf.pbc.lib.kpts.KPoints`` represents symmetry-reduced sampling
        and nonuniform irreducible-zone weights.  The first DC-Forge
        sampled-k release accepts full-BZ arrays with equal weights only, so
        a ``KPoints`` object is rejected rather than silently expanded.
    '''
    if isinstance(kpts, KPoints):
        raise NotImplementedError(
            'KPoints/IBZ symmetry objects are not supported; provide an '
            'explicit full-BZ ndarray with shape (nkpts, 3)')
    if kpts is None:
        if required:
            raise ValueError(
                'KRHFDFT requires an explicit full-BZ kpts array with shape '
                '(nkpts, 3)')
        return np.zeros((1, 3), dtype=np.float64)

    values = np.asarray(kpts)
    if np.iscomplexobj(values) and np.any(values.imag != 0):
        raise ValueError('kpts must contain real Cartesian reciprocal vectors')
    try:
        values = np.asarray(values.real, dtype=np.float64)
    except (TypeError, ValueError) as err:
        raise TypeError('kpts must be a numerical ndarray') from err
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
        raise ValueError('kpts must have shape (nkpts, 3) with nkpts > 0')
    if not np.all(np.isfinite(values)):
        raise ValueError('kpts must contain only finite values')
    values = np.array(values, dtype=np.float64, order='C', copy=True)
    # Cartesian +0.0 and -0.0 are the same k-point.  Canonicalizing signed
    # zeros keeps byte-level provenance fingerprints invariant under the
    # physically meaningful time-reversal operation k -> -k at Gamma.
    values[values == 0] = 0.0
    return values


def equal_kpts_weights(kpts):
    '''Return the first-release full-BZ weights ``w_k = 1 / nkpts``.'''
    nkpts = normalize_kpts(kpts).shape[0]
    return np.full(nkpts, 1.0 / nkpts, dtype=np.float64)


def check_kpts_equal(reference, candidate, name='candidate kpts'):
    '''Require identical k-point coordinates and ordering.'''
    reference = normalize_kpts(reference)
    candidate = normalize_kpts(candidate)
    if reference.shape != candidate.shape or not np.array_equal(
            reference, candidate):
        raise ValueError(
            f'{name} must exactly match the target full-BZ kpts and ordering')
    return candidate


def check_equal_weights(kpts, weights):
    '''Validate the equal-weight full-BZ convention.'''
    expected = equal_kpts_weights(kpts)
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != expected.shape:
        raise ValueError('kpts_weights must have shape (nkpts,)')
    if not np.all(np.isfinite(values)):
        raise ValueError('kpts_weights must contain only finite values')
    if not np.array_equal(values, expected):
        raise NotImplementedError(
            'nonuniform k-point weights and IBZ sampling are not supported')
    return values


def weighted_inner_product(x, y, weights, *, imaginary_tol=None,
                           name='weighted inner product'):
    r'''Return :math:`\sum_k w_k\,\mathrm{Re}\,\mathrm{Tr}[X_k^\dagger Y_k]`.

    The complete complex scalar is formed first.  Its imaginary component is
    therefore available as a diagnostic and can be rejected with
    ``imaginary_tol`` before the mathematically required real part is taken.
    All dimensions following the leading k-point axis are contracted.
    '''
    x = np.asarray(x)
    y = np.asarray(y)
    weights = np.asarray(weights, dtype=np.float64)
    if x.shape != y.shape or x.ndim < 1:
        raise ValueError('weighted-inner-product operands must have equal shape')
    if x.shape[0] != weights.size:
        raise ValueError('operand k-point axis and weights are inconsistent')
    products = (x.conj() * y).reshape(x.shape[0], -1)
    value = lib.einsum('k,ki->', weights, products)
    if imaginary_tol is not None and abs(value.imag) > imaginary_tol:
        raise ValueError(
            f'{name} imaginary residual {abs(value.imag):.3e} exceeds '
            f'{imaginary_tol:.3e}')
    return float(value.real)


def max_abs_residual(residual):
    '''Return the maximum complex modulus of an independently formed residual.'''
    residual = np.asarray(residual)
    if residual.size == 0:
        return 0.0
    return float(np.max(np.abs(residual)))


def validate_insulating_occupations(mo_energy, mo_occ, *, occ_tol=1e-10,
                                    gap_tol=1e-8):
    '''Validate exact 2/0 KRHF occupations and a fixed insulating topology.

    Args:
        mo_energy : sequence of ndarray
            Canonical orbital energies for every full-BZ k-point.
        mo_occ : sequence of ndarray
            Restricted orbital occupations for every k-point.

    Kwargs:
        occ_tol : float
            Absolute tolerance used only to recognize exact zero and two.
        gap_tol : float
            Minimum allowed source-HF HOMO-LUMO gap in Hartree.

    Returns:
        float
            The minimum direct source-HF gap over the supplied k-points.

    Notes:
        Smearing, fractional occupations and changing occupied dimensions
        require a Fermi-level response that is outside the first release.
        They are rejected before a fixed-density target energy is exposed as
        a validated sampled-k capability.
    '''
    energies = tuple(np.asarray(value) for value in mo_energy)
    occupations = tuple(np.asarray(value) for value in mo_occ)
    if not energies or len(energies) != len(occupations):
        raise ValueError('mo_energy and mo_occ must contain one block per k-point')

    nocc_ref = None
    occupied_energies = []
    virtual_energies = []
    for k, (energy, occupation) in enumerate(zip(energies, occupations)):
        if energy.ndim != 1 or occupation.shape != energy.shape:
            raise ValueError(f'invalid MO energy/occupation shape at k-point {k}')
        is_occ = np.isclose(occupation, 2.0, atol=occ_tol, rtol=0.0)
        is_vir = np.isclose(occupation, 0.0, atol=occ_tol, rtol=0.0)
        if not np.all(is_occ | is_vir):
            raise NotImplementedError(
                'fractional occupations, smearing and metals are not supported')
        nocc = int(np.count_nonzero(is_occ))
        if nocc_ref is None:
            nocc_ref = nocc
        elif nocc != nocc_ref:
            raise NotImplementedError(
                'all k-points must have the same occupied-orbital topology')
        if nocc == 0 or nocc == energy.size:
            raise NotImplementedError(
                'sampled-k KRHFDFT requires occupied and virtual orbitals')
        occupied_energies.append(np.asarray(energy[is_occ]).real)
        virtual_energies.append(np.asarray(energy[is_vir]).real)

    # A positive direct gap at every k-point is insufficient: the valence
    # maximum and conduction minimum may occur at different k-points.  The
    # fixed-topology gate therefore uses the global sampled-mesh gap.
    gap = float(min(np.min(value) for value in virtual_energies)
                - max(np.max(value) for value in occupied_energies))
    if not np.isfinite(gap) or gap <= gap_tol:
        raise NotImplementedError(
            f'source KRHF must be insulating; global sampled-k gap '
            f'{gap:.3e} Ha does not exceed {gap_tol:.3e} Ha')
    return gap


def validate_kresolved_hermitian(values, nkpts, nao, *, hermi_tol=1e-10,
                                  name='k-resolved matrix'):
    '''Validate shape, finiteness and Hermiticity of complex AO blocks.'''
    values = np.asarray(values)
    if values.shape != (nkpts, nao, nao):
        raise ValueError(f'{name} must have shape (nkpts, nao, nao)')
    if not np.all(np.isfinite(values)):
        raise ValueError(f'{name} contains nonfinite values')
    residual = values - values.conj().transpose(0, 2, 1)
    error = max_abs_residual(residual)
    if error > hermi_tol:
        raise ValueError(
            f'{name} Hermiticity residual {error:.3e} exceeds '
            f'{hermi_tol:.3e}')
    return values


def kpoint_fingerprint(kpts, weights, *, wrap_around=None):
    '''Return an immutable digest of the sampled-k representation.'''
    kpts = normalize_kpts(kpts)
    weights = check_equal_weights(kpts, weights)
    if wrap_around not in (None, True, False):
        raise TypeError('kpts_wrap_around must be True, False or None')
    digest = hashlib.sha256()
    digest.update(np.asarray(kpts.shape, dtype=np.int64).tobytes())
    digest.update(kpts.tobytes())
    digest.update(weights.tobytes())
    digest.update(str(wrap_around).encode('ascii'))
    return digest.hexdigest()
