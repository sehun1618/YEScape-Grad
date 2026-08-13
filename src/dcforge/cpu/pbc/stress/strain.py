'''Homogeneous-strain conventions shared by periodic stress implementations.'''

import hashlib
import json

import numpy as np

VOIGT_LABELS = ('xx', 'yy', 'zz', 'yz', 'xz', 'xy')

# Canonical public conversion fixed by ``docs/goals/goal_pbc_stress.md`` and
# ``docs/project/CONVENTIONS.md``.  PySCF 2.14.0 bundles an older rounded pair of Hartree
# and Bohr SI constants whose quotient differs in the seventh significant
# digit; using it here would make reports environment-version dependent.
HARTREE_PER_BOHR3_TO_GPA = 29421.015756


def voigt_strain_basis(dtype=float):
    r'''Return the six engineering-Voigt strain directions.

    The order is ``[xx, yy, zz, yz, xz, xy]``.  A shear coordinate is an
    engineering shear :math:`\gamma_{ab}` and therefore its tensor direction
    has :math:`\varepsilon_{ab}=\varepsilon_{ba}=1/2`.  Consequently

    .. math::

        \frac{dE}{d\gamma_{ab}}
        = \frac12\left(\frac{dE}{d\varepsilon_{ab}}
                       +\frac{dE}{d\varepsilon_{ba}}\right),

    which is the physical symmetric stress component after division by the
    central volume.

    Kwargs:
        dtype : dtype
            NumPy dtype of the returned directions.

    Returns:
        ndarray of shape (6, 3, 3)
            Symmetric engineering-Voigt strain basis.
    '''
    basis = np.zeros((6, 3, 3), dtype=dtype)
    basis[0, 0, 0] = 1
    basis[1, 1, 1] = 1
    basis[2, 2, 2] = 1
    basis[3, 1, 2] = basis[3, 2, 1] = .5
    basis[4, 0, 2] = basis[4, 2, 0] = .5
    basis[5, 0, 1] = basis[5, 1, 0] = .5
    return basis


def raw_strain_basis(dtype=float):
    '''Return the nine ordered strain directions ``e_xy`` in row-major order.'''
    return np.eye(9, dtype=dtype).reshape(9, 3, 3)


def symmetrize_tensor(tensor):
    '''Return ``(tensor + tensor.T) / 2`` after validating a ``(3, 3)`` shape.'''
    tensor = _as_tensor(tensor, 'tensor')
    return (tensor + tensor.T.conj()) * .5


def stress_tensor_to_voigt(stress):
    r'''Convert a stress tensor to ``[xx, yy, zz, yz, xz, xy]``.

    Stress is conjugate to engineering strain.  The off-diagonal Voigt value
    is therefore the average of the two raw tensor entries, not their sum.
    '''
    stress = symmetrize_tensor(stress)
    return np.asarray((
        stress[0, 0], stress[1, 1], stress[2, 2],
        stress[1, 2], stress[0, 2], stress[0, 1],
    ))


def stress_voigt_to_tensor(voigt):
    '''Convert physical stress Voigt components to a symmetric tensor.'''
    voigt = _as_voigt(voigt)
    tensor = np.zeros((3, 3), dtype=voigt.dtype)
    tensor[0, 0], tensor[1, 1], tensor[2, 2] = voigt[:3]
    tensor[1, 2] = tensor[2, 1] = voigt[3]
    tensor[0, 2] = tensor[2, 0] = voigt[4]
    tensor[0, 1] = tensor[1, 0] = voigt[5]
    return tensor


def strain_tensor_to_voigt(strain):
    r'''Convert a symmetric strain tensor to engineering-Voigt coordinates.

    Off-diagonal coordinates are doubled because
    :math:`\gamma_{ab}=2\varepsilon_{ab}`.
    '''
    strain = symmetrize_tensor(strain)
    return np.asarray((
        strain[0, 0], strain[1, 1], strain[2, 2],
        2 * strain[1, 2], 2 * strain[0, 2], 2 * strain[0, 1],
    ))


def strain_voigt_to_tensor(voigt):
    '''Convert engineering-Voigt strain coordinates to a symmetric tensor.'''
    voigt = _as_voigt(voigt)
    tensor = np.zeros((3, 3), dtype=voigt.dtype)
    tensor[0, 0], tensor[1, 1], tensor[2, 2] = voigt[:3]
    tensor[1, 2] = tensor[2, 1] = .5 * voigt[3]
    tensor[0, 2] = tensor[2, 0] = .5 * voigt[4]
    tensor[0, 1] = tensor[1, 0] = .5 * voigt[5]
    return tensor


def stress_from_work(work, volume):
    r'''Convert an energy-like strain derivative to physical stress.

    Args:
        work : ndarray of shape (3, 3)
            Raw :math:`dE/d\varepsilon_{ij}` in Hartree.
        volume : float
            Central Cell volume in Bohr cubed.

    Returns:
        ndarray of shape (3, 3)
            Tensile-positive symmetric stress in Ha/Bohr^3.
    '''
    work = _as_tensor(work, 'work')
    volume = float(volume)
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError('central Cell volume must be positive and finite')
    return symmetrize_tensor(work) / volume


def pressure_from_stress(stress):
    r'''Return compressive-positive pressure ``-trace(stress) / 3``.'''
    stress = _as_tensor(stress, 'stress')
    return -np.trace(stress).real / 3


def virial_from_stress(stress, volume):
    r'''Return the virial tensor ``-volume * stress`` in Hartree.'''
    stress = _as_tensor(stress, 'stress')
    volume = float(volume)
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError('central Cell volume must be positive and finite')
    return -volume * stress


def stress_to_gpa(stress):
    '''Convert stress from Ha/Bohr^3 to GPa.'''
    return np.asarray(stress) * HARTREE_PER_BOHR3_TO_GPA


def fractional_coordinates(cell):
    r'''Return row-vector fractional coordinates ``R_cart = R_frac @ a``.'''
    lattice = np.asarray(cell.lattice_vectors(), dtype=float)
    return np.asarray(cell.atom_coords(), dtype=float) @ np.linalg.inv(lattice)


def deform_cell(cell, strain, scale=1., mesh=None):
    r'''Return a Cell after homogeneous row-lattice deformation.

    PySCF stores lattice vectors as rows.  For the canonical column-lattice
    deformation :math:`h'=(I+\lambda\varepsilon)h`, this function therefore
    applies

    .. math::

        a' = a(I+\lambda\varepsilon)^T,
        \qquad R'_{\rm cart}=R_{\rm frac}a'.

    Atomic fractional coordinates and the resolved integer FFT mesh are held
    fixed.  ``cell.ke_cutoff`` remains provenance only; it is not allowed to
    re-round the strained mesh during :meth:`Cell.build`.

    Args:
        cell : pyscf.pbc.gto.Cell
            Built central Cell.
        strain : ndarray of shape (3, 3)
            Dimensionless strain direction or tensor.

    Kwargs:
        scale : float
            Scalar displacement multiplying ``strain``.
        mesh : array_like of length 3 or None
            Resolved central integer mesh.  ``None`` uses ``cell.mesh``.

    Returns:
        pyscf.pbc.gto.Cell
            Independently built strained Cell.
    '''
    strain = _as_tensor(strain, 'strain')
    if np.max(np.abs(strain.imag)) > 0:
        raise ValueError('homogeneous strain must be real')
    strain = np.asarray(strain.real, dtype=float)
    scale = float(scale)
    if not np.isfinite(scale):
        raise ValueError('strain scale must be finite')
    deformation = np.eye(3) + scale * strain
    determinant = np.linalg.det(deformation)
    if not np.isfinite(determinant) or determinant <= 0:
        raise ValueError('strain deformation must preserve positive volume')

    lattice = np.asarray(cell.lattice_vectors(), dtype=float)
    lattice_new = lattice @ deformation.T
    fractional = fractional_coordinates(cell)
    coordinates_new = fractional @ lattice_new
    if mesh is None:
        mesh = np.asarray(cell.mesh, dtype=int)
    mesh = np.asarray(mesh, dtype=int)
    if mesh.shape != (3,) or np.any(mesh <= 0):
        raise ValueError('resolved strain mesh must contain three positive integers')

    strained = cell.copy()
    strained.atom = [
        (cell.atom_symbol(ia), coordinates_new[ia])
        for ia in range(cell.natm)
    ]
    strained.unit = 'Bohr'
    strained.build(
        dump_input=False, parse_arg=False,
        a=lattice_new, mesh=mesh.copy())

    if not np.array_equal(np.asarray(strained.mesh), mesh):
        raise RuntimeError('strained Cell changed the resolved integer mesh')
    if not np.allclose(
            fractional_coordinates(strained), fractional,
            atol=5.e-13, rtol=0):
        raise RuntimeError('strained Cell did not preserve fractional coordinates')
    return strained


def snapshot_strain_state(cell, with_df=None, numint=None):
    r'''Return JSON-safe numerical-representation metadata for strain gates.

    Lattice vectors, Cartesian atom positions and volume are intentionally
    reported outside ``representation`` because they must change under
    strain.  Basis, pseudopotential, integer mesh, auxiliary representation,
    factorization thresholds and multigrid hierarchy belong to
    ``representation`` and must remain identical between ``+h`` and ``-h``.
    '''
    representation = {
        'dimension': int(cell.dimension),
        'low_dim_ft_type': str(cell.low_dim_ft_type),
        'mesh': np.asarray(cell.mesh, dtype=int).tolist(),
        'ke_cutoff': _optional_float(cell.ke_cutoff),
        'precision': _optional_float(cell.precision),
        'natm': int(cell.natm),
        'nao': int(cell.nao_nr()),
        'basis_sha256': _object_sha256(cell._basis),
        'pseudo_sha256': _object_sha256(cell._pseudo),
        'atom_symbols': [cell.atom_symbol(ia) for ia in range(cell.natm)],
    }
    if with_df is not None:
        provider = {
            'module': type(with_df).__module__,
            'class': type(with_df).__name__,
            'auxbasis': repr(getattr(with_df, 'auxbasis', None)),
            'omega': _optional_float(getattr(with_df, 'omega', None)),
            'linear_dep_threshold': _optional_float(
                getattr(with_df, 'linear_dep_threshold', None)),
            'exp_to_discard': _optional_float(
                getattr(with_df, 'exp_to_discard', None)),
        }
        provider_mesh = getattr(with_df, 'mesh', None)
        if provider_mesh is not None:
            provider['mesh'] = np.asarray(provider_mesh, dtype=int).tolist()
        # RSDF resolves source and metric independently.  They are part of
        # the finite Hamiltonian and must remain fixed across a strain FD;
        # omitting them would let a changed derivative representation pass
        # the same-state gate silently.
        for name in ('mesh_compact', 'mesh_j2c'):
            value = getattr(with_df, name, None)
            if value is not None:
                provider[name] = np.asarray(value, dtype=int).tolist()
        for name in (
                'omega_j2c', 'ke_cutoff', 'precision_R', 'precision_G',
                'precision_j2c'):
            value = getattr(with_df, name, None)
            if value is not None:
                provider[name] = float(value)
        resolved_omega = getattr(with_df, 'resolved_omega', None)
        resolved_source_mesh = getattr(
            with_df, 'resolved_source_mesh', None)
        resolved_metric_mesh = getattr(
            with_df, 'resolved_metric_mesh', None)
        if resolved_omega is not None:
            provider['resolved_omega'] = float(resolved_omega)
        if resolved_source_mesh is not None:
            provider['resolved_source_mesh'] = np.asarray(
                resolved_source_mesh, dtype=int).tolist()
        if resolved_metric_mesh is not None:
            provider['resolved_metric_mesh'] = np.asarray(
                resolved_metric_mesh, dtype=int).tolist()
        pp_omega = getattr(with_df, 'pp_loc_part1_omega_resolved', None)
        pp_mesh = getattr(with_df, 'pp_loc_part1_mesh', None)
        pp_mode = getattr(with_df, 'pp_loc_part1_plan_mode', None)
        if pp_omega is not None:
            provider['pp_loc_part1_omega_resolved'] = float(pp_omega)
        if pp_mesh is not None:
            provider['pp_loc_part1_mesh'] = np.asarray(
                pp_mesh, dtype=int).tolist()
        if pp_mode is not None:
            provider['pp_loc_part1_plan_mode'] = str(pp_mode)
        auxcell = getattr(with_df, 'auxcell', None)
        if auxcell is not None:
            provider['naux'] = int(auxcell.nao_nr())
            provider['auxbasis_sha256'] = _object_sha256(auxcell._basis)
        representation['provider'] = provider
    if numint is not None:
        integrator = {
            'module': type(numint).__module__,
            'class': type(numint).__name__,
        }
        for name in ('ntasks', 'ke_ratio', 'rel_cutoff', 'xc_with_j'):
            if hasattr(numint, name):
                value = getattr(numint, name)
                if isinstance(value, (bool, np.bool_)):
                    integrator[name] = bool(value)
                elif value is not None:
                    integrator[name] = float(value)
        integrator_mesh = getattr(numint, 'mesh', None)
        if integrator_mesh is not None:
            integrator['mesh'] = np.asarray(
                integrator_mesh, dtype=int).tolist()
        representation['numint'] = integrator
    return {
        'lattice_bohr': np.asarray(
            cell.lattice_vectors(), dtype=float).tolist(),
        'fractional_coordinates': fractional_coordinates(cell).tolist(),
        'volume_bohr3': float(cell.vol),
        'representation': representation,
    }


def assert_same_strain_representation(reference, candidate):
    '''Raise if two strain snapshots use different numerical representations.'''
    if not isinstance(reference, dict) or not isinstance(candidate, dict):
        raise TypeError('strain state snapshots must be dictionaries')
    reference_representation = reference.get('representation')
    candidate_representation = candidate.get('representation')
    if reference_representation != candidate_representation:
        raise ValueError(
            'strained calculation changed mesh/provider/integrator '
            'representation')


def _as_tensor(value, name):
    value = np.asarray(value)
    if value.shape != (3, 3):
        raise ValueError(f'{name} must have shape (3, 3)')
    if not np.all(np.isfinite(value)):
        raise ValueError(f'{name} must contain finite values')
    return value


def _as_voigt(value):
    value = np.asarray(value)
    if value.shape != (6,):
        raise ValueError('Voigt array must have shape (6,)')
    if not np.all(np.isfinite(value)):
        raise ValueError('Voigt array must contain finite values')
    return value


def _optional_float(value):
    if value is None:
        return None
    return float(value)


def _object_sha256(value):
    canonical = json.dumps(
        _canonical_object(value), sort_keys=True,
        separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def _canonical_object(value):
    '''Return a process-stable JSON representation of PySCF input data.'''
    if isinstance(value, dict):
        return {
            str(key): _canonical_object(item)
            for key, item in sorted(
                value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_object(item) for item in value]
    if isinstance(value, np.ndarray):
        return {
            '__ndarray_dtype__': str(value.dtype),
            '__ndarray_shape__': list(value.shape),
            'values': _canonical_object(value.tolist()),
        }
    if isinstance(value, np.generic):
        return _canonical_object(value.item())
    if isinstance(value, complex):
        return {'__complex__': [value.real, value.imag]}
    if isinstance(value, bytes):
        return {'__bytes_hex__': value.hex()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {'__type__': type(value).__name__, '__repr__': repr(value)}
