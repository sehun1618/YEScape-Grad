'''Analytic Gamma overlap and kinetic homogeneous-strain primitives.'''

import ctypes

import numpy as np
from pyscf import gto, lib
from pyscf.gto import ft_ao as molecular_ft_ao
from pyscf.gto import moleintor
from pyscf.gto.moleintor import libcgto
from pyscf.pbc.df import ft_ao as periodic_ft_ao

from dcforge.cpu.pbc.df.gaussian.ft import load_extension
from dcforge.cpu.pbc.df.gaussian.geometry import copy_with_coordinates


def overlap_strain(cell):
    r'''Return :math:`S`, :math:`S^{(\varepsilon)}` and its two components.'''
    return one_electron_strain(cell, 'int1e_ovlp')


def kinetic_strain(cell):
    r'''Return :math:`T`, :math:`T^{(\varepsilon)}` and its two components.'''
    return one_electron_strain(cell, 'int1e_kin')


def one_electron_strain(cell, integral):
    r'''Differentiate one translationally invariant Gamma two-center matrix.

    For the image-resolved matrix :math:`I_{ijL}`, homogeneous strain at fixed
    fractional coordinates gives

    .. math::

        I^{(\varepsilon)}_{xy}
        = \sum_L (\nabla_i I_{ijL})_x
          (R_j-R_i+L)_y.

    The center displacement ``R_j-R_i`` is contracted explicitly.  PySCF's
    ``kderiv=1`` periodic integral supplies
    :math:`i\sum_L L_y\nabla_i I_{ijL}`, so the lattice-image moment is
    recovered without retaining a matrix for every lattice vector.

    Args:
        cell : pyscf.pbc.gto.Cell
            Three-dimensional built Cell at real Gamma.
        integral : {'int1e_ovlp', 'int1e_kin'}
            Translationally invariant two-center integral.

    Returns:
        dict
            ``value`` with shape ``(nao,nao)`` and ``raw_derivative``,
            ``center_response``, ``image_response`` with shape
            ``(3,3,nao,nao)``.  Matrix derivatives are in the integral's
            energy unit per dimensionless strain.
    '''
    derivative_names = {
        'int1e_ovlp': 'int1e_ipovlp',
        'int1e_kin': 'int1e_ipkin',
    }
    integral = str(integral)
    if integral not in derivative_names:
        raise ValueError(
            f'unsupported one-electron strain integral {integral!r}')
    if int(cell.dimension) != 3:
        raise NotImplementedError(
            'Gamma one-electron strain currently requires a 3D Cell')

    nao = int(cell.nao_nr())
    value = np.asarray(cell.pbc_intor(integral, hermi=0))
    center_gradient = np.asarray(cell.pbc_intor(
        derivative_names[integral], comp=3, hermi=0))
    image_moment = np.asarray(cell.pbc_intor(
        derivative_names[integral], comp=3, hermi=0,
        kderiv=1)).reshape(3, 3, nao, nao) / 1j
    centers = _ao_centers(cell)
    separations = centers[None, :, :] - centers[:, None, :]
    center_response = lib.einsum(
        'xij,ijy->xyij', center_gradient, separations)
    image_response = image_moment.transpose(1, 0, 2, 3)
    raw_derivative = center_response + image_response
    raw_derivative = _real_if_close(raw_derivative)
    return {
        'integral': integral,
        'value': _real_if_close(value),
        'raw_derivative': raw_derivative,
        'voigt_derivative': matrix_strain_to_voigt(raw_derivative),
        'center_response': _real_if_close(center_response),
        'image_response': _real_if_close(image_response),
    }


def contract_one_electron_strain(strain_data, density):
    r'''Contract a one-electron matrix strain with a spin-summed AO density.

    Returns the raw ``(3,3)`` energy-like derivative
    ``Tr[P I^(strain)]``.  The density is not differentiated here.
    '''
    derivative = np.asarray(strain_data['raw_derivative'])
    density = np.asarray(density)
    if density.shape != derivative.shape[-2:]:
        raise ValueError('one-electron strain density has incompatible shape')
    return lib.einsum('xyij,ji->xy', derivative, density).real


def weighted_coulomb_strain(cell, grid_vectors, omega=0.):
    r'''Differentiate a 3D weighted reciprocal Coulomb kernel.

    For a reciprocal vector :math:`G` that follows the homogeneous Cell
    deformation, the bare kernel and integration weight are

    .. math::

        w_G = \Omega^{-1}\frac{4\pi}{|G|^2} f_\omega(G).

    The returned raw derivative contains both the reciprocal-vector response
    and :math:`d\Omega^{-1}/d\varepsilon_{xy}`.  Positive ``omega`` selects
    the long-range ``erf(omega*r)/r`` kernel and negative ``omega`` the
    short-range complement, following PySCF's ``weighted_coulG`` convention.
    The Gamma zero mode is set to zero; GDF owns its finite correction in a
    separate :func:`g0_metric_strain` term.

    Args:
        cell : pyscf.pbc.gto.Cell
            Three-dimensional central Cell.
        grid_vectors : ndarray of shape (nG, 3)
            Central reciprocal vectors in Bohr**-1.

    Kwargs:
        omega : float
            Range-separation parameter in Bohr**-1.

    Returns:
        dict
            Weighted kernel ``value`` with shape ``(nG,)`` and its raw
            ``(3,3,nG)`` strain derivative.
    '''
    if int(cell.dimension) != 3:
        raise NotImplementedError(
            'weighted Coulomb strain currently requires a 3D Cell')
    vectors = np.asarray(grid_vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError('grid_vectors must have shape (nG,3)')

    omega = float(omega)
    norm2 = lib.einsum('gx,gx->g', vectors, vectors)
    zero_mode = norm2 < 1.e-24
    safe_norm2 = norm2.copy()
    safe_norm2[zero_mode] = np.inf
    bare = 4 * np.pi / safe_norm2
    dyadic = lib.einsum('gx,gy->xyg', vectors, vectors)
    derivative = (
        2 * dyadic / safe_norm2[None, None]
        * bare[None, None])

    if omega:
        damping = np.exp(-safe_norm2 / (4 * omega ** 2))
        damping_response = (
            dyadic / (2 * omega ** 2) * damping[None, None])
        if omega > 0:
            derivative = (
                derivative * damping[None, None]
                + bare[None, None] * damping_response)
            bare *= damping
        else:
            derivative = (
                derivative * (1 - damping)[None, None]
                - bare[None, None] * damping_response)
            bare *= 1 - damping

    inverse_volume = 1 / float(cell.vol)
    value = inverse_volume * bare
    derivative *= inverse_volume
    derivative -= np.eye(3)[:, :, None] * value[None, None]
    value[zero_mode] = 0
    derivative[:, :, zero_mode] = 0
    return {
        'value': value,
        'raw_derivative': derivative,
        'voigt_derivative': matrix_strain_to_voigt(derivative),
    }


def auxiliary_fourier_strain(auxcell, grid_vectors):
    r'''Differentiate auxiliary Gaussian Fourier factors at fixed fractions.

    Write the Fourier transform of an auxiliary Gaussian centered at
    :math:`R_P` as

    .. math::

        F_P(G)=\int \phi_P(r-R_P)e^{-iG\cdot r}\,dr.

    Under affine strain, :math:`G\cdot R_P` is invariant.  Only the Gaussian
    shape response remains,

    .. math::

        F_P^{(xy)}(G)
        =iG_x\int(r_y-R_{P,y})\phi_P(r-R_P)e^{-iG\cdot r}\,dr.

    This is an analytic AO-integral derivative; no displaced Cell or finite
    difference is used.

    Returns:
        dict
            ``value`` with shape ``(naux,nG)`` and raw derivative with shape
            ``(3,3,naux,nG)``.
    '''
    vectors = np.asarray(grid_vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError('grid_vectors must have shape (nG,3)')
    value = np.asarray(periodic_ft_ao.ft_ao(auxcell, vectors)).T
    moment = _fourier_cartesian_moment(auxcell, vectors)
    centers = _ao_centers(auxcell)
    relative_moment = moment - lib.einsum(
        'Py,PG->yPG', centers, value)
    derivative = 1j * lib.einsum(
        'Gx,yPG->xyPG', vectors, relative_moment)
    derivative = _real_if_close(derivative)
    return {
        'value': _real_if_close(value),
        'raw_derivative': derivative,
        'voigt_derivative': matrix_strain_to_voigt(derivative),
    }


def ao_pair_fourier_strain(cell, grid_vectors, rcut=None):
    r'''Differentiate the periodic Gamma AO-pair Fourier tensor.

    For each ket lattice image ``L``, the molecular Fourier integrals retain
    the actual centers ``R_i`` and ``R_j+L``.  The homogeneous-strain response
    is

    .. math::

        F_{ij}^{(xy)}(G)
        =-(\partial_x^iF_{ij})R_{i,y}
         -(\partial_x^jF_{ij})(R_{j,y}+L_y)
         +iG_x\int r_y\chi_i\chi_j^L e^{-iG\cdot r}\,dr.

    The package-owned fused Fourier callback evaluates the overlap and both
    AO-center derivatives from one Gaussian recurrence.  PySCF still owns the
    Fourier driver, spherical transformation and integral conventions.

    Args:
        cell : pyscf.pbc.gto.Cell
            Built real-Gamma primary AO Cell.
        grid_vectors : ndarray of shape (nG, 3)
            Central reciprocal vectors.

    Kwargs:
        rcut : float or None
            Lattice-image cutoff in Bohr.  ``None`` uses ``cell.rcut``.

    Returns:
        dict
            ``value`` with shape ``(nG,nao,nao)`` and raw derivative with
            shape ``(3,3,nG,nao,nao)``.
    '''
    if int(cell.dimension) != 3:
        raise NotImplementedError(
            'AO-pair Fourier strain currently requires a 3D Cell')
    vectors = np.asarray(grid_vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError('grid_vectors must have shape (nG,3)')
    rcut = float(cell.rcut if rcut is None else rcut)
    if not np.isfinite(rcut) or rcut <= 0:
        raise ValueError('AO-pair Fourier strain rcut must be positive')

    lattice_vectors = np.asarray(
        cell.get_lattice_Ls(rcut=rcut), dtype=float)
    centers = _ao_centers(cell)
    nao = int(cell.nao_nr())
    value = np.zeros((len(vectors), nao, nao), dtype=np.complex128)
    derivative = np.zeros(
        (3, 3, len(vectors), nao, nao), dtype=np.complex128)
    for lattice in lattice_vectors:
        shifted = copy_with_coordinates(
            cell, cell.atom_coords() + lattice)
        combined = gto.conc_mol(cell, shifted)
        shell_slice = (0, cell.nbas, cell.nbas, 2 * cell.nbas)
        pair, center_i, center_j = _fourier_overlap_center_derivatives(
            combined, vectors, shell_slice)
        moment = molecular_ft_ao.ft_aopair(
            combined, vectors, shls_slice=shell_slice, aosym='s1',
            intor='GTO_ft_rc', comp=3)
        value += pair
        shifted_centers = centers + lattice
        image_response = -lib.einsum(
            'xGij,iy->xyGij', center_i, centers)
        image_response -= lib.einsum(
            'xGij,jy->xyGij', center_j, shifted_centers)
        image_response += 1j * lib.einsum(
            'Gx,yGij->xyGij', vectors, moment)
        derivative += image_response
    return {
        'value': _real_if_close(value),
        'raw_derivative': _real_if_close(derivative),
        'voigt_derivative': matrix_strain_to_voigt(
            _real_if_close(derivative)),
        'lattice_vectors': len(lattice_vectors),
        'rcut_bohr': rcut,
    }


def matrix_strain_to_voigt(tensor):
    '''Convert the first two strain axes of a matrix tensor to Voigt-6.'''
    tensor = np.asarray(tensor)
    symmetric = (tensor + tensor.swapaxes(0, 1).conj()) * .5
    return np.asarray((
        symmetric[0, 0], symmetric[1, 1], symmetric[2, 2],
        symmetric[1, 2], symmetric[0, 2], symmetric[0, 1],
    ))


def _ao_centers(cell):
    centers = np.empty((cell.nao_nr(), 3), dtype=float)
    for atom_id, aoslice in enumerate(cell.aoslice_by_atom()):
        start, stop = aoslice[2:]
        centers[start:stop] = cell.atom_coord(atom_id)
    return centers


def _fourier_cartesian_moment(mol, grid_vectors):
    r'''Return :math:`\int r_y\phi_Pe^{-iG\cdot r}` as ``(3,naux,nG)``.

    PySCF's molecular Fourier driver provides the Cartesian-moment integral
    through ``GTO_ft_rc``.  A normalized constant ghost shell supplies the
    second member of the AO-pair interface without changing the integral.
    '''
    ghost = gto.Mole()
    ghost._atm = np.asarray(((0, 0, 0, 0, 0, 0),), dtype=np.int32)
    ghost._bas = np.asarray(((0, 0, 1, 1, 0, 0, 3, 0),), dtype=np.int32)
    ghost._env = np.zeros(4)
    ghost._env[3] = np.sqrt(4 * np.pi)
    ghost._ecpbas = np.zeros((0, gto.BAS_SLOTS), dtype=np.int32)
    ghost.cart = bool(mol.cart)
    ghost._built = True

    combined = gto.conc_mol(mol, ghost)
    shell_slice = (0, mol.nbas, mol.nbas, mol.nbas + 1)
    moment = molecular_ft_ao.ft_aopair(
        combined, np.asarray(grid_vectors, dtype=float),
        shls_slice=shell_slice, aosym='s1', intor='GTO_ft_rc', comp=3)
    return np.asarray(moment)[:, :, :, 0].transpose(0, 2, 1)


def _fourier_overlap_center_derivatives(mol, grid_vectors, shell_slice):
    '''Evaluate Fourier overlap and i/j derivatives in one shell traversal.'''
    vectors = np.asarray(grid_vectors, dtype=float)
    vectors_t = np.asarray(vectors.T, order='C')
    extension = load_extension()
    callback = getattr(
        extension, mol._add_suffix('DCFORGE_ft_ovlp_ipij'))
    overlap_name = mol._add_suffix('GTO_ft_ovlp')
    ao_locations = moleintor.make_loc(mol._bas, overlap_name)
    i0, i1, j0, j1 = (int(value) for value in shell_slice)
    ni = int(ao_locations[i1] - ao_locations[i0])
    nj = int(ao_locations[j1] - ao_locations[j0])
    output = np.zeros(
        (7, nj, ni, len(vectors)), dtype=np.complex128)
    overlap_mask = np.ones(
        (i1 - i0, j1 - j0), dtype=np.int8, order='F')
    libcgto.GTO_ft_fill_drv(
        callback,
        libcgto.GTO_Gv_general,
        libcgto.GTO_ft_zfill_s1,
        output.ctypes.data_as(ctypes.c_void_p),
        overlap_mask.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(7),
        (ctypes.c_int * 4)(i0, i1, j0, j1),
        ao_locations.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_double(0),
        vectors_t.ctypes.data_as(ctypes.c_void_p),
        (ctypes.c_double * 1)(0),
        lib.c_null_ptr(),
        (ctypes.c_int * 3)(0, 0, 0),
        ctypes.c_int(len(vectors)),
        mol._atm.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(mol.natm),
        mol._bas.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(mol.nbas),
        mol._env.ctypes.data_as(ctypes.c_void_p),
    )
    output = np.rollaxis(output, -2, -3)
    output = np.rollaxis(output, -1, -3)
    return output[0], output[1:4], output[4:7]


def _real_if_close(value, tolerance=1.e-10):
    value = np.asarray(value)
    if value.size and np.max(np.abs(value.imag)) <= tolerance:
        return value.real
    return value
