r'''Long-range reciprocal-space three-center Gaussian-DF source.

At the Gamma point the long-range source is

.. math::

    B^{\mathrm{LR}}_{\mu\nu,P}
      = \sum_{\mathbf G} F_{\mu\nu}(\mathbf G)
        v_{\mathrm{LR}}(\mathbf G) P(\mathbf G)^*.

Its fixed-cell nuclear derivative has two independently testable terms: the
primary AO-pair center response and the auxiliary AO center response.
'''

import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.df import ft_ao, rsdf
from pyscf.pbc.tools import k2gamma

from dcforge.cpu.pbc.df.gaussian.ft import gen_ft_center_derivative_kernel


def long_range_source(cell, auxcell, omega, mesh, max_memory=None):
    r'''Return the Gamma-point source :math:`B^{\mathrm{LR}}`.

    Args:
        cell : pyscf.pbc.gto.Cell
            Primary AO cell.
        auxcell : pyscf.pbc.gto.Cell
            Auxiliary AO cell sharing the same atoms and lattice.
        omega : float
            Positive range-separation parameter in Bohr**-1.
        mesh : array-like of shape (3,)
            Fixed reciprocal mesh for the three-center source.

    Kwargs:
        max_memory : float or None
            Memory budget in MB.  The central AO-pair Fourier transform is
            accumulated in reciprocal-grid blocks bounded by this budget.

    Returns:
        ndarray of shape (nao*nao, naux)
            Long-range pre-Cholesky three-center source.
    '''
    Gv, Gvbase, reciprocal, gxyz, coulG, auxG = _context(
        cell, auxcell, omega, mesh)
    nao = cell.nao_nr()
    source = np.zeros((nao, nao, auxcell.nao_nr()), dtype=np.complex128)
    max_memory = _max_memory(cell, max_memory)
    gblksize = _grid_blksize(
        len(Gv), 16 * nao ** 2, max_memory, fraction=.65)
    for p0, p1 in lib.prange(0, len(Gv), gblksize):
        ao_pair = _ft_aopair_block(
            cell, Gv, Gvbase, reciprocal, gxyz, p0, p1)
        source += lib.einsum(
            'Gij,PG->ijP', ao_pair,
            (auxG[:, p0:p1] * coulG[p0:p1]).conj())
    return _real_if_close(source.reshape(cell.nao_nr() ** 2, -1))


def grad_long_range_source(
        cell, auxcell, omega, mesh, atmlst=None, max_memory=None):
    r'''Return the full derivative :math:`(B^{\mathrm{LR}})^\lambda`.

    Returns:
        ndarray of shape (len(atmlst), 3, nao*nao, naux)
            Sum of primary and auxiliary center responses.  Nuclear-center
            derivatives of the primary pair use the opposite sign from the
            electron-coordinate callbacks in :mod:`dcforge...gaussian.ft`.

    Kwargs:
        atmlst : sequence of int or None
            Nuclear centers returned by the derivative assembly.
        max_memory : float or None
            Memory budget in MB for reciprocal-grid blocking.
    '''
    source1 = gen_grad_long_range_source(
        cell, auxcell, omega, mesh, max_memory)
    return source1(atmlst)


def gen_grad_long_range_source(
        cell, auxcell, omega, mesh, max_memory=None):
    r'''Generate atom-block assembly of :math:`(B^{LR})^\lambda`.

    The BvK supermolecule, derivative kernels, auxiliary transforms, and
    Coulomb weights are independent of atom ownership and are built once.
    The returned function evaluates only the primary-AO shell rows and
    columns belonging to the requested atom block.  It therefore avoids the
    full :math:`(3,N_G,n_{AO},n_{AO})` i/j derivative tensors.

    The auxiliary-center term is contracted once into a compact
    ``(3,nao,nao,naux)`` basis-response tensor.  Atom blocks subsequently
    select only their auxiliary-function slices.  This ordering avoids both
    the full AO-pair Fourier cache and its repeated recomputation for every
    requested atom.  The AO-pair transform itself is streamed over
    reciprocal-grid blocks bounded by ``max_memory``.
    '''
    context = _context(cell, auxcell, omega, mesh)
    Gv, _Gvbase, _reciprocal, _gxyz, coulG, auxG = context
    supmol = _fourier_supermolecule(cell)
    kernel_i = gen_ft_center_derivative_kernel(
        supmol, 'i', return_complex=True)
    kernel_j = gen_ft_center_derivative_kernel(
        supmol, 'j', return_complex=True)
    weighted_aux = (auxG * coulG).conj()
    max_memory = _max_memory(cell, max_memory)
    ao_pair_bytes = 16 * len(Gv) * cell.nao_nr() ** 2
    auxiliary1 = _grad_auxiliary_basis_source(
        cell, auxcell, context, max_memory)

    def source1(atmlst=None):
        if atmlst is None:
            atmlst = range(cell.natm)
        atmlst = [int(ia) for ia in atmlst]
        value = _grad_primary_source(
            cell, context, atmlst, kernel_i, kernel_j, weighted_aux,
            max_memory)
        value += _grad_auxiliary_source(
            cell, auxcell, atmlst, auxiliary1)
        return _real_if_close(value)

    source1.fourier_aux_storage = 'basis_response_grid_stream'
    source1.fourier_ao_pair_bytes = int(ao_pair_bytes)
    return source1


def long_range_source_kpts(
        cell, auxcell, omega, mesh, kpti, kptj):
    r'''Return sampled-k :math:`B^{LR}(k_i,k_j)`.

    The AO-pair transform carries the absolute Bloch pair while the Coulomb
    kernel and auxiliary transform carry momentum transfer
    ``q = kptj-kpti``.
    '''
    context = _context_kpts(
        cell, auxcell, omega, mesh, kpti, kptj)
    Gv, Gvbase, reciprocal, gxyz, coulG, auxG, kpti, kptj, q = context
    ao_pair = ft_ao.ft_aopair(
        cell, Gv, aosym='s1', b=reciprocal, gxyz=gxyz,
        Gvbase=Gvbase, kpti_kptj=np.asarray((kpti, kptj)), q=q,
        intor='GTO_ft_ovlp', comp=1)
    return lib.einsum(
        'Gij,PG->ijP', ao_pair, (auxG * coulG).conj()).reshape(
            cell.nao_nr() ** 2, auxcell.nao_nr())


def grad_long_range_source_kpts(
        cell, auxcell, omega, mesh, kpti, kptj, atmlst=None):
    r'''Return sampled-k :math:`(B^{LR}(k_i,k_j))^\lambda`.

    Primary i/j-center Fourier derivatives use the package C callbacks while
    the auxiliary-center derivative is the analytic phase derivative
    ``-i(G+q) P(G+q)``.  All three contributions retain their complex Bloch
    phase until the final full-BZ energy contraction.
    '''
    if atmlst is None:
        atmlst = range(cell.natm)
    atmlst = [int(ia) for ia in atmlst]
    context = _context_kpts(
        cell, auxcell, omega, mesh, kpti, kptj)
    Gv, Gvbase, reciprocal, gxyz, coulG, auxG, kpti, kptj, q = context
    supmol = _fourier_supermolecule_kpts(cell, kptj)
    kernel_i = gen_ft_center_derivative_kernel(
        supmol, 'i', return_complex=True)
    kernel_j = gen_ft_center_derivative_kernel(
        supmol, 'j', return_complex=True)
    common = (Gv, gxyz, Gvbase)
    ip_i = kernel_i(
        *common, q=q, kptjs=np.asarray((kptj,)))[0]
    ip_j = kernel_j(
        *common, q=q, kptjs=np.asarray((kptj,)))[0]
    weighted_aux = (auxG * coulG).conj()
    nao = cell.nao_nr()
    naux = auxcell.nao_nr()
    value = np.zeros(
        (len(atmlst), 3, nao, nao, naux), dtype=np.complex128)
    aoslices = cell.aoslice_nr_by_atom()
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia, 2:]
        value[k, :, p0:p1] -= lib.einsum(
            'xGij,PG->xijP', ip_i[:, :, p0:p1], weighted_aux)
        value[k, :, :, p0:p1] -= lib.einsum(
            'xGij,PG->xijP', ip_j[:, :, :, p0:p1], weighted_aux)

    ao_pair = ft_ao.ft_aopair(
        cell, Gv, aosym='s1', b=reciprocal, gxyz=gxyz,
        Gvbase=Gvbase, kpti_kptj=np.asarray((kpti, kptj)), q=q,
        intor='GTO_ft_ovlp', comp=1)
    momentum = Gv + q
    auxslices = auxcell.aoslice_nr_by_atom()
    for k, ia in enumerate(atmlst):
        p0, p1 = auxslices[ia, 2:]
        for xyz in range(3):
            auxG1 = (
                -1j * momentum[:, xyz][None] * auxG[p0:p1])
            value[k, xyz, :, :, p0:p1] += lib.einsum(
                'Gij,PG->ijP', ao_pair,
                (auxG1 * coulG).conj())
    return _real_if_close(
        value.reshape(len(atmlst), 3, nao * nao, naux))


def _grad_primary_source(
        cell, context, atmlst, kernel_i, kernel_j, weighted_aux,
        max_memory):
    r'''Return shell-blocked primary AO-center :math:`(B^{LR})^\lambda`.

    The i-center kernel needs only ``(shells_A, all shells)`` and the
    j-center kernel only ``(all shells, shells_A)``.  Consecutive requested
    atoms are combined into one shell traversal; sparse or reordered atom
    lists remain correct by starting a new group.
    '''
    Gv, Gvbase, _reciprocal, gxyz, _coulG, auxG = context
    nao = cell.nao_nr()
    naux = auxG.shape[0]
    source1 = np.zeros(
        (len(atmlst), 3, nao, nao, naux), dtype=np.complex128)
    aoslices = cell.aoslice_nr_by_atom()
    kptjs = np.zeros((1, 3))
    nbas = cell.nbas

    for positions, atoms in _atom_groups(atmlst):
        ish0 = aoslices[atoms[0], 0]
        ish1 = aoslices[atoms[-1], 1]
        pbase = aoslices[atoms[0], 2]
        nrow = aoslices[atoms[-1], 3] - pbase
        rows = np.zeros((3, nrow, nao, naux), dtype=np.complex128)
        columns = np.zeros((3, nao, nrow, naux), dtype=np.complex128)
        bytes_per_grid = 3 * nrow * nao * 16
        gblksize = _grid_blksize(
            len(Gv), bytes_per_grid, max_memory, fraction=.2)

        for g0, g1 in lib.prange(0, len(Gv), gblksize):
            common = (Gv[g0:g1], gxyz[g0:g1], Gvbase)
            ip_i = kernel_i(
                *common, kptjs=kptjs,
                shls_slice=(ish0, ish1, 0, nbas))[0]
            rows += lib.einsum(
                'xGij,PG->xijP', ip_i, weighted_aux[:, g0:g1])
            del ip_i

            ip_j = kernel_j(
                *common, kptjs=kptjs,
                shls_slice=(0, nbas, ish0, ish1))[0]
            columns += lib.einsum(
                'xGij,PG->xijP', ip_j, weighted_aux[:, g0:g1])
            del ip_j

        for k, ia in zip(positions, atoms, strict=True):
            p0, p1 = aoslices[ia, 2:]
            q0, q1 = p0 - pbase, p1 - pbase
            source1[k, :, p0:p1] -= rows[:, q0:q1]
            source1[k, :, :, p0:p1] -= columns[:, :, q0:q1]
    return source1.reshape(len(atmlst), 3, nao * nao, naux)


def _atom_groups(atmlst):
    '''Yield position/atom groups that are consecutive in the cell.'''
    if not atmlst:
        return
    positions = [0]
    atoms = [atmlst[0]]
    for k, ia in enumerate(atmlst[1:], 1):
        if ia == atoms[-1] + 1:
            positions.append(k)
            atoms.append(ia)
        else:
            yield positions, atoms
            positions = [k]
            atoms = [ia]
    yield positions, atoms


def _ft_aopair_block(cell, Gv, Gvbase, reciprocal, gxyz, p0, p1):
    '''Return one reciprocal-grid block of the central AO-pair transform.'''
    return ft_ao.ft_aopair(
        cell, Gv[p0:p1], aosym='s1', b=reciprocal,
        gxyz=gxyz[p0:p1], Gvbase=Gvbase,
        kpti_kptj=np.zeros((2, 3)), q=np.zeros(3),
        intor='GTO_ft_ovlp', comp=1,
    )


def _max_memory(cell, max_memory):
    '''Return a positive PySCF-style memory budget in MB.'''
    if max_memory is None:
        max_memory = getattr(cell, 'max_memory', 4000)
    max_memory = float(max_memory)
    if max_memory <= 0:
        raise ValueError('long-range source max_memory must be positive')
    return max_memory


def _grid_blksize(ngrid, bytes_per_grid, max_memory, fraction):
    '''Return a reciprocal-grid block bounded by a memory-budget fraction.'''
    remaining = max(0., float(max_memory) - lib.current_memory()[0])
    budget = max(remaining * 1e6 * float(fraction), bytes_per_grid)
    return max(1, min(int(ngrid), int(budget // bytes_per_grid)))


def _grad_auxiliary_basis_source(
        cell, auxcell, context, max_memory):
    r'''Contract the LR auxiliary-center derivative once per basis function.

    Each auxiliary function belongs to exactly one atom.  Keeping its three
    Cartesian derivative components therefore contains all atom ownership
    information without materializing a separate ``3N`` source.  The much
    larger central AO-pair Fourier transform is consumed and discarded one
    reciprocal-grid block at a time.
    '''
    Gv, Gvbase, reciprocal, gxyz, coulG, auxG = context
    nao = cell.nao_nr()
    naux = auxcell.nao_nr()
    auxiliary1 = np.zeros(
        (3, nao, nao, naux), dtype=np.complex128)
    bytes_per_grid = 16 * (nao ** 2 + naux)
    gblksize = _grid_blksize(
        len(Gv), bytes_per_grid, max_memory, fraction=.65)

    for g0, g1 in lib.prange(0, len(Gv), gblksize):
        pair = _ft_aopair_block(
            cell, Gv, Gvbase, reciprocal, gxyz, g0, g1)
        for xyz in range(3):
            # dP(G)/dR_A = -i G_x P(G).  The source uses P(G)^*, so
            # conjugation below also fixes the phase sign.
            auxG1 = (
                -1j * Gv[g0:g1, xyz][None]
                * auxG[:, g0:g1])
            auxiliary1[xyz] += lib.einsum(
                'Gij,PG->ijP', pair,
                (auxG1 * coulG[g0:g1]).conj())
    return auxiliary1


def _grad_auxiliary_source(cell, auxcell, atmlst, auxiliary1):
    r'''Assemble atom blocks from the auxiliary-basis LR response.'''
    nao = cell.nao_nr()
    naux = auxcell.nao_nr()
    source1 = np.zeros(
        (len(atmlst), 3, nao, nao, naux), dtype=auxiliary1.dtype)
    aoslices = auxcell.aoslice_nr_by_atom()
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia, 2:]
        source1[k, :, :, :, p0:p1] = auxiliary1[:, :, :, p0:p1]
    return source1.reshape(len(atmlst), 3, nao * nao, naux)


def _context(cell, auxcell, omega, mesh):
    mesh = _mesh(mesh)
    Gv, Gvbase, _weights = cell.get_Gv_weights(mesh)
    reciprocal = cell.reciprocal_vectors()
    gxyz = lib.cartesian_prod(
        [np.arange(len(values)) for values in Gvbase])

    with_df = rsdf.RSGDF(cell, kpts=np.zeros((1, 3)))
    with_df.auxcell = auxcell
    coulG = with_df.weighted_coulG(np.zeros(3), mesh=mesh)
    coulG -= with_df.weighted_coulG(
        np.zeros(3), mesh=mesh, omega=-abs(float(omega)))
    auxG = ft_ao.ft_ao(
        auxcell, Gv, None, reciprocal, gxyz, Gvbase, np.zeros(3)).T
    return Gv, Gvbase, reciprocal, gxyz, coulG, auxG


def _fourier_supermolecule(cell):
    log = logger.new_logger(cell, 0)
    rs_cell = ft_ao._RangeSeparatedCell.from_cell(
        cell, ft_ao.KECUT_THRESHOLD, ft_ao.RCUT_THRESHOLD, log)
    bvk_mesh = k2gamma.kpts_to_kmesh(cell, np.zeros((1, 3)))
    rcut = ft_ao.estimate_rcut(rs_cell)
    supmol = ft_ao.ExtendedMole.from_cell(
        rs_cell, bvk_mesh, rcut.max(), log)
    return supmol.strip_basis(rcut)


def _context_kpts(cell, auxcell, omega, mesh, kpti, kptj):
    '''Return reciprocal intermediates for one absolute Bloch pair.'''
    mesh = _mesh(mesh)
    kpti = _kpoint(kpti)
    kptj = _kpoint(kptj)
    q = kptj - kpti
    Gv, Gvbase, _weights = cell.get_Gv_weights(mesh)
    reciprocal = cell.reciprocal_vectors()
    gxyz = lib.cartesian_prod(
        [np.arange(len(values)) for values in Gvbase])
    with_df = rsdf.RSGDF(cell, kpts=np.asarray((kpti, kptj)))
    with_df.auxcell = auxcell
    coulG = with_df.weighted_coulG(q, mesh=mesh)
    coulG -= with_df.weighted_coulG(
        q, mesh=mesh, omega=-abs(float(omega)))
    auxG = ft_ao.ft_ao(
        auxcell, Gv, None, reciprocal, gxyz, Gvbase, q).T
    return (
        Gv, Gvbase, reciprocal, gxyz, coulG, auxG,
        kpti, kptj, q)


def _fourier_supermolecule_kpts(cell, kptj):
    '''Return PySCF's BvK Fourier supermolecule for one ket k-point.'''
    log = logger.new_logger(cell, 0)
    rs_cell = ft_ao._RangeSeparatedCell.from_cell(
        cell, ft_ao.KECUT_THRESHOLD, ft_ao.RCUT_THRESHOLD, log)
    bvk_mesh = k2gamma.kpts_to_kmesh(cell, np.asarray((kptj,)))
    rcut = ft_ao.estimate_rcut(rs_cell)
    supmol = ft_ao.ExtendedMole.from_cell(
        rs_cell, bvk_mesh, rcut.max(), log)
    return supmol.strip_basis(rcut)


def _kpoint(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,):
        raise ValueError(f'k-point must have shape (3,), got {value.shape}')
    return value


def _mesh(mesh):
    mesh = np.asarray(mesh, dtype=int)
    if mesh.shape != (3,) or np.any(mesh <= 0):
        raise ValueError(
            f'mesh must contain three positive integers, got {mesh}')
    return mesh


def _real_if_close(value, tol=1e-10):
    value = np.asarray(value)
    if value.size and np.max(np.abs(value.imag)) <= tol:
        return value.real
    return value
