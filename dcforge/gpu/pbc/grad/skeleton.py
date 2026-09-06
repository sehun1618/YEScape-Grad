'''Explicit Gamma-point GPU semilocal-DFA@RHF gradient terms.'''

import time
from types import SimpleNamespace

import cupy as cp
import numpy as np
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.pbc.dft import multigrid_v2, numint
from gpu4pyscf.pbc.grad import krhf, krks
from gpu4pyscf.pbc.grad.pp import vppnl_nuc_grad
from gpu4pyscf.pbc.gto import int1e
from pyscf.pbc.gto.pseudo.pp import get_alphas, get_vlocG

from dcforge.gpu.pbc.df.gaussian.nonlocal_pseudo import (
    contract_nonlocal_pseudo_batch,
)
from dcforge.gpu.pbc.grad.multigrid import get_veff_deriv as get_multigrid_veff_deriv

# Match GPU4PySCF ``KNumInt.block_loop``'s upper grid-block bound while
# retaining one AO evaluator across all atom batches below.  The latter is
# essential at high cutoff: recreating the private evaluator for every atom
# batch retains device workspaces until Python/CUDA teardown.
_HCORE_GRID_BLOCK_SIZE = 16384
# The reciprocal local-potential contraction has no AO-sized dimension.
# A much wider G block therefore amortizes einsum dispatch while retaining a
# bounded ``(ndensity,natom_batch,nG)`` temporary (128 MiB for the production
# 2-density/4-atom batch in complex128).
_HCORE_RECIPROCAL_BLOCK_SIZE = 1024 * 1024
# The local-pseudopotential FFT is complex even though its physical real-space
# field is real.  At high cutoff, ``fft_output.real`` spans more than 2 GiB for
# a modest atom batch.  Generic CuPy kernels may select 32-bit indexing for
# that stride-16 view and fail before the much smaller contracted result is
# produced.  The reduction below reads the complex field directly with 64-bit
# grid indices, so no full real copy or transposed grid tensor is required.
_HCORE_LOCAL_REDUCTION_THREADS = 256
_HCORE_LOCAL_REDUCTION_MAX_SPLITS = 64
_HCORE_LOCAL_REDUCTION_KERNEL = cp.RawKernel(r'''
extern "C" __global__
void contract_local_density(const double* rho, const double2* vloc,
                            double* partial, const long long ngrids,
                            const int ncomponent, const int nsplit)
{
    const int pair = blockIdx.x / nsplit;
    const int split = blockIdx.x - pair * nsplit;
    const int density = pair / ncomponent;
    const int component = pair - density * ncomponent;
    const long long rho0 = (long long)density * ngrids;
    const long long vloc0 = (long long)component * ngrids;
    const long long stride = (long long)nsplit * blockDim.x;
    long long grid = (long long)split * blockDim.x + threadIdx.x;
    double value = 0.0;

    for (; grid < ngrids; grid += stride) {
        value += rho[rho0 + grid] * vloc[vloc0 + grid].x;
    }

    __shared__ double work[256];
    work[threadIdx.x] = value;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            work[threadIdx.x] += work[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = work[0];
    }
}
''', 'contract_local_density')
# ``krhf.hcore_generator`` constructs a complex structure-factor array for
# every atom before it can return even one H1 derivative.  The array alone is
# ``natm * ngrid * sizeof(complex128)``.  Retaining that route below this
# bound preserves GPU4PySCF's public matrix implementation where it is
# compact; above it, attempting construction first would turn a predictable
# direct-contraction calculation into an allocation failure.
_HCORE_STREAMING_SI_LIMIT_BYTES = 2 * 1024**3


def _hcore_profile_start(profile):
    '''Synchronize and start one optional Hcore backend stage timer.'''
    if profile is None:
        return None
    cp.cuda.Stream.null.synchronize()
    return time.perf_counter()


def _hcore_profile_stop(profile, name, start):
    '''Synchronize and retain one optional Hcore backend stage time.'''
    if start is None:
        return
    cp.cuda.Stream.null.synchronize()
    profile[name] = time.perf_counter() - start


def should_stream_hcore(cell):
    '''Return whether GDF Hcore must avoid all-atom structure factors.

    This is a representation choice, never a numerical approximation.  The
    threshold measures only the mandatory complex ``SI[atom, G]`` array in
    GPU4PySCF's public generator; other provider temporaries make the public
    route still more expensive.  Callers retain the same finite grid,
    pseudopotential, and Hcore derivative on both sides of the threshold.

    Args:
        cell : pyscf.pbc.gto.Cell
            Periodic cell whose resolved ``mesh`` defines the reciprocal grid.

    Returns:
        bool
            ``True`` when the density-contracted streamed implementation is
            required to keep the all-atom public intermediate bounded.
    '''
    ngrid = int(np.prod(np.asarray(cell.mesh, dtype=np.int64)))
    si_bytes = int(cell.natm) * ngrid * np.dtype(np.complex128).itemsize
    return si_bytes > _HCORE_STREAMING_SI_LIMIT_BYTES


def effective_hcore_batch_size(cell, atom_batch_size, *, streamed):
    '''Return a safe atom batch size for the selected Hcore representation.

    ``atom_batch_size`` is preserved in both representations.  In the
    large-mesh streamed route it controls only the number of atom-specific
    local-GTH FFT fields held together; AO values and row corrections are
    evaluated in one shared grid traversal independent of this batch.
    '''
    size = int(atom_batch_size)
    if size < 1:
        raise ValueError('atom_batch_size must be positive')
    return size


def _contract_local_density(rho, vloc_fft):
    r'''Contract real local-pseudopotential FFT fields with AO densities.

    This helper evaluates

    .. math::

       L_{d,A x} = \sum_g \rho_d(g)\,\Re[v^\lambda_{A x}(g)]

    without materializing ``vloc_fft.real``.  The latter is a strided view of
    a potentially larger-than-2-GiB complex array at production meshes.  The
    CUDA reduction uses 64-bit offsets and stores only a small split partial
    sum for each ``(density, atom, Cartesian)`` output.

    Args:
        rho : cupy.ndarray of shape (ndensity, ngrid)
            Real scalar AO density fields in physical FFT-grid order.
        vloc_fft : cupy.ndarray of shape (natom, 3, ngrid)
            Complex inverse-FFT output for the atom-specific local GTH
            derivative potentials.

    Returns:
        cupy.ndarray of shape (ndensity, natom, 3)
            Density-contracted local-pseudopotential derivatives.
    '''
    rho = cp.ascontiguousarray(rho, dtype=cp.float64)
    vloc_fft = cp.ascontiguousarray(vloc_fft, dtype=cp.complex128)
    ndensity, ngrids = rho.shape
    natm = vloc_fft.shape[0]
    ncomponent = natm * 3
    grids_per_split = _HCORE_LOCAL_REDUCTION_THREADS * 512
    nsplit = min(
        _HCORE_LOCAL_REDUCTION_MAX_SPLITS,
        max(1, (ngrids + grids_per_split - 1) // grids_per_split))
    noutput = ndensity * ncomponent
    partial = cp.empty((noutput, nsplit), dtype=cp.float64)
    _HCORE_LOCAL_REDUCTION_KERNEL(
        (noutput * nsplit,), (_HCORE_LOCAL_REDUCTION_THREADS,),
        (rho, vloc_fft, partial, np.int64(ngrids),
         np.int32(ncomponent), np.int32(nsplit)))
    return cp.sum(partial, axis=1).reshape(ndensity, natm, 3)


def _contract_local_density_reciprocal(rho_g, si, vloc, gv,
                                       block_size=_HCORE_RECIPROCAL_BLOCK_SIZE):
    r'''Contract local-GTH derivative potentials without inverse FFTs.

    For GPU4PySCF's FFT convention,

    .. math::

       \sum_r \rho_d(r) v^\lambda_{A x}(r)
       = {1\over N_G}\sum_G \rho_d(G)^* v^\lambda_{A x}(G),

    where ``vloc_G[A,x,G] = i G_x SI[A,G] vloc[A,G]``.  Applying this
    discrete Parseval identity replaces three inverse FFTs per atom by one
    reciprocal contraction.  G vectors are streamed so the full
    ``(natom,3,ngrid)`` derivative potential is never materialized.

    Args:
        rho_g : cupy.ndarray of shape (ndensity, ngrid)
            Unnormalised forward FFT of real AO density fields.
        si, vloc : cupy.ndarray of shape (natom, ngrid)
            Atomic structure factors and radial local-GTH vectors.
        gv : cupy.ndarray of shape (ngrid, 3)
            Reciprocal vectors in PySCF FFT ordering.

    Kwargs:
        block_size : int
            Number of reciprocal vectors in one bounded contraction.

    Returns:
        cupy.ndarray
            Density-contracted local-potential derivatives with shape
            ``(ndensity,natom,3)``.
    '''
    rho_g = cp.asarray(rho_g)
    si = cp.asarray(si)
    vloc = cp.asarray(vloc)
    gv = cp.asarray(gv)
    if rho_g.ndim != 2 or si.ndim != 2 or vloc.shape != si.shape:
        raise ValueError('invalid reciprocal local-density contraction shape')
    if rho_g.shape[1] != si.shape[1] or gv.shape != (si.shape[1], 3):
        raise ValueError('reciprocal local-density grids do not match')
    block_size = int(block_size)
    if block_size < 1:
        raise ValueError('reciprocal block_size must be positive')

    ngrids = si.shape[1]
    de = cp.zeros((len(rho_g), len(si), 3), dtype=cp.float64)
    for grid0 in range(0, ngrids, block_size):
        grid1 = min(ngrids, grid0 + block_size)
        de += cp.einsum(
            'dg,Ag,Ag,gx->dAx', rho_g[:, grid0:grid1].conj(),
            1j * si[:, grid0:grid1], vloc[:, grid0:grid1],
            gv[grid0:grid1], optimize=True).real / ngrids
    return de


def get_hcore_generator(cell):
    '''Return GPU local kinetic plus GTH local-potential derivative matrices.'''
    adapter = SimpleNamespace(base=SimpleNamespace(with_x2c=None))
    return krhf.hcore_generator(adapter, cell, np.zeros((1, 3)))


def contract_hcore(cell, hcore_gen, dm, atmlst):
    '''Return ``Tr[P h^lambda]`` including nonlocal GTH in Ha/Bohr.

    ``hcore_gen`` supplies kinetic plus local GTH matrices.  GPU4PySCF's
    nonlocal-GTH primitive is already atom-contracted, so it is added only
    after the local matrix contraction.  Both paths remain on the GPU until
    their small atom-by-Cartesian result is returned.
    '''
    de = cp.empty((len(atmlst), 3), dtype=cp.float64)
    for k, ia in enumerate(atmlst):
        h1ao = hcore_gen(ia)[0]
        de[k] = cp.einsum('xij,ji->x', h1ao, dm).real
    de += cp.asarray(vppnl_nuc_grad(cell, dm))[atmlst]
    return de


def contract_hcore_pair(cell, hcore_gen, dm0, dz, atmlst,
                        atom_batch_size=1, profile=None):
    r'''Contract one local ``h^lambda`` generation with ``P`` and ``D_Z``.

    The Gamma GDF DC-DFT gradient needs two one-electron contractions with
    the *same* nuclear derivative matrix:

    .. math::

       h^\lambda : P,\qquad -2 h^\lambda : D_Z.

    GPU4PySCF's public ``hcore_generator`` constructs the local GTH matrix
    one atom at a time.  Calling it once for each density repeats the same
    FFT/grid integration without changing either mathematical term.  This
    helper keeps the original provider primitive, generates each atom's
    matrix once, and contracts the two Hermitian densities together.  The
    nonlocal GTH endpoint is already density-contracted; it is invoked once
    per density because its current GPU4PySCF API has no density-set axis.

    Args:
        cell : pyscf.pbc.gto.Cell
            Active real-Gamma periodic cell.
        hcore_gen : callable or None
            GPU4PySCF local kinetic plus GTH-local derivative generator.
            ``None`` selects the GDF large-mesh density-contracted
            representation; it preserves the same local finite-grid algebra
            but avoids first materializing all-atom structure factors.
        dm0, dz : cupy.ndarray of shape (nao, nao)
            Source RHF density ``P`` and Hermitian Z transition density
            ``D_Z`` in the physical AO order.
        atmlst : sequence of int
            Nuclear centers in the requested output order.

    Kwargs:
        atom_batch_size : int
            Number of nuclear centers whose local GTH grid derivatives are
            formed together.  ``1`` calls the public GPU4PySCF generator
            verbatim and is the reference path.  Larger values reuse each AO
            grid block across several centers without materializing a full
            ``(3*natm, nao, nao)`` H1 tensor.  With ``hcore_gen=None``, AO
            densities and atom-row terms are already shared cell-wide, so the
            value controls only bounded atom-specific local-GTH FFT storage.
        profile : dict or None
            Optional validation-only stage timing destination.  Supplying a
            dictionary synchronizes the active stream at Hcore substage
            boundaries; production should retain ``None``.

    Returns:
        tuple of cupy.ndarray
            ``(de_hcore, de_fock_hcore)`` in Ha/Bohr.  The second value
            already includes the source-Fock coefficient ``-2``.
    '''
    atom_batch_size = int(atom_batch_size)
    if atom_batch_size < 1:
        raise ValueError('atom_batch_size must be positive')
    if hcore_gen is None:
        return _contract_hcore_pair_streamed(
            cell, dm0, dz, atmlst, atom_batch_size, profile=profile)
    if atom_batch_size > 1:
        return _contract_hcore_pair_batched(
            cell, dm0, dz, atmlst, atom_batch_size)

    densities = cp.stack((cp.asarray(dm0), cp.asarray(dz)))
    de = cp.empty((2, len(atmlst), 3), dtype=cp.float64)
    for k, ia in enumerate(atmlst):
        h1ao = hcore_gen(ia)[0]
        de[:, k] = cp.einsum('xij,dji->dx', h1ao, densities).real

    # ``vppnl_nuc_grad`` returns NumPy and accepts a single density.  Keeping
    # these two calls explicit avoids claiming an unsupported batch contract.
    de += contract_nonlocal_pseudo_batch(cell, densities)[:, atmlst]
    return de[0], -2 * de[1]


def _species_vloc_g(cell, gv_cpu):
    '''Build one local-GTH radial vector per atomic symbol, not per atom.

    ``get_vlocG`` is radial: atoms with the same GTH pseudo share exactly the
    same vector and differ only by their structure factor.  GPU4PySCF's
    upstream Hcore generator requests every atom at once, which would create
    a ``(natm, nG)`` CPU/GPU array.  The small single-atom probe cells retain
    the identical PySCF pseudo evaluator while bounding this storage by the
    number of chemical species.
    '''
    vectors = {}
    for ia in range(cell.natm):
        symbol = cell.atom_symbol(ia)
        if symbol in vectors:
            continue
        probe = cell.copy()
        probe.atom = [(symbol, (0., 0., 0.))]
        probe.build(False, False)
        vectors[symbol] = cp.asarray(get_vlocG(probe, gv_cpu)[0])
    return vectors


def _make_rho_and_local_hcore_rows(densities, ao_all, weight):
    r'''Build scalar densities and local-potential AO-row responses.

    Define the density-weighted AO column

    .. math::

       Y_{d,i}(g) = \sum_j D^d_{ji}\phi_j(g).

    The scalar density is ``rho_d = sum_i phi_i^* Y_d,i``.  The two AO-center
    sides of the local-potential derivative normally require both ``D^T phi``
    and ``D phi``.  For the supported real-Gamma path, ``phi`` and its first
    derivative are real and each ``D`` is real symmetric.  Consequently

    .. math::

       D\phi = D^T\phi = Y,

    and the two row/column contributions are identical.  Production therefore
    evaluates one density-AO contraction and doubles its row response.  If an
    upstream representation supplies complex arrays, this helper retains the
    original two-sided Hermitian expression instead of applying the real-Gamma
    identity outside its contract.

    Args:
        densities : cupy.ndarray of shape (ndensity, nao, nao)
            Source density ``P`` and Z transition density ``D_Z``.
        ao_all : cupy.ndarray of shape (nkpt, 4, nao, ngrid)
            AO values followed by their three Cartesian derivatives.
        weight : cupy.ndarray of shape (ngrid,)
            Complete local pseudopotential on the active grid block.

    Returns:
        tuple of cupy.ndarray
            Scalar density fields ``(ndensity, ngrid)`` and AO-row responses
            ``(ndensity, 3, nao)`` for the local-potential derivative.
    '''
    ao = ao_all[:, 0]
    dao_t = contract('dji,kjg->dkig', densities, ao)
    rho = cp.einsum('kig,dkig->dg', ao.conj(), dao_t).real
    deriv = ao_all[:, 1:]

    if not cp.iscomplexobj(densities) and not cp.iscomplexobj(ao_all):
        row_by_ao = 2 * cp.sum(
            deriv[None] * dao_t[:, :, None]
            * weight[None, None, None, None], axis=(1, 4))
    else:
        dao = contract('dij,kjg->dkig', densities, ao)
        first = cp.sum(
            deriv.conj()[None] * dao_t[:, :, None]
            * weight[None, None, None, None], axis=(1, 4))
        second = cp.sum(
            deriv[None] * dao.conj()[:, :, None]
            * weight[None, None, None, None], axis=(1, 4))
        row_by_ao = first + second
    return rho, row_by_ao.real


def _contract_hcore_pair_streamed(cell, dm0, dz, atmlst, atom_batch_size,
                                  profile=None):
    r'''Contract paired Hcore derivatives with one shared AO-grid traversal.

    The two scalar AO density fields required by ``P`` and ``D_Z`` are built
    once on the resolved mesh.  In the same traversal, all atom-row Pulay
    corrections are accumulated through an AO-to-atom ownership map.  The
    atom-specific local-GTH derivative potentials are then FFT transformed
    in bounded batches and contracted with the stored scalar density fields.

    This is algebraically identical to
    :func:`_contract_hcore_pair_streamed_atom_batched`, which revisits the AO
    grid once per atom batch.  The one-pass representation removes that
    repeated work and avoids its large ``(atom_batch,3,ngrid)`` lifetime from
    overlapping AO evaluator workspaces.  It does not alter ``cell.mesh``,
    pseudopotential values, or the returned term ledger.
    '''
    total_start = _hcore_profile_start(profile)
    stage_start = _hcore_profile_start(profile)
    kpts = np.zeros((1, 3))
    gv_cpu = np.asarray(cell.Gv)
    gv = cp.asarray(gv_cpu)
    vloc_by_symbol = _species_vloc_g(cell, gv_cpu)
    grids = krhf.UniformGrids(cell)
    ni = numint.KNumInt()
    eval_gto_opt = numint._GTOvalOpt(cell, kpts, deriv=1)
    grid_order = grids.argsort(tile=8)
    grid_coords = grids.coords[grid_order]
    aoslices = np.asarray(cell.aoslice_by_atom(), dtype=int)
    atmlst = list(atmlst)
    densities = cp.stack((cp.asarray(dm0), cp.asarray(dz)))
    nao = int(cell.nao_nr())
    ngrids = int(np.prod(np.asarray(cell.mesh, dtype=np.int64)))

    # Accumulate the complete local pseudopotential once.  Only chemical
    # species own radial ``vloc`` values; atoms enter through structure
    # factors and are kept in the requested bounded FFT batch.
    vpploc_g = cp.zeros(ngrids, dtype=cp.complex128)
    for atom0 in range(0, cell.natm, atom_batch_size):
        atom1 = min(cell.natm, atom0 + atom_batch_size)
        atoms = np.arange(atom0, atom1, dtype=int)
        si = krhf.get_SI(cell, atmlst=atoms)
        for iatom, ia in enumerate(atoms):
            vpploc_g -= si[iatom] * vloc_by_symbol[cell.atom_symbol(ia)]
    vpploc_g[0] = cp.sum(cp.asarray(get_alphas(cell)))
    vpploc_r = krhf.tools.ifft(vpploc_g, cell.mesh).real
    _hcore_profile_stop(
        profile, 'common_local_potential_seconds', stage_start)

    # ``owner[A,p]`` maps each AO row to its nuclear center.  It allows the
    # row correction to be reduced for every requested atom in one GPU
    # contraction per grid block, rather than launching one small reduction
    # for each atom and atom batch.
    owner = cp.zeros((len(atmlst), nao), dtype=cp.float64)
    for iatom, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia, 2:]
        owner[iatom, p0:p1] = 1

    stage_start = _hcore_profile_start(profile)
    rho = cp.empty((2, ngrids), dtype=cp.float64)
    row_de = cp.zeros((2, len(atmlst), 3), dtype=cp.float64)
    for grid0 in range(0, ngrids, _HCORE_GRID_BLOCK_SIZE):
        grid1 = min(ngrids, grid0 + _HCORE_GRID_BLOCK_SIZE)
        ao_all = ni.eval_ao(
            cell, grid_coords[grid0:grid1], kpts, deriv=1,
            opt=eval_gto_opt).transpose(0, 1, 3, 2)
        grid_indices = grid_order[grid0:grid1]
        weight = vpploc_r[grid_indices]
        rho_block, row_by_ao = _make_rho_and_local_hcore_rows(
            densities, ao_all, weight)
        rho[:, grid_indices] = rho_block
        row_de += cp.einsum('dxp,Ap->dAx', row_by_ao, owner)
    _hcore_profile_stop(profile, 'ao_density_rows_seconds', stage_start)

    # Kinetic row/column corrections have no grid dependence and are reduced
    # through the same ownership map.  Their contribution completes the
    # common kinetic-plus-local-pseudopotential H1 derivative.
    stage_start = _hcore_profile_start(profile)
    hkin = int1e.int1e_ipkin(cell, kpts)[0]
    hkin_by_ao = (
        cp.einsum('xpq,dqp->dxp', hkin, densities)
        + cp.einsum('xpq,dpq->dxp', hkin.conj(), densities)
    ).real
    row_de += cp.einsum('dxp,Ap->dAx', hkin_by_ao, owner)
    _hcore_profile_stop(profile, 'kinetic_seconds', stage_start)

    stage_start = _hcore_profile_start(profile)
    rho_g = krhf.tools.fft(rho, cell.mesh)
    local_de = cp.zeros_like(row_de)
    for atom0 in range(0, len(atmlst), atom_batch_size):
        atom1 = min(atom0 + atom_batch_size, len(atmlst))
        atoms = np.asarray(atmlst[atom0:atom1], dtype=int)
        si = krhf.get_SI(cell, atmlst=atoms)
        vloc = cp.stack([
            vloc_by_symbol[cell.atom_symbol(ia)] for ia in atoms])
        local_de[:, atom0:atom1] = _contract_local_density_reciprocal(
            rho_g, si, vloc, gv)
    _hcore_profile_stop(profile, 'local_center_seconds', stage_start)

    output = local_de - row_de
    stage_start = _hcore_profile_start(profile)
    output += contract_nonlocal_pseudo_batch(cell, densities)[:, atmlst]
    _hcore_profile_stop(profile, 'nonlocal_seconds', stage_start)
    _hcore_profile_stop(profile, 'total_seconds', total_start)
    return output[0], -2 * output[1]


def _contract_hcore_pair_streamed_atom_batched(
        cell, dm0, dz, atmlst, atom_batch_size):
    r'''Directly contract paired Hcore derivatives without all-atom FFT data.

    This is the pre-one-pass validation reference.  It evaluates the same
    GPU4PySCF finite-grid algebra but revisits AO values for every atom batch.
    Production calls :func:`_contract_hcore_pair_streamed`; validation may
    compare both representations before retiring this slower implementation.

    The public result remains exactly ``(h^lambda:P, -2 h^lambda:D_Z)``;
    only the intermediate representation changes from H1 matrices to
    density-weighted AO grid values.
    '''
    kpts = np.zeros((1, 3))
    gv_cpu = np.asarray(cell.Gv)
    gv = cp.asarray(gv_cpu)
    vloc_by_symbol = _species_vloc_g(cell, gv_cpu)
    grids = krhf.UniformGrids(cell)
    ni = numint.KNumInt()
    eval_gto_opt = numint._GTOvalOpt(cell, kpts, deriv=1)
    grid_order = grids.argsort(tile=8)
    grid_coords = grids.coords[grid_order]
    aoslices = np.asarray(cell.aoslice_by_atom(), dtype=int)
    atmlst = list(atmlst)
    densities = cp.stack((cp.asarray(dm0), cp.asarray(dz)))
    output = cp.empty((2, len(atmlst), 3), dtype=cp.float64)

    # The common local pseudopotential in upstream ``get_hcore`` is the sum
    # over atoms.  Accumulate that one reciprocal vector in small atom
    # batches; its G=0 convention follows GPU4PySCF verbatim.
    vpploc_g = cp.zeros(gv.shape[0], dtype=cp.complex128)
    for atom0 in range(0, cell.natm, atom_batch_size):
        atom1 = min(cell.natm, atom0 + atom_batch_size)
        atoms = np.arange(atom0, atom1, dtype=int)
        # Do not pass the flattened G-vector matrix here.  GPU4PySCF's
        # mesh-factorized ``get_SI(cell, atmlst=...)`` is algebraically the
        # same structure factor as ``exp(-i R.G)`` but avoids a very wide
        # ``coords @ G.T`` GEMM on high-cutoff meshes.
        si = krhf.get_SI(cell, atmlst=atoms)
        vloc = cp.stack([
            vloc_by_symbol[cell.atom_symbol(ia)] for ia in atoms])
        vpploc_g -= cp.einsum('Ag,Ag->g', si, vloc)
    vpploc_g[0] = cp.sum(cp.asarray(get_alphas(cell)))
    vpploc_r = krhf.tools.ifft(vpploc_g, cell.mesh).real
    hkin = int1e.int1e_ipkin(cell, kpts)[0]

    for atom0 in range(0, len(atmlst), atom_batch_size):
        atom1 = min(atom0 + atom_batch_size, len(atmlst))
        atoms = np.asarray(atmlst[atom0:atom1], dtype=int)
        nset = len(atoms)
        si = krhf.get_SI(cell, atmlst=atoms)
        vloc = cp.stack([
            vloc_by_symbol[cell.atom_symbol(ia)] for ia in atoms])
        vloc_g = cp.einsum('ga,Ag,Ag->Aag', gv, 1j * si, vloc)
        vloc_r = krhf.tools.ifft(
            vloc_g.reshape(-1, vloc_g.shape[-1]), cell.mesh).real
        vloc_r = vloc_r.reshape(nset, 3, -1)
        local_de = cp.zeros((2, nset, 3), dtype=cp.float64)
        row_de = cp.zeros_like(local_de)

        for grid0 in range(0, vloc_r.shape[-1], _HCORE_GRID_BLOCK_SIZE):
            grid1 = min(vloc_r.shape[-1], grid0 + _HCORE_GRID_BLOCK_SIZE)
            ao_all = ni.eval_ao(
                cell, grid_coords[grid0:grid1], kpts, deriv=1,
                opt=eval_gto_opt).transpose(0, 1, 3, 2)
            ao = ao_all[:, 0]
            grid_indices = grid_order[grid0:grid1]
            dao_t = contract('dji,kjg->dkig', densities, ao)
            rho = cp.einsum('kig,dkig->dkg', ao.conj(), dao_t).real
            local_de += cp.einsum(
                'dkg,Axg->dAx', rho, vloc_r[:, :, grid_indices])

            # ``get_hcore`` has h1[x,p,q] = <d_x p|V_pp|q>.  Contract its
            # row and Hermitian-column corrections directly for the AO rows
            # owned by this atom batch instead of forming h1[x,p,q].
            dao = contract('dij,kjg->dkig', densities, ao)
            for iatom, ia in enumerate(atoms):
                p0, p1 = aoslices[ia, 2:]
                dao_t_row = dao_t[:, :, p0:p1]
                dao_row = dao[:, :, p0:p1]
                deriv_row = ao_all[:, 1:, p0:p1]
                # The formal ``kxig,dkig->dkxg`` einsum is a reduction over
                # the small atom-owned AO row.  CuPy lowers this shape to a
                # strided-batched GEMM, which is not stable for the largest
                # 3D PBC grid in the current CUDA/cuBLAS stack.  The explicit
                # elementwise contraction is algebraically identical and
                # keeps that tiny reduction out of the GEMM dispatcher.
                first = cp.sum(
                    deriv_row.conj()[None] * dao_t_row[:, :, None], axis=3)
                second = cp.sum(
                    deriv_row[None] * dao_row.conj()[:, :, None], axis=3)
                row_de[:, iatom] += cp.sum(
                    (first + second) * vpploc_r[grid_indices][None, None,
                                                                None, :],
                    axis=(1, 3)).real

        for iatom, ia in enumerate(atoms):
            p0, p1 = aoslices[ia, 2:]
            hkin_row = hkin[:, p0:p1, :]
            row_de[:, iatom] += cp.einsum(
                'xpq,dqp->dx', hkin_row, densities[:, :, p0:p1]).real
            row_de[:, iatom] += cp.einsum(
                'xpq,dpq->dx', hkin_row.conj(),
                densities[:, p0:p1, :]).real
        output[:, atom0:atom1] = local_de - row_de

    output += contract_nonlocal_pseudo_batch(cell, densities)[:, atmlst]
    return output[0], -2 * output[1]


def _contract_hcore_pair_batched(cell, dm0, dz, atmlst, atom_batch_size):
    r'''Return paired H1 contractions while sharing grid blocks by atom batch.

    This is the batched algebraic form of GPU4PySCF
    ``krhf.hcore_generator``.  For an atom batch ``A`` it evaluates the
    local potential grid derivatives ``v_A^x(r)`` together, streams every AO
    grid block once, and immediately contracts them with ``P`` and ``D_Z``:

    .. math::

       h_A^x:D = \sum_g v_A^x(g)
          \sum_{\mu\nu} \phi_\mu^*(g)D_{\nu\mu}\phi_\nu(g).

    Thus the batched path does *not* materialize even a per-batch
    ``(natom_batch, 3, nao, nao)`` H1 tensor.  The two atom-row corrections
    from ``krhf.hcore_generator`` are contracted separately below.  This is
    algebraically identical to forming the H1 matrix first, but makes the
    large-system path scale with AO-grid blocks rather than with one dense
    matrix per Cartesian component and atom.
    '''
    kpts = np.zeros((1, 3))
    h1 = cp.asarray(krhf.get_hcore(cell, kpts))
    si = cp.asarray(krhf.get_SI(cell))
    gv = cp.asarray(cell.Gv)
    vloc_g0 = cp.asarray(krhf.get_vlocG(cell))
    grids = krhf.UniformGrids(cell)
    ni = numint.KNumInt()
    # This is exactly the evaluator created inside GPU4PySCF's
    # ``KNumInt.block_loop``.  We retain one instance through the whole atom
    # traversal instead of recreating it for every Hcore atom batch.
    eval_gto_opt = numint._GTOvalOpt(cell, kpts, deriv=0)
    grid_order = grids.argsort(tile=8)
    grid_coords = grids.coords[grid_order]
    aoslices = np.asarray(cell.aoslice_by_atom(), dtype=int)
    atmlst = list(atmlst)
    densities = cp.stack((cp.asarray(dm0), cp.asarray(dz)))
    output = cp.empty((2, len(atmlst), 3), dtype=cp.float64)

    for atom0 in range(0, len(atmlst), atom_batch_size):
        atom1 = min(atom0 + atom_batch_size, len(atmlst))
        atoms = np.asarray(atmlst[atom0:atom1], dtype=int)
        nset = len(atoms)
        # The upstream one-atom expression is
        # ``einsum('ga,g,g->ag', Gv, 1j*SI[A], vlocG[A])``.  The leading
        # ``A`` merely batches independent nuclei and changes no phase or
        # finite-grid representation.
        vloc_g = cp.einsum(
            'ga,Ag,Ag->Aag', gv, 1j * si[atoms], vloc_g0[atoms])
        vloc_r = krhf.tools.ifft(
            vloc_g.reshape(-1, vloc_g.shape[-1]), cell.mesh).real
        vloc_r = vloc_r.reshape(nset, 3, -1)[:, :, grid_order]
        local_de = cp.zeros((2, nset, 3), dtype=cp.float64)
        for grid0 in range(0, vloc_r.shape[-1], _HCORE_GRID_BLOCK_SIZE):
            grid1 = min(vloc_r.shape[-1], grid0 + _HCORE_GRID_BLOCK_SIZE)
            ao_ks = ni.eval_ao(
                cell, grid_coords[grid0:grid1], kpts, deriv=0,
                opt=eval_gto_opt)
            ao_ks = ao_ks.transpose(0, 2, 1)
            # Evaluate both density-weighted AO products once for this AO
            # grid block, then contract every atom's local-potential
            # derivative with that small ``(density, k-point, grid)`` value.
            dao = contract('dij,kjg->dkig', densities, ao_ks)
            rho = cp.einsum('kig,dkig->dkg', ao_ks.conj(), dao).real
            local_de += cp.einsum(
                'dkg,Axg->dAx', rho,
                vloc_r[:, :, grid0:grid1])

        for iatom, ia in enumerate(atoms):
            p0, p1 = aoslices[ia, 2:]
            h1row = h1[0, :, p0:p1, :]
            # These are exactly the two Hermitian row/column corrections
            # applied by ``krhf.hcore_generator`` after its local grid
            # integral.  Keep both contractions explicit rather than using a
            # factor two: this preserves the provider's complex-Hermitian
            # definition even though the present Gamma scope is real.
            local_de[:, iatom] -= cp.einsum(
                'xpq,dqp->dx', h1row, densities[:, :, p0:p1]).real
            local_de[:, iatom] -= cp.einsum(
                'xpq,dpq->dx', h1row.conj(), densities[:, p0:p1, :]).real
        output[:, atom0:atom1] = local_de

    output += contract_nonlocal_pseudo_batch(cell, densities)[:, atmlst]
    return output[0], -2 * output[1]


def grad_hcore(mf_grad, dm0, hcore_gen, atmlst, energy_derivative=None):
    '''Return the explicit target ``Tr[P_HF h^lambda]`` contribution.'''
    if energy_derivative is None:
        de = contract_hcore(mf_grad.cell, hcore_gen, dm0, atmlst)
    else:
        de = cp.asarray(energy_derivative)
        if de.shape != (len(atmlst), 3):
            raise ValueError('precontracted hcore derivative has invalid shape')
    mf_grad.de_hcore = cp.asnumpy(de)
    return de


def grad_j(mf_grad, dm0, vj1, atmlst, energy_derivative=None):
    r'''Return the explicit target ``E_J[P_HF]`` nuclear derivative.

    Args:
        mf_grad : Gradients
            Active GPU PBC gradient object.
        dm0 : cupy.ndarray
            Closed-shell HF density.
        vj1 : cupy.ndarray or None
            Full FFTDF ``J^lambda`` matrices.  Gaussian DF does not expose
            this matrix derivative publicly, so its native contracted scalar
            route supplies ``energy_derivative`` instead.
        atmlst : sequence of int
            Requested centers.

    Kwargs:
        energy_derivative : cupy.ndarray or None
            Direct native derivative of ``Tr[P J[P]]/2`` for GDF.

    Returns:
        cupy.ndarray
            ``de_j`` in Ha/Bohr.
    '''
    if energy_derivative is None:
        if vj1 is None:
            raise ValueError('FFTDF matrix or native GDF J derivative is required')
        de = .5 * cp.einsum('axij,ji->ax', vj1[atmlst], dm0).real
        mf_grad.de_j_backend = 'fftdf_full_matrix'
    else:
        de = cp.asarray(energy_derivative)
        if de.shape != (len(atmlst), 3):
            raise ValueError('native GDF Coulomb derivative has invalid shape')
        mf_grad.de_j_backend = 'gdf_native_combined'
    mf_grad.de_j = cp.asnumpy(de)
    # GPU4PySCF currently reduces native GDF source and metric derivatives
    # together.  Retain explicit ``None`` rather than claiming a fictitious
    # source/metric partition; ``de_j_native`` is the authoritative value.
    mf_grad.de_j_native = (cp.asnumpy(de) if energy_derivative is not None
                           else None)
    mf_grad.de_j_source = None
    mf_grad.de_j_metric = None
    return de


def grad_vxc(mf_grad, dm0, atmlst):
    '''Return the fixed-density target-DFA XC derivative.

    The uniform-grid path retains GPU4PySCF's conventional periodic RKS
    derivative.  When :meth:`RHFDFT.multigrid_numint` selected GPU
    multigrid-v2, the adapter supplies an already density-contracted
    XC-only value; its Hartree and local-pseudopotential pieces have been
    explicitly disabled so the active FFTDF/GDF provider remains the
    sole Hartree owner.
    '''
    mf = mf_grad.base
    if isinstance(mf._numint, multigrid_v2.MultiGridNumInt):
        if mf.multigrid_with_j:
            raise ValueError(
                'grad_vxc cannot split a fused multigrid J+XC derivative; '
                'use skeleton_gradient')
        de = get_multigrid_veff_deriv(mf_grad, dm0)[atmlst]
        mf_grad.de_vxc = cp.asnumpy(de)
        return de
    grids = mf.grids
    if grids.coords is None:
        grids.build()
    kpts = np.zeros((1, 3))
    # ``get_vxc`` returns the electronic-coordinate row contraction.  Its
    # final factor two and sign follow the GPU4PySCF PBC RKS gradient route.
    de = 2 * krks.get_vxc(
        numint.KNumInt(), mf.cell, grids, mf.xc, dm0[None], kpts)
    de = cp.asarray(de)[atmlst]
    mf_grad.de_vxc = cp.asnumpy(de)
    mf_grad.xc_deriv_backend = 'uniform_grid'
    return de


def grad_ovlp(cell, Co, foo, s1, atmlst):
    '''Return ``-2 Tr[S_row^lambda D_target^E]`` in Ha/Bohr.'''
    dme0 = 2 * Co @ foo @ Co.conj().T
    aoslices = cell.aoslice_by_atom()
    de = cp.empty((len(atmlst), 3), dtype=cp.float64)
    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia, 2:]
        de[k] = -2 * cp.einsum(
            'xij,ji->x', s1[:, p0:p1], dme0[:, p0:p1]).real
    return de


def skeleton_gradient(mf_grad, dm0, foo, Co, s1, hcore_gen, vj1, atmlst,
                      de_j_energy=None, de_hcore_energy=None):
    '''Assemble all explicit electronic GPU DFA@RHF gradient terms.

    The fixed uniform periodic grid has no separate atom-centered grid-motion
    contribution.  ``de_nuc`` is added later by :class:`Gradients`.
    '''
    grad_hcore(
        mf_grad, dm0, hcore_gen, atmlst,
        energy_derivative=de_hcore_energy)
    mf = mf_grad.base
    if (isinstance(mf._numint, multigrid_v2.MultiGridNumInt)
            and mf.multigrid_with_j):
        de_veff = get_multigrid_veff_deriv(mf_grad, dm0)[atmlst]
        mf_grad.de_j = None
        mf_grad.de_vxc = None
        mf_grad.de_j_native = None
        mf_grad.de_j_source = None
        mf_grad.de_j_metric = None
        mf_grad.de_j_xc_multigrid = cp.asnumpy(de_veff)
        mf_grad.de_j_backend = 'multigrid_v2_j_xc_combined'
    else:
        de_j = grad_j(
            mf_grad, dm0, vj1, atmlst, energy_derivative=de_j_energy)
        de_vxc = grad_vxc(mf_grad, dm0, atmlst)
        de_veff = de_j + de_vxc
        mf_grad.de_j_xc_multigrid = None
    de_ovlp = grad_ovlp(mf_grad.cell, Co, foo, s1, atmlst)

    mf_grad.de_veff = cp.asnumpy(de_veff)
    mf_grad.de_grid = np.zeros_like(mf_grad.de_hcore)
    mf_grad.de_extra = mf_grad.de_grid
    mf_grad.de_ovlp = cp.asnumpy(de_ovlp)
    mf_grad.de_skel_elec = (
        mf_grad.de_hcore + mf_grad.de_veff + mf_grad.de_ovlp)
    return mf_grad.de_skel_elec
