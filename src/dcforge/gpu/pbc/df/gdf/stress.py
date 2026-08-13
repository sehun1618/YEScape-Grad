"""Real-Gamma GPU GDF strain response development bridge."""

import ctypes
import time
from concurrent.futures import ThreadPoolExecutor

import cupy as cp
import numpy as np
from cupyx.scipy.linalg import solve_triangular
from gpu4pyscf.gto.mole import SortedGTO
from gpu4pyscf.lib.cupy_helper import contract, get_avail_mem
from gpu4pyscf.pbc.df import rsdf_builder as gpu_rsdf_builder
from gpu4pyscf.pbc.df.aft_jk import (
    _estimate_max_shm_size,
    _generate_shl_pairs,
)
from gpu4pyscf.pbc.df.ft_ao import FTOpt, libpbc
from gpu4pyscf.pbc.gto import int1e
from pyscf.pbc.df import rsdf

from dcforge.cpu.pbc.df.gaussian.cderi import (
    cholesky_cderi,
)
from dcforge.cpu.pbc.df.gaussian.source import (
    three_center_source,
)
from dcforge.cpu.pbc.df.gaussian.strain import (
    auxiliary_fourier_strain,
    weighted_coulomb_strain,
)
from dcforge.cpu.pbc.df.gdf.metric import (
    auxiliary_metric,
)
from dcforge.cpu.pbc.df.gdf.response import (
    response_settings,
)
from dcforge.cpu.pbc.df.gdf.stress import (
    _contracted_ledger,
    _real_gamma_contraction,
    contract_short_range_metric_strain,
    contract_short_range_source_strain,
    make_short_range_source_workspace,
)
from dcforge.gpu.pbc.df.cderi import (
    load_unique_gamma_cderi,
)
from dcforge.gpu.pbc.df.gaussian.reciprocal import (
    COMPLEX128_BYTES,
    DEFAULT_COMPLEX_RECONTRACTION_MAX_BYTES,
    execute_gamma_reciprocal_blocks,
    plan_gamma_reciprocal_blocks,
)
from dcforge.gpu.pbc.df.gaussian.strain import (
    contract_overlap_strain,
)
from dcforge.gpu.pbc.df.gaussian.stress import (
    stress_reciprocal_controls,
)


def make_gdf_strain_response(
    with_df,
    dm0,
    dz,
    reciprocal_block_size=None,
    auxiliary_block_size=None,
    value_tol=1.0e-7,
    profile_components=False,
    central_context='provider',
    overlap_short_long_range=True,
):
    r"""Return analytic GDF strain observables with a device reverse adjoint.

    This P7 reference separates two concerns which have different owners.
    By default the central solved source is reconstructed from the active
    provider CDERI and the same GPU4PySCF metric decomposition that built it.
    This avoids rebuilding central ``B`` and then aligning two arbitrary
    auxiliary-factor gauges.  Metric solves and the two RHF observable
    adjoints are evaluated on the GPU.  Their strain consumers are hybrid and
    term-resolved: ``B_LR`` AO-pair, ``B_vbar``, all ``M_LR`` reductions, and
    ``M_G0`` are device-contracted; screened ``B_SR`` and ``M_SR`` use explicit
    host C/OpenMP direct contractions, while the one-center reciprocal moments
    remain small host analytic formula boundaries:

    .. math::

        W=M^{-1}B,
        \qquad
        \mathcal W_o^{(xy)}
        =A_{B,o}:B^{(xy)}-A_{M,o}:M^{(xy)}.

    The two observables are target Hartree and source-Fock Z response, in
    that order.  No finite-difference Cell or full forward strain CDERI is
    used.  Every remaining host primitive is named in the returned metadata.

    The opt-in ``transparent_oracle`` context independently rebuilds PySCF
    ``B`` and ``M`` and performs an orthogonal-Procrustes gauge comparison.
    That expensive route remains a small-system validation oracle rather than
    part of the default stress path.

    Args:
        with_df : dcforge.gpu.pbc.df.gdf.GDF
            Built real-Gamma GPU provider.
        dm0 : cupy.ndarray of shape (nao,nao)
            Spin-summed source-RHF density.
        dz : cupy.ndarray of shape (nao,nao)
            Hermitian Z transition density without occupation factor two.

    Kwargs:
        reciprocal_block_size : int or None
            Bounded reciprocal primitive G-vector block.  This changes
            storage, never the resolved GDF representation.
        auxiliary_block_size : int or None
            Device auxiliary block used to build reverse sources.  ``None``
            uses the active provider policy.
        value_tol : float
            Maximum central CDERI difference after auxiliary-gauge alignment
            in ``transparent_oracle`` mode.  Provider factor reconstruction
            uses its own scale-aware roundoff gate.
        profile_components : bool
            Synchronize GPU component boundaries and retain an internal
            timing decomposition.  Disabled by default so production calls
            do not acquire profiler synchronization overhead.
        central_context : {'provider', 'transparent_oracle'}
            ``'provider'`` reconstructs :math:`M^{-1}B` from the active
            GPU4PySCF CDERI and its native metric decomposition without
            rebuilding the three-center source.  ``'transparent_oracle'``
            retains the independent PySCF source/factor/gauge route for
            small-system validation only.
        overlap_short_long_range : bool
            Run the independent host screened integral contraction while the
            device reciprocal contraction is active.  This changes only the
            execution schedule; both complete term ledgers are joined before
            assembly.

    Returns:
        dict
            CPU-compatible source/metric/total raw strain ledger.  Metadata
            reports every device contraction and remaining host primitive,
            so this development route cannot be mistaken for fully native
            production GPU stress.
    """
    _check_real_gamma_gdf(with_df, dm0, dz)
    if getattr(with_df, '_cderi', None) is None or with_df.auxcell is None:
        with_df.build(j_only=False)
    cell = with_df.cell
    nao = int(cell.nao_nr())
    omega, source_mesh, metric_mesh = response_settings(with_df)

    timings = {} if profile_components else None
    central_context = str(central_context).lower()
    if central_context == 'provider':
        started = _component_start(profile_components)
        solved_device, central_metadata = _provider_solved_source(
            with_df, omega)
        _component_stop(
            timings, 'provider_metric_and_solved_source', started,
            synchronize=True)
        parity = None
    elif central_context == 'transparent_oracle':
        solved_device, central_metadata, parity = (
            _transparent_solved_source(
                with_df, omega, source_mesh, metric_mesh, nao,
                value_tol=value_tol, profile_components=profile_components,
                timings=timings))
    else:
        raise ValueError(
            "central_context must be 'provider' or 'transparent_oracle'")
    # From this point onward GDF and RSDF share exactly the same reverse
    # observable algebra.  Only the provider-owned central W and the resolved
    # source/metric parameters differ.  Keeping assembly in the Gaussian-DF
    # common module prevents the two stress paths from drifting term by term.
    from dcforge.gpu.pbc.df.gaussian.stress import (
        make_gaussian_strain_response,
    )
    response = make_gaussian_strain_response(
        with_df, dm0, dz, solved_device, central_metadata,
        source_omega=omega, source_mesh=source_mesh,
        metric_omega=omega, metric_mesh=metric_mesh,
        reciprocal_block_size=reciprocal_block_size,
        auxiliary_block_size=auxiliary_block_size,
        profile_components=profile_components,
        overlap_short_long_range=overlap_short_long_range,
        cderi_gauge_aligned_parity=parity)
    if profile_components:
        response['component_wall_seconds'].update(timings)
    return response


def copy_source_adjoint_to_short_range_workspace(
        cell, auxcell, source_adjoint_device, pair_block_size=None):
    r'''Pack a symmetric GPU source directly into the host SR callback ABI.

    For one unique lower-triangular AO pair ``(i,j)``, the host callback needs

    .. math::

        \widetilde A_{ijP}=A_{ijP}
        + (1-\delta_{ij})A_{jiP}.

    Earlier code copied every ordered AO pair to host and then allocated this
    packed representation.  Here bounded GPU pair blocks are summed and
    copied straight into the final contiguous C workspace.  Water64 thereby
    transfers and retains about one half as many source-adjoint bytes on the
    host, without changing the C integral traversal or contraction.

    Args:
        cell, auxcell : pyscf.pbc.gto.Cell
            Primary and auxiliary Cells.
        source_adjoint_device : cupy.ndarray
            Final source channels with shape ``(nobs,nao**2,naux)``.

    Kwargs:
        pair_block_size : int or None
            Unique AO pairs packed per transaction.  ``None`` bounds the
            device and temporary host block near 128 MiB.

    Returns:
        tuple
            ``(workspace, metadata)``.  ``workspace`` is accepted directly by
            :func:`contract_short_range_source_strain`.
    '''
    nao = int(cell.nao_nr())
    naux = int(auxcell.nao_nr())
    adjoint = cp.asarray(source_adjoint_device)
    if adjoint.ndim == 2:
        adjoint = adjoint[None]
    expected = (nao * nao, naux)
    if adjoint.ndim != 3 or tuple(adjoint.shape[1:]) != expected:
        raise ValueError(
            'source_adjoint_device must have shape '
            f'(nobs,{expected[0]},{expected[1]})')
    if cp.iscomplexobj(adjoint):
        imaginary = float(cp.max(cp.abs(adjoint.imag)).get())
        if imaginary > 1.e-10:
            raise NotImplementedError(
                'short-range source staging requires real Gamma adjoints')
        adjoint = adjoint.real
    adjoint = cp.asarray(adjoint, dtype=cp.float64)
    nobs = int(adjoint.shape[0])
    workspace = make_short_range_source_workspace(cell, auxcell, nobs)
    row = workspace['row']
    column = workspace['column']
    npair = workspace['npair']
    if pair_block_size is None:
        # One selected block and its reverse-pair addition can coexist.  Use
        # two FP64 channels in the bound and keep a nonzero tail transaction.
        target_bytes = 128 * 1024 ** 2
        pair_block_size = max(
            1, target_bytes // max(1, 2 * nobs * naux * 8))
    pair_block_size = min(npair, int(pair_block_size))
    if pair_block_size < 1:
        raise ValueError('pair_block_size must be positive')

    dense = adjoint.reshape(nobs, nao, nao, naux)
    packed_host = workspace['packed_adjoint']
    peak_device_block_bytes = 0
    started = time.perf_counter()
    for p0 in range(0, npair, pair_block_size):
        p1 = min(npair, p0 + pair_block_size)
        row_block = cp.asarray(row[p0:p1])
        column_block = cp.asarray(column[p0:p1])
        packed_device = dense[:, row_block, column_block].copy()
        off_diagonal = row[p0:p1] != column[p0:p1]
        if np.any(off_diagonal):
            selected = cp.asarray(np.flatnonzero(off_diagonal))
            packed_device[:, selected] += dense[
                :, column_block[selected], row_block[selected]]
        packed_host[:, p0:p1] = cp.asnumpy(packed_device)
        peak_device_block_bytes = max(
            peak_device_block_bytes,
            int(packed_device.nbytes + row_block.nbytes
                + column_block.nbytes))
    seconds = time.perf_counter() - started
    return workspace, {
        'storage': 'unique_symmetric_ao_pairs_in_final_sr_workspace',
        'ordered_dense_host_copy_avoided': True,
        'pair_block_size': int(pair_block_size),
        'pair_block_count': int(
            (npair + pair_block_size - 1) // pair_block_size),
        'transferred_bytes': int(workspace['packed_adjoint_bytes']),
        'workspace_bytes': int(workspace['workspace_bytes']),
        'peak_device_block_bytes': int(peak_device_block_bytes),
        'wall_seconds': float(seconds),
    }


def contract_three_center_source_strain(
    with_df,
    omega,
    mesh,
    source_adjoint_device,
    source_adjoint_host=None,
    source_short_range_workspace=None,
    precision=None,
    reciprocal_block_size=None,
    overlap_short_long_range=False,
):
    r"""Contract ``B_SR + B_LR + B_vbar`` with a native GPU ``B_LR``.

    The short-range screened three-center integral retains the validated
    host direct callback.  The reciprocal AO-pair term and analytic
    ``vbar`` overlap/volume contraction run on the GPU.  Keeping the three
    named terms separate preserves the same mathematical ledger as
    :func:`dcforge.cpu.pbc.df.gdf.stress.contract_three_center_source_strain`.

    Args:
        with_df : dcforge.gpu.pbc.df.gdf.GDF
            Active provider.  It owns the primary and auxiliary Cells and the
            bounded reciprocal-memory policy.
        omega : float
            Resolved GDF range-separation parameter.
        mesh : array_like of shape (3,)
            Resolved GDF source mesh.
        source_adjoint_device : cupy.ndarray
            Reverse-CDERI source adjoints with shape
            ``(nobs,nao**2,naux)``.

    Kwargs:
        source_adjoint_host : numpy.ndarray or None
            Host copy used only by ``B_SR``.  ``None`` performs one explicit
            device-to-host transfer.
        precision : float or None
            Screened short-range source precision.  RSDF passes its
            independently resolved ``precision_R``; ``None`` retains the
            established GDF Cell precision.
        reciprocal_block_size : int or None
            Explicit G-vector block for validation.  ``None`` uses the active
            provider memory controls without changing the resolved mesh.

    Returns:
        dict
            Contracted source ledger with independent ``short_range``,
            ``long_range`` and ``vbar`` entries.
    """
    cell = with_df.cell
    auxcell = with_df.auxcell
    if (source_adjoint_host is None
            and source_short_range_workspace is None):
        source_adjoint_host = cp.asnumpy(source_adjoint_device)

    def short_range_worker():
        started = time.perf_counter()
        value = contract_short_range_source_strain(
            cell, auxcell, omega, source_adjoint_host,
            precision=precision, workspace=source_short_range_workspace)
        return value, time.perf_counter() - started

    if overlap_short_long_range:
        with ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix='dcforge-gdf-bsr') as executor:
            short_range_future = executor.submit(short_range_worker)
            started = time.perf_counter()
            long_range = contract_long_range_source_strain(
                with_df, omega, mesh, source_adjoint_device,
                reciprocal_block_size=reciprocal_block_size)
            long_range_seconds = time.perf_counter() - started
            short_range, short_range_seconds = short_range_future.result()
        execution_schedule = 'host_B_SR_overlapped_with_device_B_LR'
    else:
        short_range, short_range_seconds = short_range_worker()
        started = time.perf_counter()
        long_range = contract_long_range_source_strain(
            with_df, omega, mesh, source_adjoint_device,
            reciprocal_block_size=reciprocal_block_size)
        long_range_seconds = time.perf_counter() - started
        execution_schedule = 'sequential_B_SR_then_B_LR'
    started = time.perf_counter()
    vbar = contract_vbar_source_strain(
        cell, auxcell, omega, source_adjoint_device)
    vbar_seconds = time.perf_counter() - started
    raw = (
        np.asarray(short_range['raw_contraction'])
        + np.asarray(long_range['raw_contraction'])
        + np.asarray(vbar['raw_contraction'])
    )
    return _contracted_ledger(
        'B',
        raw,
        short_range=short_range,
        long_range=long_range,
        vbar=vbar,
        component_wall_seconds={
            'short_range': float(short_range_seconds),
            'long_range': float(long_range_seconds),
            'vbar': float(vbar_seconds),
        },
        execution_schedule=execution_schedule,
    )


def contract_vbar_source_strain(cell, auxcell, omega, source_adjoint):
    r'''Contract the analytic Gamma :math:`B^{vbar}` response on the GPU.

    The zero-transfer source is

    .. math::

        B^{vbar}_{\mu\nu,P}
        =-S_{\mu\nu}\frac{\pi q_P}{\omega^2\Omega}.

    Contracting the auxiliary charge into ``dE/dB`` first produces one
    density-like AO matrix per observable.  GPU4PySCF's native overlap-strain
    kernel then evaluates the complete affine AO/image response.  The inverse
    volume derivative is added from the central overlap contraction.  Neither
    a full ``(9,nao**2,naux)`` tensor nor a displaced Cell is constructed.

    Args:
        cell : pyscf.pbc.gto.Cell
            Central primary Cell.
        auxcell : pyscf.pbc.gto.Cell
            Charge-normalized auxiliary Cell.
        omega : float
            Fixed positive GDF range-separation parameter.
        source_adjoint : cupy.ndarray
            Source adjoints with shape ``(nobs,nao**2,naux)``.

    Returns:
        dict
            CPU-compatible raw contraction ledger.  Auxiliary charges are
            static provider metadata; all size-dependent contractions remain
            on the device.
    '''
    omega = abs(float(omega))
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError('GDF vbar source strain requires positive omega')
    nao = int(cell.nao_nr())
    naux = int(auxcell.nao_nr())
    adjoint = cp.asarray(source_adjoint)
    if adjoint.ndim == 2:
        adjoint = adjoint[None]
    if adjoint.shape[1:] != (nao * nao, naux):
        raise ValueError(
            'source_adjoint must have shape '
            f'(nobs,{nao * nao},{naux})')
    adjoint = cp.asarray(adjoint, dtype=cp.float64)
    charges = cp.asarray(rsdf.get_aux_chg(auxcell), dtype=cp.float64)
    coefficient = charges * (np.pi / (omega ** 2 * float(cell.vol)))
    density = contract('omp,p->om', adjoint, coefficient)
    density = density.reshape(len(adjoint), nao, nao)

    overlap_response = -np.asarray([
        contract_overlap_strain(cell, selected)
        for selected in density
    ])
    overlap = int1e.int1e_ovlp(cell)
    central = -contract('oij,ij->o', density, overlap).real
    volume_response = (
        -cp.asnumpy(central)[:, None, None] * np.eye(3)[None])
    raw = _real_gamma_contraction(
        overlap_response + volume_response, 'GPU B_vbar contraction')
    return _contracted_ledger(
        'B_vbar', raw,
        overlap_response=_real_gamma_contraction(
            overlap_response, 'GPU B_vbar overlap contraction'),
        volume_response=_real_gamma_contraction(
            volume_response, 'GPU B_vbar volume contraction'),
        execution_owner='gpu4pyscf_overlap_strain_direct_adjoint',
        static_host_metadata=('auxiliary_charges',))


def contract_long_range_source_strain(
    with_df, omega, mesh, source_adjoint, reciprocal_block_size=None
):
    r"""Contract :math:`B^{LR,(xy)}` with source adjoints on the GPU.

    For each observable :math:`o`, define the reciprocal AO-pair weight

    .. math::

        T^o_{\mu\nu}(G)
        =\sum_P A^B_{o,\mu\nu P}
          [F_P(G)w(G)]^*.

    GPU4PySCF's ``PBC_ft_aopair_ek_strain_deriv`` accepts this general
    :math:`G`-dependent AO matrix and evaluates

    .. math::

        \sum_{G\mu\nu}T^o_{\mu\nu}(G)
        F_{\mu\nu}^{(xy)}(G)

    inside the screened CUDA shell-pair traversal.  The routine therefore
    never materializes the host
    ``(3,3,nG,nao,nao)`` AO-pair derivative that dominated Water8.  The
    auxiliary-Fourier and Coulomb-kernel legs remain explicit and are
    contracted from the zero-order AO-pair Fourier block:

    .. math::

        A^B:\left[F^{(xy)}F_P^*w
        +F(F_P^{(xy)})^*w+FF_P^*w^{(xy)}\right].

    Args:
        with_df : dcforge.gpu.pbc.df.gdf.GDF
            Built real-Gamma GDF provider.
        omega : float
            Resolved positive GDF range-separation parameter.
        mesh : array_like of shape (3,)
            Resolved source reciprocal mesh.
        source_adjoint : cupy.ndarray or ndarray
            Real reverse source adjoints ``dE_o/dB`` with shape
            ``(nobs,nao**2,naux)``.

    Kwargs:
        reciprocal_block_size : int or None
            Explicit block size for validation.  ``None`` resolves a bounded
            block from current free VRAM and the provider's reciprocal policy.

    Returns:
        dict
            CPU-compatible contracted ``B_LR`` ledger and execution metadata.

    Notes:
        The GPU kernel traverses one triangular shell-pair representative
        when the source adjoint is symmetric in its AO indices; its strain
        accumulator then requires the same factor two used by GPU4PySCF's
        Hartree-stress driver.  For a deliberately nonsymmetric validation
        adjoint, permutation symmetry is disabled and the full ordered shell
        pairs are traversed with unit normalization.  Both routes reproduce
        the transparent CPU raw-9 contraction.
    """
    cell = with_df.cell
    auxcell = with_df.auxcell
    if int(cell.dimension) != 3:
        raise NotImplementedError('native GDF long-range source strain requires 3D PBC')
    kpts = np.asarray(with_df.kpts, dtype=float).reshape(-1, 3)
    if len(kpts) != 1 or np.max(np.abs(kpts)) > 1.0e-12:
        raise NotImplementedError(
            'native GDF long-range source strain requires Gamma point'
        )

    nao = int(cell.nao_nr())
    naux = int(auxcell.nao_nr())
    adjoint = cp.asarray(source_adjoint)
    if adjoint.ndim == 2:
        adjoint = adjoint[None]
    expected = (nao * nao, naux)
    if adjoint.ndim != 3 or tuple(adjoint.shape[1:]) != expected:
        raise ValueError(
            f'source_adjoint must have shape (nobs,{expected[0]},{expected[1]})'
        )
    if cp.iscomplexobj(adjoint):
        imaginary = float(cp.max(cp.abs(adjoint.imag)).get())
        if imaginary > 1.0e-10:
            raise NotImplementedError(
                'native GDF source strain requires real Gamma adjoints'
            )
        adjoint = adjoint.real
    adjoint = cp.asarray(
        adjoint.reshape(len(adjoint), nao, nao, naux), dtype=cp.float64, order='C'
    )
    asymmetry = float(cp.max(cp.abs(adjoint - adjoint.transpose(0, 2, 1, 3))).get())
    permutation_symmetry = asymmetry <= 1.0e-10
    if permutation_symmetry:
        # Only the symmetric source can contract a symmetric real-Gamma
        # three-center operator.  Projecting out roundoff makes the triangular
        # CUDA traversal exactly independent of which AO triangle it visits.
        adjoint = 0.5 * (adjoint + adjoint.transpose(0, 2, 1, 3))

    mesh = np.asarray(mesh, dtype=int)
    if mesh.shape != (3,) or np.any(mesh < 1):
        raise ValueError('GDF source mesh must contain three positive values')
    vectors = np.asarray(cell.get_Gv(mesh), dtype=float)
    ngrid = len(vectors)

    ft_opt = FTOpt(cell, np.ones(3, dtype=int))
    if not permutation_symmetry:
        ft_opt.permutation_symmetry = False
    ft_opt.build()
    sorted_cell = ft_opt.cell
    ft_kernel = ft_opt.gen_ft_kernel(transform_ao=True, kpts=np.zeros((1, 3)))
    pair_indices, image_indices, shell_pair_offsets = _generate_shl_pairs(ft_opt)
    shell_pair_batches = int(len(shell_pair_offsets) - 1)
    shared_memory = int(_estimate_max_shm_size(sorted_cell, (1, 0)))
    aft_envs = ft_opt.aft_envs

    if reciprocal_block_size is None:
        controls = stress_reciprocal_controls(with_df)
        memory_plan = plan_gamma_reciprocal_blocks(
            ngrid,
            int(sorted_cell.nao),
            naux,
            max(1, len(adjoint)),
            get_avail_mem(exclude_memory_pool=True),
            memory_fraction=controls['memory_fraction'],
            minimum_reserve_bytes=controls['minimum_reserve_bytes'],
            block_alignment=controls['block_alignment'],
            maximum_block_size=controls['maximum_block_size'],
            complex_recontraction_nao=nao,
        )
        initial_block_size = int(memory_plan.initial_block_size)
        memory_plan_data = memory_plan.as_dict()
        block_alignment = int(memory_plan.block_alignment)
    else:
        controls = stress_reciprocal_controls(with_df)
        requested_block_size = min(int(reciprocal_block_size), ngrid)
        if requested_block_size < 1:
            raise ValueError('reciprocal_block_size must be positive')
        # GPU4PySCF assembles a complex ``(nG,nao,sorted_nao)`` temporary
        # through separate real and imaginary AO transformations.  A byte
        # span at or above 2**31 reproducibly raises an illegal address rather
        # than an allocation OOM, so it cannot be recovered by rebatching in
        # the same CUDA context.  Preflight explicit validation blocks too.
        recontraction_bytes_per_grid = (
            nao * int(sorted_cell.nao) * COMPLEX128_BYTES)
        recontraction_limit = (
            DEFAULT_COMPLEX_RECONTRACTION_MAX_BYTES
            // recontraction_bytes_per_grid)
        if recontraction_limit < 1:
            raise MemoryError(
                'one complex Gamma AO recontraction exceeds its safe address '
                'range')
        initial_block_size = min(requested_block_size, recontraction_limit)
        if initial_block_size >= controls['block_alignment']:
            initial_block_size = max(
                controls['block_alignment'],
                initial_block_size
                // controls['block_alignment']
                * controls['block_alignment'])
        memory_plan_data = {
            'mode': 'explicit_validation_block',
            'requested_block_size': int(requested_block_size),
            'initial_block_size': int(initial_block_size),
            'complex_recontraction_nao': int(nao),
            'complex_recontraction_bytes_per_grid': int(
                recontraction_bytes_per_grid),
            'complex_recontraction_max_bytes': int(
                DEFAULT_COMPLEX_RECONTRACTION_MAX_BYTES),
            'complex_recontraction_block_limit': int(recontraction_limit),
            'complex_recontraction_limited': bool(
                initial_block_size < requested_block_size),
        }
        block_alignment = min(
            controls['block_alignment'], initial_block_size
        )

    pair_response = cp.zeros((len(adjoint), 3, 3), dtype=cp.float64)
    auxiliary_response = cp.zeros((len(adjoint), 3, 3), dtype=cp.complex128)
    kernel_response = cp.zeros_like(auxiliary_response)
    peak_explicit_block_bytes = 0

    def worker(start, stop):
        """Build and contract one transaction-local reciprocal block."""
        block_vectors = vectors[start:stop]
        auxiliary = auxiliary_fourier_strain(auxcell, block_vectors)
        coulomb = weighted_coulomb_strain(cell, block_vectors, omega=float(omega))
        factors = np.asarray(auxiliary['value'])
        factors1 = np.asarray(auxiliary['raw_derivative'])
        weights = np.asarray(coulomb['value'])
        weights1 = np.asarray(coulomb['raw_derivative'])

        weighted_auxiliary = cp.asarray((factors * weights).T.conj(), order='C')
        pair_weight = contract('gP,oijP->ogij', weighted_auxiliary, adjoint)
        block_pair = cp.zeros_like(pair_response)
        unused_atom_gradient = cp.zeros((sorted_cell.natm, 3))
        vectors_device = cp.asarray(block_vectors.T.ravel())
        for observable in range(len(adjoint)):
            # ``pair_weight`` multiplies F_ij(G), while the CUDA ABI indexes
            # its density-like input as (j,i,G).
            density_like = pair_weight[observable].transpose(0, 2, 1)
            sorted_density_like = sorted_cell.apply_C_mat_CT(density_like)
            density_g = cp.asarray(sorted_density_like.transpose(1, 2, 0), order='C')
            error = libpbc.PBC_ft_aopair_ek_strain_deriv(
                ctypes.cast(unused_atom_gradient.data.ptr, ctypes.c_void_p),
                ctypes.cast(block_pair[observable].data.ptr, ctypes.c_void_p),
                ctypes.cast(density_g.data.ptr, ctypes.c_void_p),
                ctypes.cast(vectors_device.data.ptr, ctypes.c_void_p),
                ctypes.byref(aft_envs),
                ctypes.c_int(shell_pair_batches),
                ctypes.c_int(stop - start),
                ctypes.c_int(shared_memory),
                ctypes.cast(pair_indices.data.ptr, ctypes.c_void_p),
                ctypes.cast(image_indices.data.ptr, ctypes.c_void_p),
                ctypes.cast(shell_pair_offsets.data.ptr, ctypes.c_void_p),
                ctypes.c_int(int(ft_opt.permutation_symmetry)),
            )
            if error != 0:
                raise RuntimeError('PBC_ft_aopair_ek_strain_deriv failed for GDF B_LR')

        pair_value = ft_kernel(block_vectors)[0]
        projected_pair = contract('gij,oijP->ogP', pair_value, adjoint)
        auxiliary_weight1 = cp.asarray((factors1 * weights[None, None, None]).conj())
        kernel_weight1 = cp.asarray((factors[None, None] * weights1[:, :, None]).conj())
        block_auxiliary = contract('ogP,xyPg->oxy', projected_pair, auxiliary_weight1)
        block_kernel = contract('ogP,xyPg->oxy', projected_pair, kernel_weight1)
        explicit_bytes = sum(
            array.nbytes
            for array in (
                weighted_auxiliary,
                pair_weight,
                block_pair,
                pair_value,
                projected_pair,
                auxiliary_weight1,
                kernel_weight1,
                block_auxiliary,
                block_kernel,
            )
        )
        return block_pair, block_auxiliary, block_kernel, explicit_bytes

    def accept_result(_start, _stop, result):
        """Commit a completed block exactly once after all kernels succeed."""
        nonlocal peak_explicit_block_bytes
        block_pair, block_auxiliary, block_kernel, explicit_bytes = result
        pair_response[:] += block_pair
        auxiliary_response[:] += block_auxiliary
        kernel_response[:] += block_kernel
        peak_explicit_block_bytes = max(peak_explicit_block_bytes, int(explicit_bytes))

    execution = execute_gamma_reciprocal_blocks(
        ngrid,
        initial_block_size,
        worker,
        accept_result,
        phase='gdf_stress_long_range_source',
        is_oom=_is_gpu_out_of_memory,
        block_alignment=block_alignment,
        on_oom=_release_cached_gpu_blocks,
    )
    pair_normalization = 2.0 if permutation_symmetry else 1.0
    pair_host = cp.asnumpy(pair_response * pair_normalization)
    auxiliary_host = _real_gamma_contraction(
        cp.asnumpy(auxiliary_response), 'GPU B_LR auxiliary contraction'
    )
    kernel_host = _real_gamma_contraction(
        cp.asnumpy(kernel_response), 'GPU B_LR kernel contraction'
    )
    raw = _real_gamma_contraction(
        pair_host + auxiliary_host + kernel_host, 'GPU B_LR contraction'
    )
    return _contracted_ledger(
        'B_LR',
        raw,
        ao_pair_response=pair_host,
        auxiliary_response=auxiliary_host,
        kernel_response=kernel_host,
        reciprocal_block_size=int(execution.final_block_size),
        reciprocal_block_size_initial=int(initial_block_size),
        grid_points=int(ngrid),
        block_execution=execution.as_dict(),
        memory_plan=memory_plan_data,
        peak_explicit_block_bytes=int(peak_explicit_block_bytes),
        source_adjoint_ao_asymmetry_max_abs=float(asymmetry),
        permutation_symmetry=bool(permutation_symmetry),
        pair_normalization=float(pair_normalization),
        execution_owner='gpu4pyscf_general_ao_pair_strain_direct_adjoint',
    )


def contract_long_range_metric_strain(
        with_df, omega, mesh, metric_adjoint,
        reciprocal_block_size=None):
    r'''Contract :math:`M^{LR,(xy)}` with the metric adjoint on the GPU.

    For the symmetric metric adjoint :math:`A_{PQ}`, the three formal metric
    legs are reduced to one shared projection,

    .. math::

        (AF)_{oPG}=\sum_Q A_{oPQ}F_{QG}.

    The two auxiliary-Fourier legs are then twice the real part of the same
    contraction and the weighted-Coulomb leg reuses
    :math:`F^\dagger AF`.  This is the exact contraction order used by the CPU
    reference, but all :math:`P,Q,G` reductions and accumulators are device
    resident.  The currently available GPU4PySCF API does not expose the
    one-center Fourier Cartesian moment required for affine strain, so the
    small transaction-local ``F``/``F^(xy)`` primitives are still generated
    by the validated analytic PySCF helper and explicitly reported as a host
    boundary.

    Args:
        with_df : dcforge.gpu.pbc.df.gdf.GDF
            Active provider and reciprocal-memory policy owner.
        omega : float
            Fixed positive GDF range-separation parameter.
        mesh : array_like of shape (3,)
            Fixed central metric mesh.
        metric_adjoint : cupy.ndarray
            Reverse metric adjoints with shape ``(nobs,naux,naux)``.

    Kwargs:
        reciprocal_block_size : int or None
            Validation block. ``None`` uses the provider memory planner and
            transactional allocation-OOM rebatching.

    Returns:
        dict
            CPU-compatible ``M_LR`` ledger and device-execution metadata.
    '''
    cell = with_df.cell
    auxcell = with_df.auxcell
    naux = int(auxcell.nao_nr())
    adjoint = cp.asarray(metric_adjoint)
    if adjoint.ndim == 2:
        adjoint = adjoint[None]
    if adjoint.ndim != 3 or adjoint.shape[1:] != (naux, naux):
        raise ValueError(
            'metric_adjoint must have shape '
            f'(nobs,{naux},{naux})')
    if cp.iscomplexobj(adjoint):
        imaginary = float(cp.max(cp.abs(adjoint.imag)).get())
        if imaginary > 1.e-10:
            raise NotImplementedError(
                'native GDF metric strain requires real Gamma adjoints')
        adjoint = adjoint.real
    adjoint = cp.asarray(adjoint, dtype=cp.float64, order='C')
    antisymmetry = adjoint - adjoint.transpose(0, 2, 1)
    antisymmetry_max = float(cp.max(cp.abs(antisymmetry)).get())
    adjoint = .5 * (adjoint + adjoint.transpose(0, 2, 1))

    mesh = np.asarray(mesh, dtype=int)
    if mesh.shape != (3,) or np.any(mesh < 1):
        raise ValueError('GDF metric mesh must contain three positive values')
    vectors = np.asarray(cell.get_Gv(mesh), dtype=float)
    ngrid = len(vectors)
    if reciprocal_block_size is None:
        controls = stress_reciprocal_controls(with_df)
        # One metric transaction retains roughly twelve auxiliary-length
        # complex channels per G vector.  Passing ``3*naux`` to the common
        # four-channel planner is a conservative exact-storage surrogate; it
        # changes only the block, never the reciprocal representation.
        memory_plan = plan_gamma_reciprocal_blocks(
            ngrid, 1, max(1, 3 * naux), max(1, len(adjoint)),
            get_avail_mem(exclude_memory_pool=True),
            memory_fraction=controls['memory_fraction'],
            minimum_reserve_bytes=controls['minimum_reserve_bytes'],
            block_alignment=controls['block_alignment'],
            maximum_block_size=controls['maximum_block_size'])
        initial_block_size = int(memory_plan.initial_block_size)
        memory_plan_data = memory_plan.as_dict()
        memory_plan_data['physical_naux'] = naux
        memory_plan_data['planning_naux'] = 3 * naux
        memory_plan_data['planning_reason'] = (
            'twelve_auxiliary_complex_channels_per_grid')
        block_alignment = int(memory_plan.block_alignment)
    else:
        controls = stress_reciprocal_controls(with_df)
        initial_block_size = min(int(reciprocal_block_size), ngrid)
        if initial_block_size < 1:
            raise ValueError('reciprocal_block_size must be positive')
        memory_plan_data = {
            'mode': 'explicit_validation_block',
            'initial_block_size': initial_block_size,
            'physical_naux': naux,
        }
        block_alignment = min(
            controls['block_alignment'], initial_block_size)

    auxiliary_response = cp.zeros(
        (len(adjoint), 3, 3), dtype=cp.float64)
    kernel_response = cp.zeros_like(auxiliary_response)
    peak_explicit_block_bytes = 0

    def worker(start, stop):
        '''Build analytic primitives and contract one complete metric block.'''
        block_vectors = vectors[start:stop]
        auxiliary = auxiliary_fourier_strain(auxcell, block_vectors)
        kernel = weighted_coulomb_strain(
            cell, block_vectors, omega=float(omega))
        factors = cp.asarray(auxiliary['value'])
        factors1 = cp.asarray(auxiliary['raw_derivative'])
        weights = cp.asarray(kernel['value'])
        weights1 = cp.asarray(kernel['raw_derivative'])

        projected = contract('opq,qg->opg', adjoint, factors)
        weighted_factors1 = (
            factors1.conj() * weights[None, None, None, :])
        block_auxiliary = 2 * contract(
            'xypg,opg->oxy', weighted_factors1, projected).real
        bilinear = contract('pg,opg->og', factors.conj(), projected)
        block_kernel = contract(
            'og,xyg->oxy', bilinear, weights1).real
        explicit_bytes = sum(
            value.nbytes for value in (
                factors, factors1, weights, weights1, projected,
                weighted_factors1, bilinear, block_auxiliary, block_kernel))
        return block_auxiliary, block_kernel, explicit_bytes

    def accept_result(_start, _stop, result):
        '''Commit one successful block exactly once.'''
        nonlocal peak_explicit_block_bytes
        block_auxiliary, block_kernel, explicit_bytes = result
        auxiliary_response[:] += block_auxiliary
        kernel_response[:] += block_kernel
        peak_explicit_block_bytes = max(
            peak_explicit_block_bytes, int(explicit_bytes))

    execution = execute_gamma_reciprocal_blocks(
        ngrid, initial_block_size, worker, accept_result,
        phase='gdf_stress_long_range_metric',
        is_oom=_is_gpu_out_of_memory,
        block_alignment=block_alignment,
        on_oom=_release_cached_gpu_blocks)
    auxiliary_host = cp.asnumpy(auxiliary_response)
    kernel_host = cp.asnumpy(kernel_response)
    raw = _real_gamma_contraction(
        auxiliary_host + kernel_host, 'GPU M_LR contraction')
    return _contracted_ledger(
        'M_LR', raw,
        auxiliary_response=_real_gamma_contraction(
            auxiliary_host, 'GPU M_LR auxiliary contraction'),
        kernel_response=_real_gamma_contraction(
            kernel_host, 'GPU M_LR kernel contraction'),
        contraction_order='device_shared_metric_adjoint_projection',
        adjoint_antisymmetry_max_abs=antisymmetry_max,
        reciprocal_block_size=int(execution.final_block_size),
        reciprocal_block_size_initial=initial_block_size,
        grid_points=ngrid,
        block_execution=execution.as_dict(),
        memory_plan=memory_plan_data,
        peak_explicit_block_bytes=int(peak_explicit_block_bytes),
        execution_owner='cupy_metric_reverse_adjoint_contraction',
        host_analytic_primitives=(
            'auxiliary_Fourier_strain', 'weighted_Coulomb_strain'))


def contract_g0_metric_strain(cell, auxcell, omega, metric_adjoint):
    r'''Contract the analytic Gamma zero-mode metric response on the GPU.'''
    omega = abs(float(omega))
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError('GDF G=0 metric strain requires positive omega')
    naux = int(auxcell.nao_nr())
    adjoint = cp.asarray(metric_adjoint)
    if adjoint.ndim == 2:
        adjoint = adjoint[None]
    if adjoint.shape[1:] != (naux, naux):
        raise ValueError(
            'metric_adjoint must have shape '
            f'(nobs,{naux},{naux})')
    adjoint = cp.asarray(adjoint, dtype=cp.float64)
    charges = cp.asarray(rsdf.get_aux_chg(auxcell), dtype=cp.float64)
    coefficient = np.pi / (omega ** 2 * float(cell.vol))
    projected = contract('opq,q->op', adjoint, charges)
    central = -coefficient * contract('op,p->o', projected, charges)
    raw = -cp.asnumpy(central)[:, None, None] * np.eye(3)[None]
    return _contracted_ledger(
        'M_G0', raw,
        execution_owner='cupy_zero_mode_metric_direct_adjoint',
        static_host_metadata=('auxiliary_charges',))


def contract_auxiliary_metric_strain(
        with_df, omega, mesh, metric_adjoint_device,
        metric_adjoint_host=None, precision=None,
        reciprocal_block_size=None, overlap_short_long_range=False):
    r'''Contract the complete :math:`M_{SR}+M_{LR}+M_{G0}` ledger.

    ``M_SR`` retains the exact image-resolved PySCF two-center integral
    consumer because upstream ``int2c2e_ip1`` does not expose the lattice-image
    moment required by homogeneous strain.  The expensive ``M_LR`` adjoint
    reductions and the complete ``M_G0`` response execute on the GPU.
    '''
    cell = with_df.cell
    auxcell = with_df.auxcell
    provider_backend = str(getattr(with_df, 'backend_name', 'gdf')).lower()
    metric_screening = 'rsdf' if provider_backend == 'rsdf' else 'gdf'
    if metric_adjoint_host is None:
        metric_adjoint_host = cp.asnumpy(metric_adjoint_device)
    def short_range_worker():
        started = time.perf_counter()
        value = contract_short_range_metric_strain(
            auxcell, omega, metric_adjoint_host, precision=precision,
            screening=metric_screening)
        return value, time.perf_counter() - started

    if overlap_short_long_range:
        with ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix='dcforge-gdf-msr') as executor:
            short_range_future = executor.submit(short_range_worker)
            started = time.perf_counter()
            long_range = contract_long_range_metric_strain(
                with_df, omega, mesh, metric_adjoint_device,
                reciprocal_block_size=reciprocal_block_size)
            long_range_seconds = time.perf_counter() - started
            short_range, short_range_seconds = short_range_future.result()
        execution_schedule = 'host_M_SR_overlapped_with_device_M_LR'
    else:
        short_range, short_range_seconds = short_range_worker()
        started = time.perf_counter()
        long_range = contract_long_range_metric_strain(
            with_df, omega, mesh, metric_adjoint_device,
            reciprocal_block_size=reciprocal_block_size)
        long_range_seconds = time.perf_counter() - started
        execution_schedule = 'sequential_M_SR_then_M_LR'
    started = time.perf_counter()
    zero_mode = contract_g0_metric_strain(
        cell, auxcell, omega, metric_adjoint_device)
    zero_mode_seconds = time.perf_counter() - started
    raw = (
        np.asarray(short_range['raw_contraction'])
        + np.asarray(long_range['raw_contraction'])
        + np.asarray(zero_mode['raw_contraction']))
    return _contracted_ledger(
        'M', raw, short_range=short_range, long_range=long_range,
        g0=zero_mode, component_wall_seconds={
            'short_range': float(short_range_seconds),
            'long_range': float(long_range_seconds),
            'g0': float(zero_mode_seconds),
        }, execution_schedule=execution_schedule,
        provider_backend=provider_backend,
        short_range_screening=metric_screening)


def _component_start(enabled):
    """Return a timer state only for the opt-in component profiler."""
    return time.perf_counter() if enabled else None


def _component_stop(timings, name, started, synchronize=False):
    """Record one optional GDF component without affecting default calls."""
    if started is None:
        return
    if synchronize:
        cp.cuda.Stream.null.synchronize()
    timings[str(name)] = float(time.perf_counter() - started)


def _is_gpu_out_of_memory(error):
    """Whether a reciprocal transaction can be retried with a smaller block."""
    if isinstance(error, cp.cuda.memory.OutOfMemoryError):
        return True
    message = str(error).lower()
    return error.__class__.__name__ == 'CUDARuntimeError' and (
        'out of memory' in message or 'cudaerrormemoryallocation' in message
    )


def _release_cached_gpu_blocks(_start, _stop, _error):
    """Release unused CuPy blocks before retrying one reciprocal interval."""
    cp.get_default_memory_pool().free_all_blocks()


def _provider_solved_source(with_df, omega):
    r'''Return :math:`M^{-1}B` from the active provider factorization.

    GPU4PySCF stores the Gamma CDERI

    .. math::

        C = U^\mathsf{T} B,
        \qquad U U^\mathsf{T}=M^{-1},

    where ``U`` is either the retained eigen inverse square root or
    :math:`L^{-\mathsf T}` for a Cholesky metric.  Consequently the primitive
    solved source required by the stress reverse adjoint is simply

    .. math::

        M^{-1}B = U C.

    Reconstructing ``U`` with the same GPU4PySCF metric builder removes the
    independent three-center source build and the auxiliary Procrustes gauge
    alignment.  The fixed provider CDERI screening and auxiliary rank are
    thereby inherited exactly.
    '''
    if getattr(with_df, '_cderip', None) is not None:
        raise NotImplementedError(
            'provider stress context requires a positive 3D Gamma metric')
    auxcell = with_df.auxcell
    naux = int(auxcell.nao_nr())
    provider_cderi = _dense_provider_cderi_device(with_df)
    physical_omega = abs(float(getattr(with_df, '_omega', 0.)))
    if physical_omega < float(omega):
        metric_cutoff = gpu_rsdf_builder.estimate_ke_cutoff_for_omega(
            auxcell, float(omega), auxcell.precision * 1.e-6)
        expected_metric_mesh = np.asarray(
            auxcell.cutoff_to_mesh(metric_cutoff), dtype=int)
    else:
        expected_metric_mesh = np.ones(3, dtype=int)
    recorded_metric_mesh = getattr(with_df, 'resolved_metric_mesh', None)
    if recorded_metric_mesh is None:
        raise RuntimeError(
            'provider stress context requires the resolved metric mesh '
            'recorded by the active GDF build')
    recorded_metric_mesh = np.asarray(recorded_metric_mesh, dtype=int)
    if not np.array_equal(expected_metric_mesh, recorded_metric_mesh):
        raise RuntimeError(
            'provider metric reconstruction would use a different mesh: '
            f'expected {expected_metric_mesh.tolist()}, active provider '
            f'recorded {recorded_metric_mesh.tolist()}')
    sorted_auxcell = SortedGTO.from_cell(auxcell)
    metric = gpu_rsdf_builder._get_2c2e(
        sorted_auxcell, None, physical_omega, float(omega))
    metric = cp.asarray(metric, dtype=cp.float64, order='C')
    if metric.shape != (naux, naux):
        raise RuntimeError(
            'provider metric reconstruction has shape '
            f'{metric.shape}, expected {(naux, naux)}')
    decomposition, negative, tag = gpu_rsdf_builder.decompose_j2c(
        metric,
        prefer_ed=gpu_rsdf_builder.PREFER_ED,
        linear_dep_threshold=float(with_df.linear_dep_threshold))
    if negative is not None:
        raise NotImplementedError(
            'provider stress context does not support a negative metric rank')
    if tag == 'ED':
        inverse_factor = cp.asarray(decomposition, order='C')
    elif tag == 'CD':
        identity = cp.eye(naux, dtype=cp.float64)
        inverse_factor = solve_triangular(
            decomposition.T, identity, lower=False)
        del identity
    else:
        raise RuntimeError(f'unknown GPU4PySCF metric decomposition {tag!r}')
    del decomposition
    rank = int(inverse_factor.shape[1])
    if provider_cderi.shape[0] != rank:
        raise RuntimeError(
            'provider CDERI rank differs from reconstructed metric rank: '
            f'{provider_cderi.shape[0]} and {rank}')
    flat_cderi = provider_cderi.reshape(rank, -1)
    solved = inverse_factor @ flat_cderi
    solved = solved.reshape(
        naux, int(with_df.cell.nao_nr()), int(with_df.cell.nao_nr()))

    # This is an algebraic factorization residual, not a comparison with a
    # separately rebuilt source.  It verifies that the reconstructed inverse
    # factor maps the active provider CDERI back to the same retained rank.
    factor_scale = max(
        1., float(cp.max(cp.abs(flat_cderi)).get()))
    # Validate the same factor identity in bounded AO-pair columns.  The
    # former full GEMMs retained both ``M W`` and ``U.T M W`` alongside W and
    # dense provider CDERI, adding two complete naux*nao**2 tensors solely
    # for a diagnostic residual.  Blocking changes no element of the check.
    pair_count = int(flat_cderi.shape[1])
    validation_target_bytes = 128 * 1024 ** 2
    validation_pair_block_size = max(
        1, validation_target_bytes // max(1, 2 * naux * 8))
    validation_pair_block_size = min(
        pair_count, int(validation_pair_block_size))
    residual = 0.0
    for q0 in range(0, pair_count, validation_pair_block_size):
        q1 = min(pair_count, q0 + validation_pair_block_size)
        primitive_block = metric @ solved.reshape(naux, -1)[:, q0:q1]
        reconstructed_block = inverse_factor.T @ primitive_block
        residual = max(residual, float(cp.max(cp.abs(
            reconstructed_block - flat_cderi[:, q0:q1])).get()))
    factor_tolerance = max(
        1.e-10,
        1000. * np.finfo(np.float64).eps * max(naux, rank) * factor_scale,
    )
    if residual > factor_tolerance:
        raise RuntimeError(
            'provider metric/CDERI reconstruction residual is too large: '
            f'{residual:.3e} > {factor_tolerance:.3e}')
    return solved, {
        'mode': 'provider_native_metric',
        'metric_owner': 'gpu4pyscf_rsdf_builder',
        'source_owner': 'active_provider_CDERI',
        'decomposition': str(tag),
        'physical_omega': physical_omega,
        'partition_omega': float(omega),
        'metric_mesh': recorded_metric_mesh.tolist(),
        'physical_auxiliary_dimension': naux,
        'retained_metric_rank': rank,
        'factor_reconstruction_max_abs': residual,
        'factor_reconstruction_tolerance': factor_tolerance,
        'factor_reconstruction_pair_block_size': int(
            validation_pair_block_size),
        'factor_reconstruction_full_tensor_temporaries': False,
        'provider_metric_api': (
            'gpu4pyscf.pbc.df.rsdf_builder._get_2c2e'),
        'private_api_boundary': True,
        'transparent_source_rebuilt': False,
        'gauge_alignment_performed': False,
        'host_to_device_bytes': int(provider_cderi.nbytes),
    }


def _transparent_solved_source(
        with_df, omega, source_mesh, metric_mesh, nao, value_tol,
        profile_components, timings):
    '''Return the independent transparent central source validation route.'''
    cell = with_df.cell
    auxcell = with_df.auxcell
    started = _component_start(profile_components)
    metric = auxiliary_metric(cell, auxcell, omega, metric_mesh)
    _component_stop(timings, 'central_metric', started)
    started = _component_start(profile_components)
    source = three_center_source(
        cell, auxcell, omega, source_mesh, max_memory=with_df.max_memory)
    _component_stop(timings, 'central_source', started)
    started = _component_start(profile_components)
    factor, reference_cderi = cholesky_cderi(source, metric, nao)
    _component_stop(timings, 'reference_factor_and_cderi', started)
    started = _component_start(profile_components)
    provider_cderi = _dense_provider_cderi(with_df)
    _component_stop(timings, 'provider_cderi_unpack', started)
    started = _component_start(profile_components)
    parity = _gauge_aligned_cderi_parity(
        reference_cderi, provider_cderi)
    _component_stop(timings, 'cderi_gauge_alignment', started)
    if parity > float(value_tol):
        raise RuntimeError(
            'GPU and transparent GDF central operators differ after '
            f'auxiliary-gauge alignment: {parity:.3e} > '
            f'{float(value_tol):.3e}')

    started = _component_start(profile_components)
    factor_device = cp.asarray(factor, dtype=cp.float64)
    source_device = cp.asarray(np.asarray(source).T, dtype=cp.float64)
    cderi_device = solve_triangular(
        factor_device, source_device, lower=True)
    solved = solve_triangular(
        factor_device.T, cderi_device, lower=False)
    solved = solved.reshape(auxcell.nao_nr(), nao, nao)
    _component_stop(
        timings, 'device_metric_solve', started, synchronize=True)
    metadata = {
        'mode': 'transparent_oracle',
        'metric_owner': 'dcforge_cpu_transparent_GDF',
        'source_owner': 'dcforge_cpu_transparent_GDF',
        'decomposition': 'CD',
        'physical_auxiliary_dimension': int(auxcell.nao_nr()),
        'retained_metric_rank': int(auxcell.nao_nr()),
        'factor_reconstruction_max_abs': None,
        'transparent_source_rebuilt': True,
        'gauge_alignment_performed': True,
        'host_to_device_bytes': int(factor.nbytes + source.nbytes),
    }
    return solved, metadata, parity


def _dense_provider_cderi_device(with_df):
    '''Return the active Gamma CDERI as dense physical AO pairs on device.'''
    packed, layout = load_unique_gamma_cderi(
        with_df,
        auxiliary_block_size=with_df.adjoint_auxiliary_block_size)
    naux = int(packed.shape[0])
    nao = int(with_df.cell.nao_nr())
    dense = cp.zeros((naux, nao, nao), dtype=packed.dtype)
    row = cp.asarray(layout['row'], dtype=cp.int32)
    column = cp.asarray(layout['column'], dtype=cp.int32)
    dense[:, row, column] = packed
    dense[:, column, row] = packed
    return dense


def _dense_provider_cderi(with_df):
    """Return GPU4PySCF's Gamma CDERI in dense physical AO-pair storage."""
    return cp.asnumpy(_dense_provider_cderi_device(with_df))


def _gauge_aligned_cderi_parity(reference, provider):
    r"""Return CDERI parity after the optimal auxiliary orthogonal rotation.

    A left orthogonal rotation of ``C[P,mu,nu]`` leaves
    ``sum_P C_P C_P`` and hence every J/K matrix unchanged.  The SVD below
    removes only that mathematically irrelevant gauge; it does not rescale or
    refit either provider.
    """
    reference = np.asarray(reference, dtype=float)
    provider = np.asarray(provider, dtype=float)
    if reference.shape != provider.shape:
        raise ValueError(
            'CPU/GPU central CDERI dimensions differ: '
            f'{reference.shape} and {provider.shape}'
        )
    reference_flat = reference.reshape(reference.shape[0], -1)
    provider_flat = provider.reshape(provider.shape[0], -1)
    left, _singular, right = np.linalg.svd(
        reference_flat @ provider_flat.T, full_matrices=False
    )
    rotation = left @ right
    return float(np.max(np.abs(rotation @ provider_flat - reference_flat)))


def _check_real_gamma_gdf(with_df, dm0, dz):
    """Validate the restricted real-Gamma domain before any allocation."""
    kpts = np.asarray(with_df.kpts, dtype=float).reshape(-1, 3)
    if len(kpts) != 1 or np.max(np.abs(kpts)) > 1.0e-12:
        raise NotImplementedError('GPU GDF stress requires real Gamma point')
    if with_df.cell.dimension != 3:
        raise NotImplementedError('GPU GDF stress requires a 3D Cell')
    nao = int(with_df.cell.nao_nr())
    for name, value in (('dm0', dm0), ('dz', dz)):
        array = cp.asarray(value)
        if array.shape != (nao, nao):
            raise ValueError(f'{name} must have shape ({nao},{nao})')
        if cp.iscomplexobj(array):
            imaginary = float(cp.max(cp.abs(array.imag)).get())
            if imaginary > 1.0e-10:
                raise NotImplementedError(
                    f'{name} has non-Gamma imaginary part {imaginary:.3e}'
                )
        asymmetry = float(cp.max(cp.abs(array - array.T.conj())).get())
        if asymmetry > 1.0e-9:
            raise ValueError(f'{name} must be Hermitian at Gamma')


__all__ = ['make_gdf_strain_response']
