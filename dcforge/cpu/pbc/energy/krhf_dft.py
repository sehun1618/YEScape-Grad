'''Full-BZ sampled-k semilocal-DFA@KRHF energy with shared FFTDF/GDF.'''

import hashlib

import numpy as np
from pyscf.pbc.dft import gen_grid, krks, multigrid
from pyscf.pbc.scf import khf

from dcforge.cpu.pbc.df.fft import FFTDF
from dcforge.cpu.pbc.df.gdf import GDF
from dcforge.cpu.pbc.kpts import (
    IMAGINARY_RESIDUAL_TOL,
    check_equal_weights,
    check_kpts_equal,
    equal_kpts_weights,
    kpoint_fingerprint,
    normalize_kpts,
    validate_insulating_occupations,
    validate_kresolved_hermitian,
)
from dcforge.tools.pbc_hf_state import (
    PBCStateError,
    assert_pbc_source_state,
    clear_pbc_source_state,
    configure_pbc_provider,
    load_pbc_hf_state,
    record_pbc_source_state,
    restore_pbc_source_state,
    save_pbc_hf_state,
)


def _real_scalar(value, name, tol=IMAGINARY_RESIDUAL_TOL):
    '''Return a physical real scalar after checking its imaginary residual.'''
    value = np.asarray(value).item()
    if abs(np.imag(value)) > tol:
        raise ValueError(
            f'{name} imaginary residual {abs(np.imag(value)):.3e} exceeds '
            f'{tol:.3e}')
    return float(np.real(value))


def _check_input(mf):
    '''Validate the full-BZ fixed-density semilocal-DFA@KRHF capability.'''
    cell = mf.cell
    if cell.dimension != 3:
        raise NotImplementedError('sampled-k KRHFDFT currently requires a 3D cell')
    if cell.spin != 0 or cell.nelectron % 2:
        raise NotImplementedError('sampled-k KRHFDFT requires a closed-shell cell')
    if not cell._pseudo or cell._ecp:
        raise NotImplementedError('sampled-k KRHFDFT currently requires GTH pseudo')
    if not isinstance(mf.with_df, (FFTDF, GDF)):
        raise NotImplementedError(
            'sampled-k KRHFDFT currently supports CPU FFTDF and GDF only')
    if mf.with_df is not mf.mf_hf.with_df:
        raise ValueError('source KRHF and target KRKS must share one DF provider')
    if mf.with_df.cell is not cell or mf.mf_hf.cell is not cell:
        raise ValueError('source, target, and DF provider must own one Cell')

    kpts = check_kpts_equal(
        mf._kpts_runtime_reference, mf.kpts, 'target KRKS runtime kpts')
    check_kpts_equal(
        mf._source_kpts_runtime_reference, mf.mf_hf.kpts,
        'source KRHF runtime kpts')
    check_kpts_equal(
        mf._provider_kpts_runtime_reference, mf.with_df.kpts,
        'density-fitting provider runtime kpts')
    check_kpts_equal(kpts, mf.mf_hf.kpts, 'source KRHF kpts')
    check_kpts_equal(kpts, mf.with_df.kpts, 'density-fitting provider kpts')
    check_equal_weights(mf._kpts_reference, mf.kpts_weights)
    if kpoint_fingerprint(
            mf._kpts_reference, mf.kpts_weights,
            wrap_around=mf.kpts_wrap_around) != mf.kpts_fingerprint:
        raise ValueError(
            'sampled-k coordinates, ordering, weights, or wrap metadata '
            'changed; construct a new KRHFDFT object')
    if mf.exxdiv != mf.mf_hf.exxdiv:
        raise ValueError('source and target exxdiv conventions must match')
    if (isinstance(mf.with_df, FFTDF) and not np.array_equal(
            np.asarray(mf.with_df.mesh), np.asarray(cell.mesh))):
        raise ValueError('sampled-k FFTDF energy requires with_df.mesh == cell.mesh')

    if isinstance(
            mf._numint,
            (multigrid.MultiGridNumInt, multigrid.MultiGridNumInt2)):
        raise NotImplementedError(
            'CPU sampled-k multigrid is unavailable because PySCF 2.14 '
            'pair-multigrid density collocation and AO pass-2 kernels are '
            'Gamma-only; use the sampled-k uniform-grid XC route')
    if not isinstance(mf.grids, gen_grid.UniformGrids):
        raise NotImplementedError('sampled-k target XC requires UniformGrids')
    if mf._numint.libxc.is_hybrid_xc(mf.xc):
        raise NotImplementedError(
            'sampled-k target functionals must be pure semilocal DFAs')
    if mf._numint.libxc.is_nlc(mf.xc) or mf.nlc not in ('', 0, False):
        raise NotImplementedError('sampled-k NLC/VV10 is not supported')
    xctype = mf._numint._xc_type(mf.xc)
    if xctype not in ('LDA', 'GGA', 'MGGA'):
        raise NotImplementedError(
            'sampled-k target functional must be pure LDA, GGA or MGGA')
    if xctype == 'MGGA' and mf._numint.libxc.needs_laplacian(mf.xc):
        raise NotImplementedError(
            'sampled-k MGGA targets requiring the density Laplacian are '
            'not supported')


class DFKRHF(khf.KRHF):
    '''Full-BZ KRHF whose J/K is evaluated by the shared DF provider.'''


FFTDFKRHF = DFKRHF


class KRHFDFT(krks.KRKS):
    r'''Evaluate :math:`E_{\rm DFA}[P_{\rm KRHF}]` on a full-BZ k mesh.

    Args:
        cell : pyscf.pbc.gto.Cell
            Built three-dimensional closed-shell GTH-pseudopotential cell.
        kpts : array_like
            Explicit full-Brillouin-zone Cartesian k-points with shape
            ``(nkpts, 3)`` in reciprocal Bohr.

    Kwargs:
        xc : str
            Pure LDA, GGA, or tau-dependent MGGA target functional.  Hybrid,
            range-separated hybrid, Laplacian-dependent MGGA, NLC, and VV10
            targets are rejected.
        mf_hf : pyscf.pbc.scf.khf.KRHF or None
            Optional source object using the same Cell, k-points, and the
            same DC-Forge FFTDF or GDF provider.
        kpts_weights : array_like or None
            Full-BZ weights.  Only exact equal weights ``1 / nkpts`` are
            supported.
        kpts_wrap_around : bool or None
            Provenance metadata describing how ``cell.make_kpts`` generated
            the explicit array; it does not modify the coordinates.
        exxdiv : str
            PySCF exact-exchange divergence convention for source KRHF.

    Notes:
        ``kernel`` runs source KRHF only.  It then calls PySCF's supplied-DM
        ``get_veff`` and ``energy_tot`` methods; target KRKS orbitals are
        never optimized.  PySCF already applies the ``1 / nkpts`` energy
        normalization, so DC-Forge does not multiply by the weights again.
    '''

    _keys = krks.KRKS._keys | {
        'mf_hf', 'e_hf', 'e_dft', 'dm_hf', 'target_fock',
        'target_fock_signature', 'source_state_signature',
        'kpts_weights', 'kpts_wrap_around', 'kpts_fingerprint',
        'source_gap', 'kpts_roundtrip_max_abs',
        '_kpts_runtime_reference', '_source_kpts_runtime_reference',
        '_provider_kpts_runtime_reference',
        '_source_state_receipt',
    }
    _hf_options = (
        'max_memory', 'verbose', 'stdout', 'chkfile', 'conv_tol',
        'conv_tol_grad', 'max_cycle', 'init_guess', 'DIIS', 'diis',
        'diis_space', 'diis_start_cycle', 'damp', 'level_shift',
        'direct_scf', 'direct_scf_tol', 'conv_check', 'callback', 'exxdiv',
    )

    def __init__(self, cell, kpts=None, xc='PBE', mf_hf=None,
                 kpts_weights=None, kpts_wrap_around=None, exxdiv='ewald'):
        kpts = normalize_kpts(kpts)
        krks.KRKS.__init__(self, cell, kpts=kpts, xc=xc, exxdiv=exxdiv)
        self._kpts_runtime_reference = np.asarray(self.kpts).copy()
        self.kpts_roundtrip_max_abs = float(np.max(np.abs(
            self._kpts_runtime_reference - kpts)))
        self.conv_tol = 1e-10
        self.conv_tol_grad = 1e-8
        self.max_cycle = 80

        provider = FFTDF(cell, kpts=kpts)
        if mf_hf is None:
            mf_hf = DFKRHF(cell, kpts=kpts, exxdiv=exxdiv)
            mf_hf.with_df = provider
        elif not isinstance(mf_hf, khf.KRHF):
            raise TypeError('mf_hf must be a pyscf.pbc.scf.khf.KRHF object')
        else:
            if mf_hf.cell is not cell:
                raise ValueError('mf_hf and KRHFDFT must reference the same Cell')
            if not isinstance(mf_hf.with_df, (FFTDF, GDF)):
                raise TypeError(
                    'mf_hf.with_df must be a dcforge CPU PBC FFTDF or GDF')
            check_kpts_equal(
                self._kpts_runtime_reference, mf_hf.kpts,
                'source KRHF runtime kpts')
            provider = mf_hf.with_df

        if provider.cell is not cell:
            raise ValueError('DF provider must own the active Cell')
        check_kpts_equal(
            self._kpts_runtime_reference, provider.kpts,
            'density-fitting provider runtime kpts')
        if mf_hf.exxdiv != self.exxdiv:
            raise ValueError('source and target exxdiv conventions must match')

        if kpts_weights is None:
            kpts_weights = equal_kpts_weights(kpts)
        self.kpts_weights = check_equal_weights(kpts, kpts_weights).copy()
        self.kpts_wrap_around = kpts_wrap_around
        self.kpts_fingerprint = kpoint_fingerprint(
            kpts, self.kpts_weights, wrap_around=kpts_wrap_around)
        self._kpts_reference = kpts.copy()
        self._source_kpts_runtime_reference = np.asarray(mf_hf.kpts).copy()
        self._provider_kpts_runtime_reference = np.asarray(provider.kpts).copy()
        self.mf_hf = mf_hf
        self.with_df = self.mf_hf.with_df = provider
        self.e_hf = None
        self.e_dft = None
        self.dm_hf = None
        self.target_fock = None
        self.target_fock_signature = None
        self.source_gap = None
        self.source_state_signature = None
        self._source_state_receipt = None

    def kernel_hf(self, dm0=None, **kwargs):
        '''Run the shared-provider source KRHF calculation.'''
        _check_input(self)
        self._copy_hf_options(self, self.mf_hf)
        e_hf = self.mf_hf.kernel(dm0=dm0, **kwargs)
        if not self.mf_hf.converged:
            raise RuntimeError('periodic source KRHF calculation did not converge')
        self._sync_hf_results()
        record_pbc_source_state(self)
        return e_hf

    def run_hf(self, dm0=None, **kwargs):
        '''Run source KRHF and return this target workflow object.'''
        self.kernel_hf(dm0=dm0, **kwargs)
        return self

    def make_hf_rdm1(self):
        '''Return the converged complex Hermitian KRHF density blocks.'''
        assert_pbc_source_state(self)
        return validate_kresolved_hermitian(
            self.mf_hf.make_rdm1(), len(self.kpts), self.cell.nao_nr(),
            name='source KRHF density')

    def kernel(self, dm0=None, run_hf=True, **kwargs):
        r'''Run KRHF and evaluate :math:`E_{\rm DFA}[P_{\rm KRHF}]`.'''
        _check_input(self)
        if run_hf:
            self.kernel_hf(dm0=dm0, **kwargs)
        else:
            if dm0 is not None or kwargs:
                raise ValueError('dm0/kwargs require run_hf=True')
            assert_pbc_source_state(self)
            self._sync_hf_results()

        # The branch above already validated the canonical state receipt.
        self.dm_hf = np.asarray(validate_kresolved_hermitian(
            self.mf_hf.make_rdm1(), len(self.kpts), self.cell.nao_nr(),
            name='source KRHF density'))
        h1e = self.get_hcore(self.cell, self.kpts)
        vhf = self.get_veff(self.cell, self.dm_hf, kpts=self.kpts)
        self.e_coul = _real_scalar(vhf.ecoul, 'target Coulomb energy')
        self.e_xc = _real_scalar(vhf.exc, 'target XC energy')
        self.target_fock = np.asarray(h1e) + np.asarray(vhf)
        validate_kresolved_hermitian(
            self.target_fock, len(self.kpts), self.cell.nao_nr(),
            name='target DFA Fock')
        self.target_fock_signature = _target_fock_signature(self, self.dm_hf)
        self.e_dft = _real_scalar(
            self.energy_tot(dm=self.dm_hf, h1e=h1e, vhf=vhf),
            'target fixed-density energy')
        self.e_tot = self.e_dft
        from dcforge.tools.logger import energy_summary
        energy_summary(self)
        return self.e_dft

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        '''Evaluate a supplied-density sampled-k target-DFA energy.'''
        _check_input(self)
        if dm is None:
            dm = self.make_hf_rdm1()
            self._sync_hf_results()
        return krks.KRKS.energy_tot(self, dm, h1e, vhf)

    @classmethod
    def from_hf_state(cls, path, *, xc='PBE'):
        '''Construct this CPU sampled-k workflow from a checkpoint.'''
        state = load_pbc_hf_state(path)
        sampling = state.metadata['sampling']
        if sampling['kind'] != 'sampled_k':
            raise PBCStateError('KRHFDFT requires a sampled-k checkpoint')
        mf = cls(
            state.cell, kpts=np.asarray(sampling['kpts_bohr_inverse']),
            xc=xc, kpts_weights=np.asarray(sampling['weights']),
            kpts_wrap_around=sampling.get('wrap_around'),
            exxdiv=sampling.get('exxdiv'))
        mf = configure_pbc_provider(mf, state)
        mf.restore_hf_state(state)
        return mf

    def save_hf_state(self, path, *, extra_provenance=None):
        '''Save the current gradient-ready source KRHF state.'''
        return save_pbc_hf_state(
            path, self, extra_provenance=extra_provenance)

    def restore_hf_state(self, state_or_path):
        '''Restore and physically validate one source KRHF checkpoint.'''
        restore_pbc_source_state(self, state_or_path)
        return self

    @property
    def source_validation(self):
        '''Return the current immutable source-state receipt or ``None``.'''
        return self._source_state_receipt

    def get_hcore(self, cell=None, kpts=None):
        '''Return the source-KRHF one-electron Hamiltonian at all k-points.'''
        if cell is None:
            cell = self.cell
        if kpts is None:
            kpts = self.kpts
        check_kpts_equal(self.kpts, kpts, 'requested Hcore kpts')
        return self.mf_hf.get_hcore(cell=cell, kpts=kpts)

    def fft_density_fit(self, mesh=None):
        '''Replace source and target by one sampled-k FFTDF provider.'''
        return self._set_provider(FFTDF(
            self.cell, mesh=mesh, kpts=self._kpts_reference))

    def density_fit(self, auxbasis=None, with_df=None):
        '''Replace source and target by one sampled-k GDF provider.'''
        if with_df is None:
            with_df = GDF(
                self.cell, kpts=self._kpts_reference, auxbasis=auxbasis)
        elif not isinstance(with_df, GDF):
            raise TypeError('with_df must be a dcforge CPU PBC GDF')
        elif auxbasis is not None:
            raise ValueError('auxbasis and with_df cannot both be supplied')
        return self._set_provider(with_df)

    def multigrid_numint(self, mesh=None, ntasks=None, ke_ratio=None,
                         rel_cutoff=None):
        '''Reject PySCF 2.14's unavailable CPU sampled-k MultiGrid route.

        Pair-MultiGrid rejects non-Gamma density collocation and AO pass-2.
        This method fails before source SCF rather than exposing a workflow
        that would stop only after the expensive KRHF calculation.
        '''
        raise NotImplementedError(
            'CPU sampled-k multigrid is unavailable because PySCF 2.14 '
            'pair-multigrid density collocation and AO pass-2 kernels are '
            'Gamma-only; use the sampled-k uniform-grid XC route')

    def nuc_grad_method(self):
        '''Construct the sampled-k FFTDF/GDF analytic gradient object.'''
        from dcforge.cpu.pbc.grad.krhf_dft import Gradients
        return Gradients(self)

    Gradients = nuc_grad_method

    def as_scanner(self):
        '''Reject sampled-k scanners outside the current energy slice.'''
        raise NotImplementedError('sampled-k KRHFDFT scanner is not implemented')

    def reset(self, cell=None):
        '''Invalidate source and target state without changing the Cell.'''
        if cell is not None and cell is not self.cell:
            raise NotImplementedError('sampled-k Cell replacement is not supported')
        krks.KRKS.reset(self, self.cell)
        self.mf_hf.reset(self.cell)
        self._invalidate_results()
        return self

    def _sync_hf_results(self):
        mf_hf = self.mf_hf
        self.source_gap = validate_insulating_occupations(
            mf_hf.mo_energy, mf_hf.mo_occ)
        self.source_state_signature = (
            _array_digest(mf_hf.mo_occ),
            _array_digest(mf_hf.make_rdm1()),
        )
        self.e_hf = _real_scalar(mf_hf.e_tot, 'source KRHF energy')
        self.e_dft = self.dm_hf = self.target_fock = None
        self.target_fock_signature = None
        self.e_tot = self.e_hf
        self.mo_energy = mf_hf.mo_energy
        self.mo_coeff = mf_hf.mo_coeff
        self.mo_occ = mf_hf.mo_occ
        self.converged = mf_hf.converged
        self.cycles = getattr(mf_hf, 'cycles', 0)

    def _copy_hf_options(self, source, target):
        for key in self._hf_options:
            if hasattr(source, key) and hasattr(target, key):
                setattr(target, key, getattr(source, key))

    def _set_provider(self, provider):
        '''Attach one provider to source and target after identity checks.'''
        if provider.cell is not self.cell:
            raise ValueError('density-fitting provider must own the active Cell')
        check_kpts_equal(
            self._kpts_runtime_reference, provider.kpts,
            'density-fitting provider runtime kpts')
        self.with_df = self.mf_hf.with_df = provider
        self._provider_kpts_runtime_reference = np.asarray(
            provider.kpts).copy()
        self._invalidate_results()
        return self

    def _invalidate_results(self):
        self.e_hf = self.e_dft = self.dm_hf = self.target_fock = None
        self.target_fock_signature = None
        self.source_gap = None
        self.source_state_signature = None
        self.converged = False
        clear_pbc_source_state(self)


def energy_tot(mf_hf, mf_dft):
    '''Evaluate sampled-k target DFT energy with a converged KRHF density.'''
    if not mf_hf.converged:
        raise RuntimeError('periodic source KRHF calculation must be converged')
    if mf_hf.cell is not mf_dft.cell:
        raise ValueError('source and target must reference the same Cell')
    check_kpts_equal(mf_dft.kpts, mf_hf.kpts, 'source KRHF kpts')
    return _real_scalar(
        mf_dft.energy_tot(dm=mf_hf.make_rdm1()),
        'target fixed-density energy')


def _array_digest(value):
    '''Return a shape-, dtype-, and byte-sensitive ndarray digest.'''
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode('ascii'))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _target_fock_signature(mf, dm):
    '''Return immutable state defining reusable sampled-k target Fock blocks.'''
    provider = mf.with_df
    return (
        str(mf.xc), str(mf.exxdiv), id(provider), mf.kpts_fingerprint,
        np.asarray(mf.cell.atom_coords()).tobytes(),
        np.asarray(mf.cell.lattice_vectors()).tobytes(),
        float(mf.cell.precision),
        np.asarray(mf.cell.mesh, dtype=int).tobytes(),
        _provider_signature(provider),
        type(mf._numint).__module__, type(mf._numint).__name__,
        type(mf.grids).__module__, type(mf.grids).__name__,
        np.asarray(mf.grids.mesh, dtype=int).tobytes(),
        mf.source_state_signature,
        _array_digest(dm),
    )


def _provider_signature(provider):
    '''Return immutable FFTDF/GDF representation-defining provider state.'''
    mesh = getattr(provider, 'mesh', None)
    mesh_signature = (
        None if mesh is None else np.asarray(mesh, dtype=int).tobytes())
    return (
        type(provider).__module__, type(provider).__name__,
        mesh_signature, repr(getattr(provider, 'auxbasis', None)),
        float(getattr(provider, 'linear_dep_threshold', 0.0)),
        repr(getattr(provider, 'exp_to_discard', None)),
        bool(getattr(provider, 'j2c_eig_always', False)),
    )
