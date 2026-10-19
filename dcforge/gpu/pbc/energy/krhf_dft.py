'''GPU full-BZ semilocal-DFA@KRHF energy for FFTDF and GDF.'''

import hashlib

import cupy as cp
import numpy as np
from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.pbc.dft import gen_grid, krks, multigrid_v2
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.scf import khf
from pyscf.pbc.tools.k2gamma import kpts_to_kmesh

from dcforge.cpu.pbc.kpts import (
    IMAGINARY_RESIDUAL_TOL,
    check_equal_weights,
    check_kpts_equal,
    equal_kpts_weights,
    kpoint_fingerprint,
    normalize_kpts,
    validate_insulating_occupations,
)
from dcforge.gpu.pbc import multigrid as dc_multigrid
from dcforge.gpu.pbc._compat import contiguous_int1e_hermitian
from dcforge.gpu.pbc.df.fft import FFTDF
from dcforge.gpu.pbc.df.gdf import GDF
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

_SUPPORTED_MGGA_TARGET_XC = ('R2SCAN',)


def _target_xc_name(mf):
    '''Return the canonical sampled-k target-DFA name.'''
    return str(mf.xc).replace(' ', '').upper()


def _real_scalar(value, name, tol=IMAGINARY_RESIDUAL_TOL):
    '''Return a physical real scalar after checking its imaginary residual.'''
    if hasattr(value, 'get'):
        value = value.get()
    value = np.asarray(value).item()
    if abs(np.imag(value)) > tol:
        raise ValueError(
            f'{name} imaginary residual {abs(np.imag(value)):.3e} exceeds '
            f'{tol:.3e}')
    return float(np.real(value))


class DFKRHF(khf.KRHF):
    '''GPU KRHF whose J/K is owned by one explicit FFTDF or GDF provider.'''

    _keys = khf.KRHF._keys | {
        'int1e_layout_copy_count', 'int1e_layout_copy_bytes',
        'kpts_reference', 'kpts_runtime_reference',
        'kpts_roundtrip_max_abs',
    }

    def __init__(self, cell, kpts=None, exxdiv='ewald'):
        kpts = normalize_kpts(kpts)
        khf.KRHF.__init__(self, cell, kpts=kpts, exxdiv=exxdiv)
        self.kpts_reference = kpts.copy()
        self.kpts_runtime_reference = np.asarray(self.kpts).copy()
        self.kpts_roundtrip_max_abs = float(np.max(np.abs(
            self.kpts_runtime_reference - self.kpts_reference)))
        self.int1e_layout_copy_count = 0
        self.int1e_layout_copy_bytes = 0

    def get_ovlp(self, cell=None, kpts=None):
        '''Return sampled-k overlap with the localized layout adapter.'''
        with contiguous_int1e_hermitian(self):
            return khf.KRHF.get_ovlp(self, cell=cell, kpts=kpts)

    def get_hcore(self, cell=None, kpts=None):
        '''Return sampled-k Hcore using the active provider convention.'''
        if isinstance(self.with_df, GDF):
            if cell is None:
                cell = self.cell
            if kpts is None:
                kpts = self.kpts
            if cell is not self.with_df.cell:
                raise ValueError(
                    'Gaussian Hcore and provider must own the same Cell')
            with contiguous_int1e_hermitian(self):
                return self.with_df.get_pp(kpts) + int1e.int1e_kin(cell, kpts)
        with contiguous_int1e_hermitian(self):
            return khf.KRHF.get_hcore(self, cell=cell, kpts=kpts)

    def get_veff(self, cell=None, dm_kpts=None, dm_last=None, vhf_last=None,
                 hermi=1, kpts=None, kpts_band=None):
        '''Return source J/K with the localized sampled-k layout adapter.'''
        with contiguous_int1e_hermitian(self):
            return khf.KRHF.get_veff(
                self, cell, dm_kpts, dm_last, vhf_last, hermi,
                kpts, kpts_band)


FFTDFKRHF = DFKRHF


def _check_input(mf):
    '''Validate the current GPU full-BZ fixed-density energy capability.'''
    cell = mf.cell
    if cell.dimension != 3:
        raise NotImplementedError('GPU sampled-k KRHFDFT requires a 3D cell')
    if cell.spin != 0 or cell.nelectron % 2:
        raise NotImplementedError(
            'GPU sampled-k KRHFDFT requires a closed-shell cell')
    if not cell._pseudo or cell._ecp:
        raise NotImplementedError(
            'GPU sampled-k KRHFDFT currently requires GTH pseudo')
    if not isinstance(mf.with_df, (FFTDF, GDF)):
        raise NotImplementedError(
            'GPU sampled-k KRHFDFT currently supports FFTDF and GDF only')
    if mf.with_df is not mf.mf_hf.with_df:
        raise ValueError(
            'source KRHF and target KRKS must share one GPU DF provider')
    if mf.with_df.cell is not cell or mf.mf_hf.cell is not cell:
        raise ValueError('source, target and provider must own the same Cell')

    kpts = check_kpts_equal(
        mf._kpts_runtime_reference, mf.kpts,
        'target GPU KRKS runtime kpts')
    check_kpts_equal(
        kpts, mf.mf_hf.kpts_runtime_reference,
        'source GPU KRHF runtime kpts')
    check_kpts_equal(
        mf._kpts_reference, mf.mf_hf.kpts_reference,
        'source GPU KRHF public kpts')
    check_kpts_equal(
        mf._kpts_reference, mf.with_df.kpts_reference,
        'GPU DF provider kpts')
    check_equal_weights(kpts, mf.kpts_weights)
    if kpoint_fingerprint(
            mf._kpts_reference, mf.kpts_weights,
            wrap_around=mf.kpts_wrap_around) != mf.kpts_fingerprint:
        raise ValueError(
            'sampled-k coordinates, ordering, weights or wrap metadata '
            'changed; construct a new GPU KRHFDFT object')
    if mf.exxdiv != mf.mf_hf.exxdiv:
        raise ValueError('source and target exxdiv conventions must match')
    if (isinstance(mf.with_df, FFTDF) and not np.array_equal(
            np.asarray(mf.with_df.mesh), np.asarray(cell.mesh))):
        raise ValueError(
            'GPU sampled-k FFTDF energy requires with_df.mesh == cell.mesh')
    if isinstance(mf.with_df, GDF):
        _check_gdf_canonical_kpoint_order(mf)
    if not isinstance(mf.grids, gen_grid.UniformGrids):
        raise NotImplementedError(
            'GPU sampled-k target XC currently requires UniformGrids')
    if isinstance(mf._numint, multigrid_v2.MultiGridNumInt):
        if cell.ke_cutoff is None or cell.ke_cutoff <= 0:
            raise ValueError(
                'GPU sampled-k multigrid XC requires a positive '
                'cell.ke_cutoff set before cell.build()')
        if mf._numint.cell is not cell:
            raise ValueError(
                'GPU sampled-k multigrid XC and KRHFDFT must own the same Cell')
        if not np.array_equal(
                np.asarray(mf._numint.mesh), np.asarray(cell.mesh)):
            raise ValueError(
                'GPU sampled-k multigrid XC mesh must equal cell.mesh')
    elif mf.multigrid_with_j:
        raise ValueError(
            'multigrid_with_j=True requires multigrid_numint(with_j=True)')
    if mf._numint.libxc.is_hybrid_xc(mf.xc):
        raise NotImplementedError(
            'GPU sampled-k target functionals must be pure semilocal DFAs')
    if mf._numint.libxc.is_nlc(mf.xc) or mf.nlc not in ('', 0, False):
        raise NotImplementedError('GPU sampled-k NLC/VV10 is not supported')
    xctype = mf._numint._xc_type(mf.xc)
    if xctype not in ('LDA', 'GGA', 'MGGA'):
        raise NotImplementedError(
            'GPU sampled-k target functional must be LDA, GGA or R2SCAN')
    if (xctype == 'MGGA'
            and _target_xc_name(mf) not in _SUPPORTED_MGGA_TARGET_XC):
        raise NotImplementedError(
            'the GPU sampled-k MGGA target functional must be R2SCAN')


def _check_gdf_canonical_kpoint_order(mf):
    '''Fail before GPU4PySCF GDF builds an unsupported reordered CDERI.'''
    kpts = np.asarray(mf._kpts_reference)
    kmesh = kpts_to_kmesh(
        mf.cell, kpts, rcut=mf.cell.rcut + 10,
        bound_by_supmol=False)
    if mf.kpts_wrap_around is None:
        wrap_options = (False, True)
    else:
        wrap_options = (bool(mf.kpts_wrap_around),)
    canonical = [
        np.asarray(mf.cell.make_kpts(kmesh, wrap_around=wrap))
        for wrap in wrap_options]
    if not any(np.allclose(kpts, value, atol=1e-12, rtol=0)
               for value in canonical):
        raise NotImplementedError(
            'GPU4PySCF 1.8.1 sampled-k GDF requires canonical '
            'cell.make_kpts ordering; reordered full-BZ arrays can fail '
            'inside the CDERI builder')


class KRHFDFT(krks.KRKS):
    r'''Evaluate :math:`E_{\rm DFA}[P_{\rm KRHF}]` on a full-BZ k mesh.

    Args:
        cell : pyscf.pbc.gto.Cell
            Built three-dimensional closed-shell GTH cell.
        kpts : array_like
            Explicit full-BZ Cartesian k-points with shape ``(nkpts,3)``.

    Kwargs:
        xc : str
            Pure LDA/GGA functional or ``'R2SCAN'``.
        mf_hf : DFKRHF or None
            Optional converged-source owner using the same Cell and k-points.
        kpts_weights : array_like or None
            Equal full-BZ weights.  IBZ/nonuniform weights are unsupported.
        kpts_wrap_around : bool or None
            Provenance for the supplied explicit k-point array.
        exxdiv : str
            Source/target exact-exchange divergence convention.

    Notes:
        ``kernel`` runs only source KRHF, then evaluates the inherited KRKS
        supplied-density energy.  It never optimizes target-DFA orbitals.
        GPU sampled-k FFTDF and GDF analytic gradients use the same complex
        real-linear Z-vector and explicit full-BZ weights.
    '''

    _keys = krks.KRKS._keys | {
        'mf_hf', 'e_hf', 'e_dft', 'dm_hf', 'target_fock',
        'target_fock_signature', 'kpts_weights', 'kpts_wrap_around',
        'kpts_fingerprint', 'source_gap', 'source_state_signature',
        'state_validation_host_bytes', 'scientific_host_fallback',
        'host_fallback_components', 'int1e_layout_copy_count',
        'int1e_layout_copy_bytes', 'kpts_roundtrip_max_abs',
        'multigrid_backend', 'multigrid_with_j',
        '_kpts_runtime_reference', '_kpts_reference',
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
        elif not isinstance(mf_hf, DFKRHF):
            raise TypeError('mf_hf must be a dcforge GPU PBC DFKRHF object')
        else:
            if mf_hf.cell is not cell:
                raise ValueError(
                    'mf_hf and GPU KRHFDFT must reference the same Cell')
            if not isinstance(mf_hf.with_df, (FFTDF, GDF)):
                raise TypeError(
                    'mf_hf.with_df must be a dcforge GPU PBC FFTDF or GDF')
            check_kpts_equal(
                kpts, mf_hf.kpts_reference, 'source GPU KRHF kpts')
            provider = mf_hf.with_df

        check_kpts_equal(
            kpts, provider.kpts_reference, 'GPU DF provider kpts')
        if mf_hf.exxdiv != self.exxdiv:
            raise ValueError('source and target exxdiv conventions must match')
        if kpts_weights is None:
            kpts_weights = equal_kpts_weights(kpts)
        self.kpts_weights = check_equal_weights(kpts, kpts_weights).copy()
        self.kpts_wrap_around = kpts_wrap_around
        self.kpts_fingerprint = kpoint_fingerprint(
            kpts, self.kpts_weights, wrap_around=kpts_wrap_around)
        self._kpts_reference = kpts.copy()
        self.mf_hf = mf_hf
        self.with_df = self.mf_hf.with_df = provider
        self.e_hf = self.e_dft = self.dm_hf = self.target_fock = None
        self.target_fock_signature = None
        self.source_gap = None
        self.source_state_signature = None
        self.state_validation_host_bytes = 0
        self.scientific_host_fallback = True
        self.host_fallback_components = (
            'PySCF non-Gamma GTH nonlocal-projector half-integrals',)
        self.int1e_layout_copy_count = 0
        self.int1e_layout_copy_bytes = 0
        self.multigrid_backend = None
        self.multigrid_with_j = False
        self._source_state_receipt = None

    @property
    def device(self):
        '''Return the scientific-array owner without transferring state.'''
        return 'gpu'

    def kernel_hf(self, dm0=None, **kwargs):
        '''Run the shared-provider source GPU KRHF calculation.'''
        _check_input(self)
        self._copy_hf_options(self, self.mf_hf)
        e_hf = self.mf_hf.kernel(dm0=dm0, **kwargs)
        if not self.mf_hf.converged:
            raise RuntimeError('GPU periodic source KRHF did not converge')
        self._sync_hf_results()
        record_pbc_source_state(self)
        return _real_scalar(e_hf, 'source GPU KRHF energy')

    def run_hf(self, dm0=None, **kwargs):
        '''Run source GPU KRHF and return this target workflow.'''
        self.kernel_hf(dm0=dm0, **kwargs)
        return self

    def make_hf_rdm1(self):
        '''Return converged complex Hermitian device density blocks.'''
        assert_pbc_source_state(self)
        return _validate_device_hermitian(
            cp.asarray(self.mf_hf.make_rdm1()), len(self.kpts),
            self.cell.nao_nr(), name='source GPU KRHF density')

    def kernel(self, dm0=None, run_hf=True, **kwargs):
        r'''Run GPU KRHF and evaluate :math:`E_{\rm DFA}[P_{\rm KRHF}]`.'''
        _check_input(self)
        if run_hf:
            self.kernel_hf(dm0=dm0, **kwargs)
        else:
            if dm0 is not None or kwargs:
                raise ValueError('dm0/kwargs require run_hf=True')
            assert_pbc_source_state(self)
            self._sync_hf_results()

        # The branch above already validated the canonical state receipt.
        self.dm_hf = _validate_device_hermitian(
            cp.asarray(self.mf_hf.make_rdm1()), len(self.kpts),
            self.cell.nao_nr(), name='source GPU KRHF density')
        h1e = self.get_hcore(self.cell, self.kpts)
        vhf = self.get_veff(self.cell, self.dm_hf, kpts=self.kpts)
        self.e_coul = _real_scalar(vhf.ecoul, 'target GPU Coulomb energy')
        self.e_xc = _real_scalar(vhf.exc, 'target GPU XC energy')
        self.target_fock = cp.asarray(h1e) + cp.asarray(vhf)
        _validate_device_hermitian(
            self.target_fock, len(self.kpts), self.cell.nao_nr(),
            name='target GPU DFA Fock')
        self.target_fock_signature = _target_fock_signature(
            self, self.dm_hf)
        self.e_dft = _real_scalar(
            self.energy_tot(dm=self.dm_hf, h1e=h1e, vhf=vhf),
            'target GPU fixed-density energy')
        self.e_tot = self.e_dft
        self._sync_backend_metadata()
        from dcforge.tools.logger import energy_summary
        energy_summary(self)
        return self.e_dft

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        '''Evaluate a supplied-density GPU sampled-k target-DFA energy.'''
        _check_input(self)
        if dm is None:
            dm = self.make_hf_rdm1()
            self._sync_hf_results()
        return krks.KRKS.energy_tot(self, dm, h1e, vhf)

    @classmethod
    def from_hf_state(cls, path, *, xc='PBE'):
        '''Construct this GPU sampled-k workflow from a checkpoint.'''
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
        '''Return the source-KRHF device Hcore at all sampled k-points.'''
        if cell is None:
            cell = self.cell
        if kpts is None:
            kpts = self.kpts
        check_kpts_equal(self.kpts, kpts, 'requested GPU Hcore kpts')
        return self.mf_hf.get_hcore(cell=cell, kpts=kpts)

    def get_veff(self, cell=None, dm=None, dm_last=None, vhf_last=None,
                 hermi=1, kpts=None, kpts_band=None):
        '''Return uniform-grid XC plus provider-owned sampled-k Hartree.'''
        if not isinstance(self._numint, multigrid_v2.MultiGridNumInt):
            with contiguous_int1e_hermitian(self):
                return krks.KRKS.get_veff(
                    self, cell, dm, dm_last, vhf_last, hermi,
                    kpts, kpts_band)
        if self.multigrid_with_j:
            raise NotImplementedError(
                'GPU sampled-k multigrid J+XC is not yet validated')
        if cell is None:
            cell = self.cell
        if dm is None:
            dm = self.make_rdm1()
        if kpts is None:
            kpts = self.kpts
        check_kpts_equal(self.kpts, kpts, 'sampled-k multigrid target kpts')
        ni = self._numint
        _nelec, exc, vxc = ni.nr_rks(
            cell, self.grids, self.xc, dm, 0, hermi,
            kpts, kpts_band, with_j=False, verbose=self.verbose)
        with contiguous_int1e_hermitian(self.mf_hf):
            vj, _vk = self.mf_hf.get_jk(
                cell=cell, dm_kpts=cp.asarray(dm), hermi=hermi,
                kpts=kpts, kpts_band=kpts_band,
                with_j=True, with_k=False)
        if vj is None:
            raise RuntimeError(
                'active sampled-k DF provider did not return target Hartree J')
        vj = cp.asarray(vj)
        ecoul = None
        if kpts_band is None:
            ecoul = .5 * cp.einsum(
                'kij,kji->', cp.asarray(dm), vj).real / len(kpts)
        return tag_array(cp.asarray(vxc) + vj, ecoul=ecoul, exc=exc)

    def fft_density_fit(self, mesh=None):
        '''Replace source and target by one GPU sampled-k FFTDF provider.'''
        return self._set_provider(FFTDF(
            self.cell, mesh=mesh, kpts=self._kpts_reference))

    def density_fit(self, auxbasis=None, with_df=None):
        '''Replace source and target by one GPU sampled-k GDF provider.'''
        if with_df is None:
            with_df = GDF(
                self.cell, kpts=self._kpts_reference, auxbasis=auxbasis)
        elif not isinstance(with_df, GDF):
            raise TypeError('with_df must be a dcforge GPU PBC GDF')
        elif auxbasis is not None:
            raise ValueError('auxbasis and with_df cannot both be supplied')
        return self._set_provider(with_df)

    def multigrid_numint(self, mesh=None, ntasks=None, ke_ratio=None,
                         rel_cutoff=None, with_j=False):
        '''Return a copy whose target XC uses GPU sampled-k MultiGrid.

        Source KRHF, source CPHF, and target Hartree remain owned by the
        shared FFTDF/GDF provider.  Only target semilocal XC is delegated to
        GPU4PySCF multigrid-v2.  Sampled-k J+XC is deliberately deferred.
        '''
        if mesh is not None:
            raise ValueError(
                'set cell.ke_cutoff before cell.build(); a separate GPU '
                'sampled-k multigrid mesh is not supported')
        if any(value is not None for value in (ntasks, ke_ratio, rel_cutoff)):
            raise NotImplementedError(
                'GPU4PySCF multigrid-v2 does not expose ntasks, ke_ratio, '
                'or rel_cutoff controls')
        if not isinstance(with_j, (bool, np.bool_)):
            raise TypeError('multigrid with_j must be boolean')
        if with_j:
            raise NotImplementedError(
                'GPU sampled-k multigrid J+XC is not yet validated; '
                'use with_j=False')
        if self.cell.ke_cutoff is None or self.cell.ke_cutoff <= 0:
            raise ValueError(
                'multigrid_numint requires a positive cell.ke_cutoff set '
                'before cell.build()')
        mf = self.copy()
        ni = dc_multigrid.MultiGridNumInt(self.cell)
        ni.mesh = np.asarray(self.cell.mesh, dtype=int).copy()
        mf._numint = ni
        mf.multigrid_with_j = False
        mf.multigrid_backend = 'gpu4pyscf_multigrid_v2_xc_only'
        mf.e_dft = mf.dm_hf = mf.target_fock = None
        mf.target_fock_signature = None
        if mf.e_hf is not None:
            mf.e_tot = mf.e_hf
        return mf

    def nuc_grad_method(self):
        '''Construct the sampled-k FFTDF or GDF analytic gradient object.'''
        from dcforge.gpu.pbc.grad.krhf_dft import Gradients
        return Gradients(self)

    Gradients = nuc_grad_method

    def as_scanner(self):
        '''Reject scanner state reuse outside the current energy slice.'''
        raise NotImplementedError('GPU sampled-k scanner is not implemented')

    def reset(self, cell=None):
        '''Invalidate source and target state without changing the Cell.'''
        if cell is not None and cell is not self.cell:
            raise NotImplementedError(
                'GPU sampled-k Cell replacement is not supported')
        krks.KRKS.reset(self, self.cell)
        self.mf_hf.reset(self.cell)
        self._invalidate_results()
        return self

    def _sync_hf_results(self):
        mo_energy = cp.asnumpy(cp.asarray(self.mf_hf.mo_energy))
        mo_occ = cp.asnumpy(cp.asarray(self.mf_hf.mo_occ))
        self.state_validation_host_bytes += mo_energy.nbytes + mo_occ.nbytes
        self.source_gap = validate_insulating_occupations(mo_energy, mo_occ)
        density_digest, density_bytes = _device_array_digest(
            self.mf_hf.make_rdm1())
        occ_digest, occ_bytes = _device_array_digest(self.mf_hf.mo_occ)
        self.state_validation_host_bytes += density_bytes + occ_bytes
        self.source_state_signature = (occ_digest, density_digest)
        self.e_hf = _real_scalar(self.mf_hf.e_tot, 'source GPU KRHF energy')
        self.e_dft = self.dm_hf = self.target_fock = None
        self.target_fock_signature = None
        self.e_tot = self.e_hf
        self.mo_energy = self.mf_hf.mo_energy
        self.mo_coeff = self.mf_hf.mo_coeff
        self.mo_occ = self.mf_hf.mo_occ
        self.converged = self.mf_hf.converged
        self.cycles = getattr(self.mf_hf, 'cycles', 0)
        self._sync_backend_metadata()

    def _sync_backend_metadata(self):
        self.int1e_layout_copy_count = int(
            self.mf_hf.int1e_layout_copy_count)
        self.int1e_layout_copy_bytes = int(
            self.mf_hf.int1e_layout_copy_bytes)
        components = [
            'PySCF non-Gamma GTH nonlocal-projector half-integrals']
        if isinstance(self.with_df, GDF):
            components.append('PySCF sampled-k GDF zero-order Hcore')
        self.host_fallback_components = tuple(components)

    def _copy_hf_options(self, source, target):
        for key in self._hf_options:
            if hasattr(source, key) and hasattr(target, key):
                setattr(target, key, getattr(source, key))

    def _set_provider(self, provider):
        if provider.cell is not self.cell:
            raise ValueError('GPU DF provider must own the active Cell')
        check_kpts_equal(
            self._kpts_reference, provider.kpts_reference,
            'GPU DF provider kpts')
        self.with_df = self.mf_hf.with_df = provider
        self._invalidate_results()
        return self

    def _invalidate_results(self):
        self.e_hf = self.e_dft = self.dm_hf = self.target_fock = None
        self.target_fock_signature = None
        self.source_gap = None
        self.source_state_signature = None
        self.state_validation_host_bytes = 0
        self.int1e_layout_copy_count = 0
        self.int1e_layout_copy_bytes = 0
        self.mf_hf.int1e_layout_copy_count = 0
        self.mf_hf.int1e_layout_copy_bytes = 0
        self.converged = False
        clear_pbc_source_state(self)


def energy_tot(mf_hf, mf_dft):
    '''Evaluate target GPU DFT energy with a converged KRHF density.'''
    if not mf_hf.converged:
        raise RuntimeError('GPU periodic source KRHF must be converged')
    if mf_hf.cell is not mf_dft.cell:
        raise ValueError('source and target must reference the same Cell')
    check_kpts_equal(mf_dft.kpts, mf_hf.kpts, 'source GPU KRHF kpts')
    return _real_scalar(
        mf_dft.energy_tot(dm=mf_hf.make_rdm1()),
        'target GPU fixed-density energy')


def _validate_device_hermitian(
        values, nkpts, nao, *, hermi_tol=1e-10,
        name='GPU k-resolved matrix'):
    '''Validate device shape, finiteness and Hermiticity.'''
    values = cp.asarray(values)
    if values.shape != (nkpts, nao, nao):
        raise ValueError(f'{name} must have shape (nkpts, nao, nao)')
    if not bool(cp.all(cp.isfinite(values)).get()):
        raise ValueError(f'{name} contains nonfinite values')
    residual = values - values.conj().transpose(0, 2, 1)
    error = float(cp.max(cp.abs(residual)).get()) if values.size else 0.0
    if error > hermi_tol:
        raise ValueError(
            f'{name} Hermiticity residual {error:.3e} exceeds '
            f'{hermi_tol:.3e}')
    return values


def _device_array_digest(value):
    '''Return a strict digest and explicit host-transfer byte count.'''
    host = np.ascontiguousarray(cp.asnumpy(cp.asarray(value)))
    digest = hashlib.sha256()
    digest.update(str(host.dtype).encode('ascii'))
    digest.update(np.asarray(host.shape, dtype=np.int64).tobytes())
    digest.update(host.tobytes())
    return digest.hexdigest(), int(host.nbytes)


def _target_fock_signature(mf, dm):
    '''Return immutable state defining reusable GPU target-Fock blocks.'''
    dm_digest, host_bytes = _device_array_digest(dm)
    mf.state_validation_host_bytes += host_bytes
    return (
        str(mf.xc), str(mf.exxdiv), id(mf.with_df), mf.kpts_fingerprint,
        np.asarray(mf.cell.atom_coords()).tobytes(),
        np.asarray(mf.cell.lattice_vectors()).tobytes(),
        float(mf.cell.precision),
        np.asarray(mf.cell.mesh, dtype=int).tobytes(),
        type(mf._numint).__module__, type(mf._numint).__name__,
        type(mf.grids).__module__, type(mf.grids).__name__,
        np.asarray(mf.grids.mesh, dtype=int).tobytes(),
        mf.source_state_signature, dm_digest,
    )
