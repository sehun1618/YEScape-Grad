'''Full-BZ sampled-k KRHF-DFT energy with a shared FFTDF/GDF provider.'''

import hashlib

import numpy as np
from pyscf.pbc.dft import gen_grid, krks, multigrid
from pyscf.pbc.scf import khf

from dcforge.cpu.pbc.df.fft import FFTDF
from dcforge.cpu.pbc.df.gdf import GDF
from dcforge.cpu.pbc.df.rsdf import RSDF
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

_SUPPORTED_MGGA_TARGET_XC = ('R2SCAN',)


def _target_xc_name(mf):
    '''Return the canonical sampled-k target-DFA name.'''
    return str(mf.xc).replace(' ', '').upper()


def _real_scalar(value, name, tol=IMAGINARY_RESIDUAL_TOL):
    '''Return a physical real scalar after checking its imaginary residual.'''
    value = np.asarray(value).item()
    if abs(np.imag(value)) > tol:
        raise ValueError(
            f'{name} imaginary residual {abs(np.imag(value)):.3e} exceeds '
            f'{tol:.3e}')
    return float(np.real(value))


def _check_input(mf):
    '''Validate the full-BZ semilocal-DFA@KRHF energy capability.'''
    cell = mf.cell
    if cell.dimension != 3:
        raise NotImplementedError('sampled-k KRHFDFT currently requires a 3D cell')
    if cell.spin != 0 or cell.nelectron % 2:
        raise NotImplementedError('sampled-k KRHFDFT requires a closed-shell cell')
    if not cell._pseudo or cell._ecp:
        raise NotImplementedError('sampled-k KRHFDFT currently requires GTH pseudo')
    if not isinstance(mf.with_df, (FFTDF, GDF, RSDF)):
        raise NotImplementedError(
            'sampled-k KRHFDFT supports CPU FFTDF, GDF and RSDF only')
    if mf.with_df is not mf.mf_hf.with_df:
        raise ValueError(
            'source KRHF and target KRKS must share one DF provider')
    if mf.with_df.cell is not cell or mf.mf_hf.cell is not cell:
        raise ValueError(
            'source, target and DF provider must own the same Cell')

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
            'sampled-k coordinates, ordering, weights or wrap metadata changed; '
            'construct a new KRHFDFT object')
    if mf.exxdiv != mf.mf_hf.exxdiv:
        raise ValueError('source and target exxdiv conventions must match')
    if (isinstance(mf.with_df, FFTDF) and not np.array_equal(
            np.asarray(mf.with_df.mesh), np.asarray(cell.mesh))):
        raise ValueError(
            'sampled-k FFTDF energy requires with_df.mesh == cell.mesh')

    if isinstance(
            mf._numint,
            (multigrid.MultiGridNumInt, multigrid.MultiGridNumInt2)):
        raise NotImplementedError(
            'CPU sampled-k multigrid is unavailable because PySCF 2.14 '
            'pair-multigrid density collocation and AO pass-2 kernels are '
            'Gamma-only; use the sampled-k uniform-grid XC route')
    if not isinstance(mf.grids, gen_grid.UniformGrids):
        raise NotImplementedError(
            'sampled-k target XC currently requires UniformGrids')
    if mf._numint.libxc.is_hybrid_xc(mf.xc):
        raise NotImplementedError(
            'sampled-k target functionals must be pure semilocal DFAs')
    if mf._numint.libxc.is_nlc(mf.xc) or mf.nlc not in ('', 0, False):
        raise NotImplementedError('sampled-k NLC/VV10 is not supported')
    xctype = mf._numint._xc_type(mf.xc)
    if xctype not in ('LDA', 'GGA', 'MGGA'):
        raise NotImplementedError(
            'sampled-k target functional must be pure LDA, GGA or R2SCAN')
    if (xctype == 'MGGA'
            and _target_xc_name(mf) not in _SUPPORTED_MGGA_TARGET_XC):
        raise NotImplementedError(
            'the first sampled-k MGGA target functional must be R2SCAN')

class DFKRHF(khf.KRHF):
    '''Full-BZ KRHF whose J/K operator is owned by one shared DF provider.'''


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
            Target pure LDA/GGA functional or ``'R2SCAN'``.  Hybrid and
            NLC/VV10 functionals are not supported.
        mf_hf : pyscf.pbc.scf.khf.KRHF or None
            Optional converged or unconverged source object.  It must own the
            same Cell, k-points and a DC-Forge CPU FFTDF or GDF provider.
        kpts_weights : array_like or None
            Full-BZ weights.  ``None`` creates ``1 / nkpts``.  Explicit
            values are accepted only when exactly equal to this convention.
        kpts_wrap_around : bool or None
            Provenance metadata recording how ``cell.make_kpts`` generated
            the explicit array.  It does not alter the supplied coordinates.
        exxdiv : str
            PySCF periodic exact-exchange divergence convention used by the
            source KRHF.  Source and target objects must match exactly.

    Notes:
        ``kernel`` runs only the source KRHF SCF.  The target KRKS orbitals
        are never optimized: PySCF's supplied-density ``get_veff`` and
        ``energy_tot`` evaluate the target DFA on the converged k-resolved
        HF density.
        PySCF already applies the ``1 / nkpts`` normalization to the
        one-electron, Hartree and XC energy components, so DC-Forge does not
        apply another k-point factor to those returned scalars.
    '''

    _keys = krks.KRKS._keys | {
        'mf_hf', 'e_hf', 'e_dft', 'dm_hf', 'target_fock',
        'target_fock_signature', 'kpts_weights', 'kpts_wrap_around',
        'kpts_fingerprint', 'source_gap', 'source_state_signature',
        'kpts_roundtrip_max_abs', '_kpts_runtime_reference',
        '_source_kpts_runtime_reference',
        '_provider_kpts_runtime_reference',
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
        self.with_df = provider
        if mf_hf is None:
            mf_hf = DFKRHF(cell, kpts=kpts, exxdiv=exxdiv)
            mf_hf.with_df = provider
        elif not isinstance(mf_hf, khf.KRHF):
            raise TypeError('mf_hf must be a pyscf.pbc.scf.khf.KRHF object')
        else:
            if mf_hf.cell is not cell:
                raise ValueError('mf_hf and KRHFDFT must reference the same Cell')
            if not isinstance(mf_hf.with_df, (FFTDF, GDF, RSDF)):
                raise TypeError(
                    'mf_hf.with_df must be a dcforge CPU PBC FFTDF, GDF '
                    'or RSDF')
            check_kpts_equal(
                self._kpts_runtime_reference, mf_hf.kpts,
                'source KRHF runtime kpts')
            provider = mf_hf.with_df
            self.with_df = provider

        if provider.cell is not cell:
            raise ValueError('DF provider must own the active Cell')
        check_kpts_equal(
            self._kpts_runtime_reference, provider.kpts,
            'DF provider runtime kpts')
        if mf_hf.exxdiv != self.exxdiv:
            raise ValueError('source and target exxdiv conventions must match')

        if kpts_weights is None:
            kpts_weights = equal_kpts_weights(kpts)
        self.kpts_weights = check_equal_weights(kpts, kpts_weights).copy()
        self.kpts_wrap_around = kpts_wrap_around
        self.kpts_fingerprint = kpoint_fingerprint(
            kpts, self.kpts_weights, wrap_around=kpts_wrap_around)
        self._kpts_reference = kpts.copy()
        self._source_kpts_runtime_reference = np.asarray(
            mf_hf.kpts).copy()
        self._provider_kpts_runtime_reference = np.asarray(
            provider.kpts).copy()
        self.mf_hf = mf_hf
        self.with_df = self.mf_hf.with_df = provider
        self.e_hf = None
        self.e_dft = None
        self.dm_hf = None
        self.target_fock = None
        self.target_fock_signature = None
        self.source_gap = None
        self.source_state_signature = None

    def kernel_hf(self, dm0=None, **kwargs):
        '''Run the shared-provider source KRHF calculation.'''
        _check_input(self)
        self._copy_hf_options(self, self.mf_hf)
        e_hf = self.mf_hf.kernel(dm0=dm0, **kwargs)
        if not self.mf_hf.converged:
            raise RuntimeError('periodic source KRHF calculation did not converge')
        self._sync_hf_results()
        return e_hf

    def run_hf(self, dm0=None, **kwargs):
        '''Run source KRHF and return this target workflow object.'''
        self.kernel_hf(dm0=dm0, **kwargs)
        return self

    def make_hf_rdm1(self):
        '''Return the converged complex Hermitian KRHF density blocks.'''
        if not self.mf_hf.converged:
            raise RuntimeError('periodic source KRHF must be converged')
        dm = self.mf_hf.make_rdm1()
        return validate_kresolved_hermitian(
            dm, len(self.kpts), self.cell.nao_nr(), name='source KRHF density')

    def kernel(self, dm0=None, run_hf=True, **kwargs):
        r'''Run KRHF and evaluate :math:`E_{\rm DFA}[P_{\rm KRHF}]`.

        No target ``KRKS.kernel`` call occurs.  ``run_hf=False`` may be used
        only when the attached source object is already converged.
        '''
        _check_input(self)
        if run_hf:
            self.kernel_hf(dm0=dm0, **kwargs)
        else:
            if dm0 is not None or kwargs:
                raise ValueError('dm0/kwargs require run_hf=True')
            if not self.mf_hf.converged:
                raise RuntimeError('run_hf=False requires converged source KRHF')
            self._sync_hf_results()

        self.dm_hf = np.asarray(self.make_hf_rdm1())
        h1e = self.get_hcore(self.cell, self.kpts)
        vhf = self.get_veff(self.cell, self.dm_hf, kpts=self.kpts)
        self.e_coul = _real_scalar(vhf.ecoul, 'target Coulomb energy')
        self.e_xc = _real_scalar(vhf.exc, 'target XC energy')
        self.target_fock = np.asarray(h1e) + np.asarray(vhf)
        validate_kresolved_hermitian(
            self.target_fock, len(self.kpts), self.cell.nao_nr(),
            name='target DFA Fock')
        self.target_fock_signature = _target_fock_signature(
            self, self.dm_hf)
        self.e_dft = _real_scalar(
            self.energy_tot(dm=self.dm_hf, h1e=h1e, vhf=vhf),
            'target fixed-density energy')
        self.e_tot = self.e_dft
        return self.e_dft

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        '''Evaluate a supplied-density sampled-k target-DFA energy.'''
        _check_input(self)
        return krks.KRKS.energy_tot(self, dm, h1e, vhf)

    def get_hcore(self, cell=None, kpts=None):
        '''Return the k-resolved one-electron Hamiltonian of source KRHF.'''
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
                self.cell, kpts=self._kpts_reference,
                auxbasis=auxbasis)
        elif not isinstance(with_df, GDF):
            raise TypeError('with_df must be a dcforge CPU PBC GDF')
        elif auxbasis is not None:
            raise ValueError('auxbasis and with_df cannot both be supplied')
        return self._set_provider(with_df)

    def rs_density_fit(self, auxbasis=None, with_df=None):
        '''Replace source and target by one sampled-k RSDF provider.'''
        if with_df is None:
            with_df = RSDF(
                self.cell, kpts=self._kpts_reference,
                auxbasis=auxbasis)
        elif not isinstance(with_df, RSDF):
            raise TypeError('with_df must be a dcforge CPU PBC RSDF')
        elif auxbasis is not None:
            raise ValueError('auxbasis and with_df cannot both be supplied')
        return self._set_provider(with_df)

    def multigrid_numint(self, mesh=None, ntasks=None, ke_ratio=None,
                         rel_cutoff=None):
        '''Reject the unavailable CPU sampled-k pair-multigrid route.

        PySCF 2.14 pair-multigrid explicitly rejects non-Gamma k-points in
        both density collocation and AO pass-2.  Keeping this PySCF-style
        method makes the capability boundary discoverable without exposing
        an object that fails only after an expensive source calculation.
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
        '''Reject sampled-k scanners outside the first fixed-cell release.'''
        raise NotImplementedError('sampled-k KRHFDFT scanner is not implemented')

    def reset(self, cell=None):
        '''Invalidate sampled-k source and target state without changing Cell.'''
        if cell is not None and cell is not self.cell:
            raise NotImplementedError(
                'sampled-k geometry scanners and Cell replacement are deferred')
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


def energy_tot(mf_hf, mf_dft):
    '''Evaluate a sampled-k target DFT energy with converged KRHF density.'''
    if not mf_hf.converged:
        raise RuntimeError('periodic source KRHF calculation must be converged')
    if mf_hf.cell is not mf_dft.cell:
        raise ValueError('source and target must reference the same Cell')
    check_kpts_equal(mf_dft.kpts, mf_hf.kpts, 'source KRHF kpts')
    return _real_scalar(
        mf_dft.energy_tot(dm=mf_hf.make_rdm1()),
        'target fixed-density energy')


def _array_digest(value):
    '''Return a shape-, dtype- and byte-sensitive ndarray digest.'''
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode('ascii'))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _target_fock_signature(mf, dm):
    '''Return immutable state defining reusable sampled-k target Fock blocks.'''
    provider = mf.with_df
    grids = mf.grids
    return (
        str(mf.xc), str(mf.exxdiv), id(provider),
        mf.kpts_fingerprint,
        np.asarray(mf.cell.atom_coords()).tobytes(),
        np.asarray(mf.cell.lattice_vectors()).tobytes(),
        float(mf.cell.precision),
        np.asarray(mf.cell.mesh, dtype=int).tobytes(),
        _provider_signature(provider),
        type(mf._numint).__module__, type(mf._numint).__name__,
        type(grids).__module__, type(grids).__name__,
        np.asarray(grids.mesh, dtype=int).tobytes(),
        mf.source_state_signature,
        _array_digest(dm),
    )

def _provider_signature(provider):
    '''Return immutable FFTDF/GDF representation-defining provider state.'''
    mesh = getattr(provider, 'mesh', None)
    if mesh is None:
        mesh_signature = None
    else:
        mesh_signature = np.asarray(mesh, dtype=int).tobytes()
    return (
        type(provider).__module__, type(provider).__name__,
        mesh_signature, repr(getattr(provider, 'auxbasis', None)),
        float(getattr(provider, 'linear_dep_threshold', 0.0)),
        repr(getattr(provider, 'exp_to_discard', None)),
        bool(getattr(provider, 'j2c_eig_always', False)),
    )
