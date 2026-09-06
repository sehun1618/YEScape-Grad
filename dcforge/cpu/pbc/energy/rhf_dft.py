'''Gamma-point periodic RHF-DFT energy with an explicit DF provider.'''

import hashlib
from collections.abc import Mapping

import numpy as np
from pyscf.pbc.dft import gen_grid, multigrid, rks
from pyscf.pbc.scf import hf

from dcforge.cpu.pbc.df.fft import FFTDF
from dcforge.cpu.pbc.df.gdf import GDF
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

# PySCF's pair-multigrid implementation uses four levels by default.  For
# the compact GTH basis/cell combinations supported by this route, the
# coarsest level contributes no resolved difference at the tested precision,
# while still paying one FFT and one C grid-integral traversal.  Keep the
# PySCF partition available through ``ntasks=4`` as the reference route.
_MULTIGRID_NTASKS_DEFAULT = 3
_MULTIGRID_KE_RATIO_DEFAULT = 3.0
_MULTIGRID_REL_CUTOFF_DEFAULT = 20.0
_SUPPORTED_TARGET_XC = ('PBE', 'R2SCAN')


def _target_xc_name(mf):
    '''Return the canonical name of the restricted periodic target DFA.'''
    return str(mf.xc).replace(' ', '').upper()


def _check_input(mf):
    '''Validate the restricted Gamma-point semilocal-DFA@RHF capability.'''
    cell = mf.cell
    if cell.dimension != 3:
        raise NotImplementedError('periodic RHF-DFT currently requires a 3D cell')
    if cell.spin != 0 or cell.nelectron % 2:
        raise NotImplementedError('periodic RHF-DFT requires a closed-shell cell')
    if not cell._pseudo or cell._ecp:
        raise NotImplementedError('periodic RHF-DFT currently requires GTH pseudo')
    if not isinstance(mf.with_df, (FFTDF, GDF)):
        raise NotImplementedError('unsupported periodic density-fitting provider')
    if mf.mf_hf.cell is not cell or mf.with_df.cell is not cell:
        raise ValueError(
            'source, target and density-fitting provider must own the same Cell')
    if mf.with_df is not mf.mf_hf.with_df:
        raise ValueError(
            'source RHF and target RKS must share one density-fitting provider')
    if isinstance(mf._numint, multigrid.MultiGridNumInt2):
        if cell.ke_cutoff is None or cell.ke_cutoff <= 0:
            raise ValueError(
                'multigrid XC requires a positive cell.ke_cutoff set '
                'before cell.build()')
        cutoff_mesh = cell.symmetrize_mesh(
            cell.cutoff_to_mesh(cell.ke_cutoff))
        if not np.array_equal(np.asarray(cell.mesh), cutoff_mesh):
            raise ValueError(
                'cell.ke_cutoff and cell.mesh are inconsistent; set the '
                'cutoff before cell.build() and rebuild the Cell')
        if mf._numint.cell is not cell:
            raise ValueError('multigrid XC and RHFDFT must own the same Cell')
        if not np.array_equal(
                np.asarray(mf._numint.mesh), np.asarray(cell.mesh)):
            raise ValueError(
                'multigrid XC mesh is determined by cell.ke_cutoff and must '
                'equal cell.mesh')
        if bool(mf._numint.xc_with_j) != bool(mf.multigrid_with_j):
            raise ValueError(
                'multigrid Hartree ownership is inconsistent between '
                'RHFDFT.multigrid_with_j and the numerical integrator')
    else:
        if not isinstance(mf.grids, gen_grid.UniformGrids):
            raise NotImplementedError(
                'Gamma-point target XC requires UniformGrids or the '
                'supported MultiGridNumInt2 route; atom-centered grid '
                'response is not implemented')
        if mf.grids.cell is not cell:
            raise ValueError(
                'target XC UniformGrids and RHFDFT must own the same Cell')
        if mf.multigrid_with_j:
            raise ValueError(
                'multigrid_with_j=True requires '
                'multigrid_numint(with_j=True)')
    if isinstance(mf.with_df, FFTDF) and not np.array_equal(
            np.asarray(mf.with_df.mesh), np.asarray(cell.mesh)):
        raise ValueError(
            'FFTDF energy and gradient require with_df.mesh == cell.mesh'
        )
    if mf._numint.libxc.is_hybrid_xc(mf.xc):
        raise NotImplementedError(
            'periodic target functionals must be pure semilocal PBE or R2SCAN')
    if mf._numint.libxc.is_nlc(mf.xc) or mf.nlc not in ('', 0, False):
        raise NotImplementedError('periodic NLC/VV10 is not supported')
    target_xc = _target_xc_name(mf)
    if target_xc not in _SUPPORTED_TARGET_XC:
        raise NotImplementedError(
            'periodic target functional must be PBE or R2SCAN')
    if (target_xc == 'R2SCAN'
            and isinstance(mf._numint, multigrid.MultiGridNumInt2)):
        raise NotImplementedError(
            'CPU PySCF 2.14.0 multigrid does not implement the MGGA tau '
            'energy/gradient channel; use uniform-grid R2SCAN on CPU')


class _DFJK:
    '''Force J/K evaluation through the attached periodic DF provider.'''

    def get_jk(self, cell=None, dm=None, hermi=1, kpt=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, **kwargs):
        if cell is None:
            cell = self.cell
        if dm is None:
            dm = self.make_rdm1()
        if kpt is None:
            kpt = self.kpt
        return self.with_df.get_jk(
            np.asarray(dm), hermi=hermi, kpts=kpt,
            kpts_band=kpts_band, with_j=with_j, with_k=with_k,
            omega=omega, exxdiv=self.exxdiv,
        )


class DFRHF(_DFJK, hf.RHF):
    '''Gamma RHF whose Hamiltonian is owned by one DF provider.'''


FFTDFRHF = DFRHF


class RHFDFT(_DFJK, rks.RKS):
    '''Evaluate a supported target DFA on converged Gamma-point RHF orbitals.'''

    _keys = rks.RKS._keys | {
        'mf_hf', 'e_hf', 'e_dft', 'dm_hf', 'target_fock',
        'target_fock_signature',
        'multigrid_ntasks', 'multigrid_ke_ratio',
        'multigrid_rel_cutoff', 'multigrid_with_j',
        'multigrid_backend',
        '_source_state_receipt',
    }
    _hf_options = (
        'max_memory', 'verbose', 'stdout', 'chkfile', 'conv_tol',
        'conv_tol_grad', 'max_cycle', 'init_guess', 'DIIS', 'diis',
        'diis_space', 'diis_start_cycle', 'damp', 'level_shift',
        'direct_scf', 'direct_scf_tol', 'conv_check', 'callback', 'exxdiv',
    )

    def __init__(self, cell, xc='PBE', mf_hf=None):
        rks.RKS.__init__(self, cell, xc=xc)
        self.conv_tol = 1e-10
        self.conv_tol_grad = 1e-8
        self.max_cycle = 80
        provider = FFTDF(cell)
        self.with_df = provider
        if mf_hf is None:
            mf_hf = DFRHF(cell, exxdiv=self.exxdiv)
            mf_hf.with_df = provider
        elif not isinstance(mf_hf, hf.RHF):
            raise TypeError('mf_hf must be a pyscf.pbc.scf.hf.RHF object')
        if mf_hf.cell is not cell:
            raise ValueError('mf_hf and RHFDFT must reference the same Cell')
        if not isinstance(mf_hf.with_df, (FFTDF, GDF)):
            raise TypeError('mf_hf.with_df must be a dcforge CPU PBC provider')

        self.mf_hf = mf_hf
        self.with_df = mf_hf.with_df
        self.e_hf = None
        self.e_dft = None
        self.dm_hf = None
        self.target_fock = None
        self.target_fock_signature = None
        self._source_state_receipt = None
        # These settings are copied to MultiGridNumInt2 by
        # multigrid_numint().  They remain visible on the source object so a
        # caller can select the four-level PySCF reference before copying.
        self.multigrid_ntasks = _MULTIGRID_NTASKS_DEFAULT
        self.multigrid_ke_ratio = _MULTIGRID_KE_RATIO_DEFAULT
        self.multigrid_rel_cutoff = _MULTIGRID_REL_CUTOFF_DEFAULT
        self.multigrid_with_j = False
        self.multigrid_backend = None

    def kernel_hf(self, dm0=None, **kwargs):
        '''Run source RHF with the active FFTDF or GDF provider.'''
        _check_input(self)
        self._copy_hf_options(self, self.mf_hf)
        e_hf = self.mf_hf.kernel(dm0=dm0, **kwargs)
        if not self.mf_hf.converged:
            raise RuntimeError('periodic source RHF calculation did not converge')
        self._sync_hf_results()
        record_pbc_source_state(self)
        return e_hf

    def run_hf(self, dm0=None, **kwargs):
        self.kernel_hf(dm0=dm0, **kwargs)
        return self

    def make_hf_rdm1(self):
        assert_pbc_source_state(self)
        return self.mf_hf.make_rdm1()

    def kernel(self, dm0=None, run_hf=True, **kwargs):
        '''Run RHF and evaluate ``E_DFA[P_HF]`` without target SCF.'''
        _check_input(self)
        if run_hf:
            self.kernel_hf(dm0=dm0, **kwargs)
        else:
            if dm0 is not None or kwargs:
                raise ValueError('dm0/kwargs require run_hf=True')
            assert_pbc_source_state(self)
            self._sync_hf_results()

        # kernel_hf() recorded, or the run_hf=False branch just validated,
        # this exact state.  Avoid hashing the same canonical arrays twice.
        self.dm_hf = self.mf_hf.make_rdm1()
        h1e = self.get_hcore(self.cell)
        vhf = self.get_veff(self.cell, self.dm_hf)
        self.e_coul = float(np.asarray(vhf.ecoul).real)
        self.e_xc = float(np.asarray(vhf.exc).real)
        self.target_fock = np.asarray(h1e) + np.asarray(vhf)
        self.target_fock_signature = _target_fock_signature(
            self, self.dm_hf)
        self.e_dft = float(self.energy_tot(
            dm=self.dm_hf, h1e=h1e, vhf=vhf))
        self.e_tot = self.e_dft
        return self.e_dft

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        '''Evaluate a supplied-density periodic target-DFA energy.'''
        _check_input(self)
        if dm is None:
            dm = self.make_hf_rdm1()
            self._sync_hf_results()
        return rks.RKS.energy_tot(self, dm, h1e, vhf)

    @classmethod
    def from_hf_state(cls, path, *, xc='PBE'):
        '''Construct this CPU Gamma workflow from a validated checkpoint.'''
        state = load_pbc_hf_state(path)
        if state.metadata['sampling']['kind'] != 'gamma':
            raise PBCStateError('RHFDFT requires a Gamma-point checkpoint')
        mf = configure_pbc_provider(cls(state.cell, xc=xc), state)
        mf.restore_hf_state(state)
        return mf

    def save_hf_state(self, path, *, extra_provenance=None):
        '''Save the current gradient-ready source RHF state.'''
        return save_pbc_hf_state(
            path, self, extra_provenance=extra_provenance)

    def restore_hf_state(self, state_or_path):
        '''Restore and physically validate one source RHF checkpoint.'''
        restore_pbc_source_state(self, state_or_path)
        return self

    @property
    def source_validation(self):
        '''Return the current immutable source-state receipt or ``None``.'''
        return self._source_state_receipt

    def get_hcore(self, cell=None, kpt=None):
        '''Return the one-electron Hamiltonian owned by the source provider.

        PySCF normally lets ``MultiGridNumInt2`` split the long-range local
        pseudopotential between ``get_hcore`` and its combined J/XC build.
        That split is correct when multigrid owns Hartree, but it would omit
        the long-range local potential when ``xc_with_j=False``.  The hybrid
        DC-Forge route instead keeps the complete one-electron Hamiltonian in
        the same FFTDF/GDF representation as source RHF and Hartree.
        '''
        return self.mf_hf.get_hcore(cell=cell, kpt=kpt)

    def fft_density_fit(self, mesh=None):
        '''Replace source and target by one explicit FFTDF provider.'''
        return self._set_provider(FFTDF(self.cell, mesh=mesh))

    def density_fit(self, auxbasis=None, with_df=None):
        '''Replace source and target by one shared Gamma-point GDF provider.'''
        if with_df is None:
            with_df = GDF(self.cell, auxbasis=auxbasis)
        elif not isinstance(with_df, GDF):
            raise TypeError('with_df must be dcforge CPU PBC GDF')
        return self._set_provider(with_df)

    def multigrid_numint(self, mesh=None, ntasks=None, ke_ratio=None,
                         rel_cutoff=None, with_j=False):
        r'''Apply PySCF multigrid to target XC and optionally target Hartree.

        This method preserves the source/target FFTDF or GDF provider.  In
        Source RHF J/K and the source CPHF action always continue to use
        ``with_df``.  With ``with_j=False`` (the default), target Hartree also
        uses that provider and multigrid owns only XC.  With ``with_j=True``,
        fixed-density target :math:`J+V_{\rm xc}` and its explicit nuclear
        derivative are evaluated together by multigrid.

        The returned object follows ``pyscf.pbc.dft.rks.RKS`` convention and
        is a shallow copy.  It shares the source RHF object and its active DF
        provider with ``self`` but owns a new ``MultiGridNumInt2`` object.

        Kwargs:
            mesh : None
                Retained for PySCF signature compatibility.  DC-Forge does
                not accept a separate target-XC mesh on the CPU because
                PySCF 2.14.0 forms the master FFT and finest multigrid task
                from different mesh owners.  Set ``cell.ke_cutoff`` before
                ``cell.build()`` instead.
            ntasks : int or None
                Number of reciprocal multigrid levels.  ``None`` uses the
                DC-Forge CPU default of three levels.  Set ``ntasks=4`` for
                the PySCF default/reference partition.
            ke_ratio : float or None
                Ratio between adjacent level kinetic-energy cutoffs.  The
                default is ``3.0``.
            rel_cutoff : float or None
                Relative real-space cutoff passed to PySCF's multigrid task
                builder.  The default is ``20.0``.
            with_j : bool
                If ``True``, include target Hartree J in the multigrid
                energy, Fock, and explicit derivative.

        Returns:
            dcforge.cpu.pbc.energy.rhf_dft.RHFDFT
                A copied workflow whose target XC numerical integrator is
                multigrid with the requested target-Hartree ownership.
        '''
        if mesh is not None:
            raise ValueError(
                'set cell.ke_cutoff before cell.build(); a separate '
                'multigrid mesh is not supported')
        if ntasks is None:
            ntasks = getattr(self, 'multigrid_ntasks',
                             _MULTIGRID_NTASKS_DEFAULT)
        if ke_ratio is None:
            ke_ratio = getattr(self, 'multigrid_ke_ratio',
                               _MULTIGRID_KE_RATIO_DEFAULT)
        if rel_cutoff is None:
            rel_cutoff = getattr(self, 'multigrid_rel_cutoff',
                                 _MULTIGRID_REL_CUTOFF_DEFAULT)
        ntasks = int(ntasks)
        ke_ratio = float(ke_ratio)
        rel_cutoff = float(rel_cutoff)
        if not isinstance(with_j, (bool, np.bool_)):
            raise TypeError('multigrid with_j must be boolean')
        with_j = bool(with_j)
        if ntasks < 2:
            raise ValueError('multigrid ntasks must be at least 2')
        if ke_ratio <= 1:
            raise ValueError('multigrid ke_ratio must be greater than 1')
        if rel_cutoff <= 0:
            raise ValueError('multigrid rel_cutoff must be positive')
        if self.cell.ke_cutoff is None or self.cell.ke_cutoff <= 0:
            raise ValueError(
                'multigrid_numint requires a positive cell.ke_cutoff set '
                'before cell.build()')
        cutoff_mesh = self.cell.symmetrize_mesh(
            self.cell.cutoff_to_mesh(self.cell.ke_cutoff))
        if not np.array_equal(np.asarray(self.cell.mesh), cutoff_mesh):
            raise ValueError(
                'cell.ke_cutoff and cell.mesh are inconsistent; set the '
                'cutoff before cell.build() and rebuild the Cell')

        mf = self.copy()
        ni = multigrid.MultiGridNumInt2(self.cell)
        ni.mesh = np.asarray(self.cell.mesh, dtype=int).copy()
        ni.ntasks = ntasks
        ni.ke_ratio = ke_ratio
        ni.rel_cutoff = rel_cutoff
        ni.xc_with_j = with_j
        mf._numint = ni
        mf.multigrid_ntasks = ntasks
        mf.multigrid_ke_ratio = ke_ratio
        mf.multigrid_rel_cutoff = rel_cutoff
        mf.multigrid_with_j = with_j
        mf.multigrid_backend = (
            'pyscf_multigrid_j_xc' if with_j
            else 'pyscf_multigrid_xc_only')
        mf.e_dft = mf.dm_hf = mf.target_fock = None
        mf.target_fock_signature = None
        if mf.e_hf is not None:
            mf.e_tot = mf.e_hf
        return mf

    def reset(self, cell=None):
        rks.RKS.reset(self, cell)
        self.mf_hf.reset(cell)
        self.with_df = self.mf_hf.with_df
        self.e_hf = self.e_dft = self.dm_hf = self.target_fock = None
        self.target_fock_signature = None
        clear_pbc_source_state(self)
        return self

    def nuc_grad_method(self):
        if getattr(self.mf_hf, 'dcforge_experimental_source', None):
            raise NotImplementedError(
                'the active experimental source RHF has no analytic gradient')
        from dcforge.cpu.pbc.grad.rhf_dft import Gradients
        return Gradients(self)

    Gradients = nuc_grad_method

    def as_scanner(self):
        raise NotImplementedError('periodic HF-DFT scanner is not implemented')

    def _sync_hf_results(self):
        mf_hf = self.mf_hf
        self.e_hf = float(mf_hf.e_tot)
        self.e_dft = self.dm_hf = self.target_fock = None
        self.target_fock_signature = None
        self.e_tot = self.e_hf
        self.mo_energy = mf_hf.mo_energy
        self.mo_coeff = mf_hf.mo_coeff
        self.mo_occ = mf_hf.mo_occ
        self.converged = mf_hf.converged
        self.cycles = mf_hf.cycles

    def _copy_hf_options(self, source, target):
        for key in self._hf_options:
            if hasattr(source, key) and hasattr(target, key):
                setattr(target, key, getattr(source, key))

    def _set_provider(self, provider):
        if provider.cell is not self.cell:
            raise ValueError('density-fitting provider must own the active Cell')
        if getattr(self.mf_hf, 'dcforge_experimental_source', None):
            mf_hf = DFRHF(self.cell, exxdiv=self.exxdiv)
            self._copy_hf_options(self, mf_hf)
            self.mf_hf = mf_hf
        self.with_df = self.mf_hf.with_df = provider
        self.e_hf = self.e_dft = self.dm_hf = self.target_fock = None
        self.target_fock_signature = None
        self.converged = False
        clear_pbc_source_state(self)
        return self


def energy_tot(mf_hf, mf_dft):
    '''Evaluate periodic target DFT energy with a converged RHF density.'''
    if not mf_hf.converged:
        raise RuntimeError('periodic source RHF calculation must be converged')
    if mf_hf.cell is not mf_dft.cell:
        raise ValueError('source and target must reference the same Cell')
    return float(mf_dft.energy_tot(dm=mf_hf.make_rdm1()))


def _target_fock_signature(mf, dm):
    '''Return immutable state that defines the reusable target Fock.'''
    provider = mf.with_df
    ni = mf._numint
    values = [
        'dcforge.cpu.pbc.target_fock.v2',
        str(mf.xc), id(provider), type(provider).__module__,
        type(provider).__name__, bool(mf.multigrid_with_j),
        type(ni).__module__, type(ni).__name__,
        type(mf.grids).__module__, type(mf.grids).__name__,
        np.asarray(mf.cell.atom_coords()).tobytes(),
        np.asarray(mf.cell.lattice_vectors()).tobytes(),
        float(mf.cell.ke_cutoff) if mf.cell.ke_cutoff is not None else None,
        float(mf.cell.precision),
        np.asarray(mf.cell.mesh, dtype=int).tobytes(),
        _array_digest(dm),
    ]
    values.append(_signature_value(getattr(mf.grids, 'mesh', None)))
    for name in ('mesh', 'xc_with_j', 'ntasks', 'ke_ratio', 'rel_cutoff'):
        values.append(_signature_value(getattr(ni, name, None)))
    for name in (
            'mesh', 'auxbasis', 'omega', 'mesh_compact', 'omega_j2c',
            'mesh_j2c', 'linear_dep_threshold'):
        values.append(_signature_value(getattr(provider, name, None)))
    return tuple(values)


def _target_fock_is_current(mf, dm):
    '''Return whether the cached target Fock belongs to exactly ``dm``.'''
    return (
        mf.target_fock is not None
        and mf.dm_hf is not None
        and mf.target_fock_signature == _target_fock_signature(mf, dm)
        # The digest exposes density identity to dependent cache signatures;
        # exact equality remains the collision-independent final guard.
        and np.array_equal(np.asarray(mf.dm_hf), np.asarray(dm))
    )


def _array_digest(value):
    '''Return a storage-independent digest for one CPU density matrix.'''
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode('ascii'))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _signature_value(value):
    '''Return a stable scalar or array description for one cache option.'''
    if value is None or isinstance(value, (str, bool, int, float, complex)):
        return value
    if isinstance(value, Mapping):
        return ('mapping', tuple(
            (repr(key), _signature_value(value[key]))
            for key in sorted(value, key=repr)))
    if isinstance(value, (list, tuple)):
        return ('sequence', tuple(_signature_value(item) for item in value))
    value = np.asarray(value)
    if value.dtype.hasobject:
        return repr(value.item() if value.ndim == 0 else value.tolist())
    value = np.ascontiguousarray(value)
    return value.dtype.str, value.shape, value.tobytes()
