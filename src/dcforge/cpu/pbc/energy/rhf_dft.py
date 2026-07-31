'''Gamma-point periodic RHF-DFT energy with an explicit DF provider.'''

import numpy as np
from pyscf.pbc.dft import multigrid, rks
from pyscf.pbc.scf import hf

from dcforge.cpu.pbc.df.fft import FFTDF
from dcforge.cpu.pbc.df.gdf import GDF
from dcforge.cpu.pbc.df.rsdf import RSDF

# PySCF's pair-multigrid implementation uses four levels by default.  For
# the compact GTH basis/cell combinations supported by this route, the
# coarsest level contributes no resolved difference at the tested precision,
# while still paying one FFT and one C grid-integral traversal.  Keep the
# PySCF partition available through ``ntasks=4`` as the reference oracle.
_MULTIGRID_NTASKS_DEFAULT = 3
_MULTIGRID_KE_RATIO_DEFAULT = 3.0
_MULTIGRID_REL_CUTOFF_DEFAULT = 20.0


def _check_input(mf):
    '''Validate the restricted Gamma-point PBE@RHF capability.'''
    cell = mf.cell
    if cell.dimension != 3:
        raise NotImplementedError('periodic RHF-DFT currently requires a 3D cell')
    if cell.spin != 0 or cell.nelectron % 2:
        raise NotImplementedError('periodic RHF-DFT requires a closed-shell cell')
    if not cell._pseudo or cell._ecp:
        raise NotImplementedError('periodic RHF-DFT currently requires GTH pseudo')
    if not isinstance(mf.with_df, (FFTDF, GDF, RSDF)):
        raise NotImplementedError('unsupported periodic density-fitting provider')
    if isinstance(mf._numint, multigrid.MultiGridNumInt2):
        if not isinstance(mf.with_df, (GDF, RSDF)):
            raise NotImplementedError(
                'multigrid XC requires a GDF or RSDF Hartree provider')
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
        if mf._numint.xc_with_j:
            raise ValueError(
                'multigrid XC must set xc_with_j=False; '
                'Hartree is owned by GDF/RSDF')
    if isinstance(mf.with_df, FFTDF) and not np.array_equal(
            np.asarray(mf.with_df.mesh), np.asarray(cell.mesh)):
        raise ValueError(
            'FFTDF energy and gradient require with_df.mesh == cell.mesh'
        )
    if mf._numint.libxc.is_hybrid_xc(mf.xc):
        raise NotImplementedError('the first periodic target supports pure PBE only')
    if mf._numint.libxc.is_nlc(mf.xc) or mf.nlc not in ('', 0, False):
        raise NotImplementedError('periodic NLC/VV10 is not supported')
    if str(mf.xc).replace(' ', '').upper() != 'PBE':
        raise NotImplementedError('the first periodic target functional is PBE')


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
    '''Evaluate periodic PBE energy on converged Gamma-point RHF orbitals.'''

    _keys = rks.RKS._keys | {
        'mf_hf', 'e_hf', 'e_dft', 'dm_hf', 'target_fock',
        'target_fock_signature',
        'multigrid_ntasks', 'multigrid_ke_ratio',
        'multigrid_rel_cutoff',
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
        if not isinstance(mf_hf.with_df, (FFTDF, GDF, RSDF)):
            raise TypeError('mf_hf.with_df must be a dcforge CPU PBC provider')

        self.mf_hf = mf_hf
        self.with_df = mf_hf.with_df
        self.e_hf = None
        self.e_dft = None
        self.dm_hf = None
        self.target_fock = None
        self.target_fock_signature = None
        # These settings are copied to MultiGridNumInt2 by
        # multigrid_numint().  They remain visible on the source object so a
        # caller can select the four-level PySCF reference before copying.
        self.multigrid_ntasks = _MULTIGRID_NTASKS_DEFAULT
        self.multigrid_ke_ratio = _MULTIGRID_KE_RATIO_DEFAULT
        self.multigrid_rel_cutoff = _MULTIGRID_REL_CUTOFF_DEFAULT

    def kernel_hf(self, dm0=None, **kwargs):
        '''Run source RHF with the active FFTDF/GDF/RSDF provider.'''
        _check_input(self)
        self._copy_hf_options(self, self.mf_hf)
        e_hf = self.mf_hf.kernel(dm0=dm0, **kwargs)
        if not self.mf_hf.converged:
            raise RuntimeError('periodic source RHF calculation did not converge')
        self._sync_hf_results()
        return e_hf

    def run_hf(self, dm0=None, **kwargs):
        self.kernel_hf(dm0=dm0, **kwargs)
        return self

    def make_hf_rdm1(self):
        if not self.mf_hf.converged:
            raise RuntimeError('periodic source RHF must be converged')
        return self.mf_hf.make_rdm1()

    def kernel(self, dm0=None, run_hf=True, **kwargs):
        '''Run RHF and then evaluate ``E_PBE[P_HF]`` without target SCF.'''
        _check_input(self)
        if run_hf:
            self.kernel_hf(dm0=dm0, **kwargs)
        else:
            if dm0 is not None or kwargs:
                raise ValueError('dm0/kwargs require run_hf=True')
            if not self.mf_hf.converged:
                raise RuntimeError('run_hf=False requires converged source RHF')
            self._sync_hf_results()

        self.dm_hf = self.make_hf_rdm1()
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
        '''Evaluate supplied-density periodic PBE energy in Hartree.'''
        _check_input(self)
        return rks.RKS.energy_tot(self, dm, h1e, vhf)

    def get_hcore(self, cell=None, kpt=None):
        '''Return the one-electron Hamiltonian owned by the source provider.

        PySCF normally lets ``MultiGridNumInt2`` split the long-range local
        pseudopotential between ``get_hcore`` and its combined J/XC build.
        That split is correct when multigrid owns Hartree, but it would omit
        the long-range local potential when ``xc_with_j=False``.  The hybrid
        DC-Forge route instead keeps the complete one-electron Hamiltonian in
        the same GDF/RSDF representation as source RHF and Hartree.
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

    def rs_density_fit(self, auxbasis=None, with_df=None):
        '''Replace source and target by one shared Gamma-point RSDF provider.'''
        if with_df is None:
            with_df = RSDF(self.cell, auxbasis=auxbasis)
        elif not isinstance(with_df, RSDF):
            raise TypeError('with_df must be dcforge CPU PBC RSDF')
        return self._set_provider(with_df)

    def multigrid_numint(self, mesh=None, ntasks=None, ke_ratio=None,
                         rel_cutoff=None):
        r'''Apply PySCF multigrid numerical integration to target XC only.

        This method preserves the source/target GDF or RSDF provider.  In
        particular, source RHF J/K, the source CPHF action, target Hartree J,
        and their nuclear derivatives continue to use ``with_df``.  Only
        :math:`E_{\rm xc}`, :math:`V_{\rm xc}`, and the explicit
        :math:`V_{\rm xc}^{I\alpha}` derivative use the multigrid mesh.

        The returned object follows ``pyscf.pbc.dft.rks.RKS`` convention and
        is a shallow copy.  It shares the source RHF object and its Gaussian
        DF provider with ``self`` but owns a new ``MultiGridNumInt2`` object.

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

        Returns:
            dcforge.cpu.pbc.energy.rhf_dft.RHFDFT
                A copied workflow whose target XC numerical integrator is
                multigrid and whose Hartree contribution is excluded from
                that integrator.
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
        ni.xc_with_j = False
        mf._numint = ni
        mf.multigrid_ntasks = ntasks
        mf.multigrid_ke_ratio = ke_ratio
        mf.multigrid_rel_cutoff = rel_cutoff
        mf.e_dft = mf.dm_hf = mf.target_fock = None
        mf.target_fock_signature = None
        if mf.e_hf is not None:
            mf.e_tot = mf.e_hf
        return mf

    def reset(self, cell=None):
        rks.RKS.reset(self, cell)
        self.mf_hf.reset(cell)
        self.e_hf = self.e_dft = self.dm_hf = self.target_fock = None
        self.target_fock_signature = None
        return self

    def nuc_grad_method(self):
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
        self.with_df = self.mf_hf.with_df = provider
        self.e_hf = self.e_dft = self.dm_hf = self.target_fock = None
        self.target_fock_signature = None
        self.converged = False
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
        str(mf.xc), id(provider),
        type(ni).__module__, type(ni).__name__,
        np.asarray(mf.cell.atom_coords()).tobytes(),
        np.asarray(mf.cell.lattice_vectors()).tobytes(),
        float(mf.cell.ke_cutoff) if mf.cell.ke_cutoff is not None else None,
        float(mf.cell.precision),
        np.asarray(mf.cell.mesh, dtype=int).tobytes(),
        np.asarray(dm).shape,
    ]
    for name in ('mesh', 'xc_with_j', 'ntasks', 'ke_ratio', 'rel_cutoff'):
        value = getattr(ni, name, None)
        if value is None:
            values.append(None)
        else:
            values.append(np.asarray(value).tobytes())
    for name in ('mesh', 'omega', 'mesh_compact', 'omega_j2c', 'mesh_j2c'):
        value = getattr(provider, name, None)
        if value is None:
            values.append(None)
        else:
            values.append(np.asarray(value).tobytes())
    return tuple(values)
