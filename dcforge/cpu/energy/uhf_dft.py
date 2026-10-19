'''Unrestricted HF-DFT energy workflow built on PySCF UKS.'''

from pyscf import scf
from pyscf.df import df_jk
from pyscf.dft import uks

from dcforge.cpu.disp.d2c import evaluate_d2c, normalize_dispersion
from dcforge.cpu.energy.rhf_dft import (
    RHFDFT,
    _check_df_owner,
    _check_source_state,
    _check_xc,
    _clear_target_state,
    _make_source_signature,
    _record_source_signature,
)


class UHFDFT(uks.UKS):
    '''Evaluate a target UKS functional on converged UHF orbitals.

    Args:
        mol : pyscf.gto.Mole
            Molecule shared by the source UHF and target UKS evaluator.

    Kwargs:
        xc : str
            NLC-free target functional understood by PySCF UKS.
        mf_hf : pyscf.scf.uhf.UHF or None
            Optional conventional source UHF object.  A new UHF object is
            created when omitted.

    Notes:
        This class inherits UKS integration, Fock, and supplied-density energy
        methods.  Its ``kernel`` runs source UHF, not target UKS SCF.
    '''

    _keys = uks.UKS._keys | {
        'mf_hf', 'e_uhf', 'e_dft', 'e_d2c', 'dm_hf',
        'dispersion', 'd2c_result', '_source_signature',
        '_source_state_signature',
    }
    _hf_options = RHFDFT._hf_options

    def __init__(self, mol, xc='LDA,VWN', mf_hf=None):
        uks.UKS.__init__(self, mol, xc=xc)
        supplied_mf_hf = mf_hf is not None
        if mf_hf is None:
            mf_hf = scf.UHF(mol)
        if type(mf_hf) is not scf.uhf.UHF:
            raise TypeError('mf_hf must be a conventional pyscf.scf.uhf.UHF object')
        if mf_hf.mol is not mol:
            raise ValueError('mf_hf and UHFDFT must reference the same Mole object')

        self.mf_hf = mf_hf
        if supplied_mf_hf:
            self._copy_hf_options(mf_hf, self)
        self.e_uhf = None
        self.e_dft = None
        self.e_d2c = None
        self.dm_hf = None
        self.dispersion = None
        self.d2c_result = None
        self._source_signature = _make_source_signature(self)
        self._source_state_signature = None

    def set_dispersion(self, correction=None):
        '''Select D2C-D4 parameters matching the current target ``xc``.'''
        self.dispersion = normalize_dispersion(correction)
        self.e_d2c = self.d2c_result = None
        self.e_tot = self.e_dft if self.dispersion is None else None
        return self

    def kernel_uhf(self, dm0=None, **kwargs):
        '''Run source UHF and synchronize its canonical orbital state.

        Kwargs:
            dm0 : ndarray or tuple of ndarray or None
                Initial alpha/beta density passed to PySCF UHF.
            **kwargs
                Additional keyword arguments passed to ``UHF.kernel``.

        Returns:
            float
                Converged source UHF total energy in Hartree.
        '''
        self._copy_hf_options(self, self.mf_hf)
        e_uhf = self.mf_hf.kernel(dm0=dm0, **kwargs)
        if not self.mf_hf.converged:
            raise RuntimeError('source UHF calculation did not converge')
        self._sync_hf_results()
        _record_source_signature(self)
        return e_uhf

    def run_uhf(self, dm0=None, **kwargs):
        '''Run source UHF and return this workflow object.'''
        self.kernel_uhf(dm0=dm0, **kwargs)
        return self

    def make_uhf_rdm1(self):
        '''Return converged alpha/beta UHF AO densities.

        Returns:
            ndarray of shape (2, nao, nao)
                Alpha/beta source density matrices with occupation one per
                occupied spin orbital.
        '''
        _check_source_state(self)
        return self.mf_hf.make_rdm1()

    def kernel(self, dm0=None, run_hf=True, **kwargs):
        '''Run UHF, then evaluate the target functional at the UHF density.

        Kwargs:
            dm0 : ndarray or tuple of ndarray or None
                Initial source-UHF density.  Valid only when ``run_hf=True``.
            run_hf : bool
                Run source UHF when true.  False requires the attached
                ``mf_hf`` to be already converged.
            **kwargs
                Additional keyword arguments passed only to source UHF.

        Returns:
            float
                Fixed-density target ``E_DFA[P_UHF^alpha,P_UHF^beta]`` in
                Hartree.  No target UKS SCF optimization is performed.
        '''
        _check_xc(self)
        if run_hf:
            self.kernel_uhf(dm0=dm0, **kwargs)
        else:
            if dm0 is not None or kwargs:
                raise ValueError('dm0/kwargs require run_hf=True')
            _check_source_state(self)
            self._sync_hf_results()

        self.dm_hf = self.make_uhf_rdm1()
        self.e_tot = float(self.energy_tot(dm=self.dm_hf))
        from dcforge.tools.logger import energy_summary
        energy_summary(self, unrestricted=True)
        return self.e_tot

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        '''Evaluate target UHF-DFT energy and optional D2C-D4 correction.

        Args:
            dm : ndarray or tuple of ndarray or None
                Supplied alpha/beta AO density.
            h1e : ndarray or None
                Optional one-electron Hamiltonian.
            vhf : ndarray or None
                Optional precomputed target effective potential.

        Returns:
            float
                Target UKS-on-UHF plus optional D2C-D4 energy in Hartree.
        '''
        _check_xc(self)
        if dm is None:
            dm = self.make_uhf_rdm1()
            self._sync_hf_results()
        self.e_dft = float(uks.UKS.energy_tot(self, dm, h1e, vhf))
        self.dispersion = normalize_dispersion(self.dispersion)
        self.e_d2c = 0.0
        self.d2c_result = None
        self.scf_summary.pop('dispersion', None)
        if self.dispersion == 'd2c':
            self.d2c_result = evaluate_d2c(self.mol, self.xc)
            self.e_d2c = self.d2c_result.energy
            self.scf_summary['dispersion'] = self.e_d2c
        self.e_tot = self.e_dft + self.e_d2c
        return self.e_tot

    def reset(self, mol=None):
        '''Reset source and target objects to the same molecule.'''
        uks.UKS.reset(self, mol)
        self.mf_hf.reset(mol)
        self.e_uhf = self.e_dft = self.e_d2c = self.dm_hf = None
        self.d2c_result = None
        return self

    def nuc_grad_method(self):
        '''Return the DC-Forge unrestricted analytic gradient object.'''
        from dcforge.cpu.grad.uhf_dft import Gradients
        return Gradients(self)

    def as_scanner(self):
        raise NotImplementedError('UHF-DFT scanner support is not implemented')

    def density_fit(self, auxbasis=None, with_df=None, only_dfj=False):
        '''Apply one PySCF molecular DF provider to source and target.

        Args:
            auxbasis : str, basis dict, or None
                Auxiliary basis passed to :func:`pyscf.df.df_jk.density_fit`.
                When omitted, PySCF selects the matching HF/JK-fit basis and
                uses its normal even-tempered fallback if no match exists.

        Kwargs:
            with_df : pyscf.df.df.DF or None
                Existing provider shared by source UHF and target UKS.
            only_dfj : bool
                Use DF for Coulomb only. Energy is available, but the current
                analytic UHF-DFT gradient requires ``only_dfj=False``.

        Returns:
            UHFDFT
                Copied density-fitted UHF-DFT workflow.
        '''
        _check_df_owner(self, with_df)
        source_was_converged = bool(self.mf_hf.converged)
        mf_hf = df_jk.density_fit(
            self.mf_hf,
            auxbasis=auxbasis,
            with_df=with_df,
            only_dfj=only_dfj,
        )
        mf = _DFUHFDFT(self, mf_hf.with_df, only_dfj)
        mf.mf_hf = mf_hf
        _clear_target_state(mf)
        if not source_was_converged:
            _record_source_signature(mf)
        return mf

    def _sync_hf_results(self):
        mf_hf = self.mf_hf
        self.e_uhf = float(mf_hf.e_tot)
        self.e_dft = self.e_d2c = self.dm_hf = None
        self.d2c_result = None
        self.e_tot = self.e_uhf
        self.mo_energy = mf_hf.mo_energy
        self.mo_coeff = mf_hf.mo_coeff
        self.mo_occ = mf_hf.mo_occ
        self.converged = mf_hf.converged
        self.cycles = mf_hf.cycles

    def _copy_hf_options(self, source, target):
        for key in self._hf_options:
            if hasattr(source, key) and hasattr(target, key):
                setattr(target, key, getattr(source, key))


class _DFUHFDFT(df_jk._DFHF, UHFDFT):
    '''PySCF density-fitting mixin specialized for unrestricted HF-DFT.'''

    def undo_df(self):
        '''Remove density fitting from both source and target objects.'''
        source_was_converged = bool(self.mf_hf.converged)
        mf = df_jk._DFHF.undo_df(self)
        mf.mf_hf = self.mf_hf.undo_df()
        _clear_target_state(mf)
        if not source_was_converged:
            _record_source_signature(mf)
        return mf

    def nuc_grad_method(self):
        from dcforge.cpu.df.grad.uhf_dft import Gradients
        return Gradients(self)

    Gradients = nuc_grad_method
