# DC-Forge
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

'''Unrestricted non-SCF energy workflow built on PySCF UKS.'''

from pyscf import scf
from pyscf.df import df_jk
from pyscf.dft import uks

from dcforge.tools.logger import log_kernel_time
from dcforge.tools.source import SourceAPI, make_source
from dcforge.cpu.disp.d2c import evaluate_d2c, normalize_dispersion
from dcforge.cpu.energy.rnsc import (
    RNSC,
    _check_df_owner,
    _check_source_state,
    _check_xc,
    _clear_target_state,
    _make_source_signature,
    _record_source_signature,
)


class UNSC(SourceAPI, uks.UKS):
    '''Molecular non-self-consistent target energy on a stationary source.

    Args:
        mol : pyscf.gto.Mole
            Shared molecule for source SCF and target evaluation.
        xc : str
            Target functional, including semilocal, hybrid or RSH terms.
        mf_source : native HF/KS object or None
            Optional conventional source with matching spin/backend. Its grids
            and XC are retained. Do not also pass source_xc.

    Kwargs:
        source_xc : str or None
            Source functional. None or 'HF' constructs HF; otherwise KS.
        mf_hf : native HF/KS object or None
            Compatibility spelling of mf_source; do not supply both.

    Notes:
        Public source-neutral name: UNSC. The historical class names are
        retained as compatibility aliases. Target SCF is never performed. Configure
        source grids on mf_source.grids, independently of target self.grids.
    '''

    _keys = uks.UKS._keys | {
        'mf_source', 'e_source', 'e_dft', 'e_d2c', 'dm_source',
        'dispersion', 'd2c_result', '_source_signature',
    }
    _source_options = RNSC._source_options

    def __init__(self, mol, xc='LDA,VWN', mf_source=None, *,
                 source_xc=None, mf_hf=None):
        uks.UKS.__init__(self, mol, xc=xc)
        mf_source, supplied_source = make_source(
            mol, scf.uhf.UHF, uks.UKS, mf_hf, mf_source, source_xc)

        self.mf_source = mf_source
        if supplied_source:
            self._copy_source_options(mf_source, self)
        self.e_source = None
        self.e_dft = None
        self.e_d2c = None
        self.dm_source = None
        self.dispersion = None
        self.d2c_result = None
        self._source_signature = _make_source_signature(self)

    def set_dispersion(self, correction=None):
        '''Select D2C dispersion parameters matching the current target ``xc``.'''
        self.dispersion = normalize_dispersion(correction)
        self.e_d2c = self.d2c_result = None
        self.e_tot = self.e_dft if self.dispersion is None else None
        return self

    def kernel_source(self, dm0=None, **kwargs):
        '''Run unrestricted source SCF and synchronize its canonical orbital state.

        Kwargs:
            dm0 : ndarray or tuple of ndarray or None
                Initial alpha/beta density passed to source SCF.
            **kwargs
                Additional keyword arguments passed to the source ``kernel``.

        Returns:
            float
                Converged unrestricted source SCF total energy in Hartree.
        '''
        _check_xc(self)
        _clear_target_state(self)
        self.mf_source.converged = False
        self._copy_source_options(self, self.mf_source)
        self._prepare_source_grid()
        e_source = self.mf_source.kernel(dm0=dm0, **kwargs)
        if not self.mf_source.converged:
            raise RuntimeError('unrestricted source SCF calculation did not converge')
        self._sync_source_results()
        _record_source_signature(self)
        return e_source

    def run_source(self, dm0=None, **kwargs):
        '''Run unrestricted source SCF and return this workflow object.'''
        self.kernel_source(dm0=dm0, **kwargs)
        return self

    def make_source_rdm1(self):
        '''Return alpha/beta densities from the attached source orbitals.

        Returns:
            ndarray of shape (2, nao, nao)
                Alpha/beta source density matrices with occupation one per
                occupied spin orbital.
        '''
        _check_source_state(self)
        return self.mf_source.make_rdm1()

    @log_kernel_time('Energy workflow')
    def kernel(self, dm0=None, run_source=True, *, run_hf=None, **kwargs):
        '''Run source SCF, then evaluate the target functional at its density.

        Kwargs:
            dm0 : ndarray or tuple of ndarray or None
                Initial unrestricted source-SCF density.  Valid only when ``run_source=True``.
            run_source : bool
                Run unrestricted source SCF when true. False reuses attached
                orbitals; the caller is responsible for source stationarity.
            **kwargs
                Additional keyword arguments passed only to unrestricted source SCF.

        Returns:
            float
                Fixed-density target ``E_target[P_source^alpha,P_source^beta]`` in
                Hartree.  No target UKS SCF optimization is performed.
        '''
        if run_hf is not None:
            if run_source is not True and run_source != run_hf:
                raise ValueError("Conflicting run_source and run_hf arguments")
            run_source = run_hf
        _check_xc(self)
        if run_source:
            self.kernel_source(dm0=dm0, **kwargs)
        else:
            if dm0 is not None or kwargs:
                raise ValueError('dm0/kwargs require run_source=True')
            _check_source_state(self)
            self._sync_source_results()

        self.dm_source = self.make_source_rdm1()
        self.e_tot = float(self.energy_tot(dm=self.dm_source))
        from dcforge.tools.logger import energy_summary
        energy_summary(self, unrestricted=True)
        return self.e_tot

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        '''Evaluate target non-SCF energy and optional D2C dispersion correction.

        Args:
            dm : ndarray or tuple of ndarray or None
                Supplied alpha/beta AO density.
            h1e : ndarray or None
                Optional one-electron Hamiltonian.
            vhf : ndarray or None
                Optional precomputed target effective potential.

        Returns:
            float
                Target UKS-on-source plus optional D2C dispersion energy in Hartree.
        '''
        _check_xc(self)
        if dm is None:
            dm = self.make_source_rdm1()
            self._sync_source_results()
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
        self.mf_source.reset(mol)
        _clear_target_state(self)
        return self

    def nuc_grad_method(self):
        '''Return the DC-Forge unrestricted analytic gradient object.'''
        from dcforge.cpu.grad.unsc import Gradients
        return Gradients(self)

    Gradients = nuc_grad_method
    scf = kernel

    def Hessian(self):
        raise NotImplementedError('Non-SCF Hessians are not implemented')

    def as_scanner(self):
        from dcforge.tools.scanner import energy_scanner
        return energy_scanner(self)

    def density_fit(self, auxbasis=None, with_df=None, only_dfj=False):
        '''Apply one PySCF molecular DF provider to source and target.

        Args:
            auxbasis : str, basis dict, or None
                Auxiliary basis passed to :func:`pyscf.df.df_jk.density_fit`.
                When omitted, PySCF selects the matching HF/JK-fit basis and
                uses its normal even-tempered fallback if no match exists.

        Kwargs:
            with_df : pyscf.df.df.DF or None
                Existing provider shared by unrestricted source SCF and target UKS.
            only_dfj : bool
                Use DF for Coulomb only. Energy is available, but the current
                analytic unrestricted non-SCF gradient requires ``only_dfj=False``.

        Returns:
            UNSC
                Copied density-fitted unrestricted non-SCF workflow.
        '''
        _check_df_owner(self, with_df)
        source_was_converged = bool(self.mf_source.converged)
        if auxbasis is None and with_df is None:
            # A semilocal source can still feed a hybrid target. Resolve one
            # exchange-capable provider, as in the original HF-source path.
            from pyscf.df.addons import make_auxbasis
            auxbasis = make_auxbasis(self.mol, xc='HF')
        mf_source = df_jk.density_fit(
            self.mf_source,
            auxbasis=auxbasis,
            with_df=with_df,
            only_dfj=only_dfj,
        )
        mf = _DFUNSC(self, mf_source.with_df, only_dfj)
        mf.mf_source = mf_source
        _clear_target_state(mf)
        if not source_was_converged:
            _record_source_signature(mf)
        return mf

    def _sync_source_results(self):
        mf_source = self.mf_source
        self.e_source = float(mf_source.e_tot)
        self.e_dft = self.e_d2c = self.dm_source = None
        self.d2c_result = None
        self.e_tot = self.e_source
        self.mo_energy = mf_source.mo_energy
        self.mo_coeff = mf_source.mo_coeff
        self.mo_occ = mf_source.mo_occ
        self.converged = mf_source.converged
        self.cycles = mf_source.cycles

    def _copy_source_options(self, source, target):
        for key in self._source_options:
            if hasattr(source, key) and hasattr(target, key):
                setattr(target, key, getattr(source, key))


class _DFUNSC(df_jk._DFHF, UNSC):
    '''PySCF density-fitting mixin specialized for unrestricted non-SCF.'''

    # The native DF mixin otherwise exposes a target-SCF Hessian.
    Hessian = UNSC.Hessian

    def undo_df(self):
        '''Remove density fitting from both source and target objects.'''
        source_was_converged = bool(self.mf_source.converged)
        mf = df_jk._DFHF.undo_df(self)
        mf.mf_source = self.mf_source.undo_df()
        _clear_target_state(mf)
        if not source_was_converged:
            _record_source_signature(mf)
        return mf

    def nuc_grad_method(self):
        from dcforge.cpu.df.grad.unsc import Gradients
        return Gradients(self)

    Gradients = nuc_grad_method


# Compatibility name; internal code uses the source-neutral class.
UHFDFT = UNSC
