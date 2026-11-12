# YEScape-Grad
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

'''Restricted conventional and density-fitted non-SCF energy on GPU.'''

from gpu4pyscf.df import df_jk
from gpu4pyscf.dft import rks
from gpu4pyscf.scf import hf

from yescape_grad.tools.logger import log_kernel_time
from yescape_grad.tools.source import SourceAPI, make_source
from yescape_grad.cpu.disp.d2c import evaluate_d2c, normalize_dispersion
from yescape_grad.cpu.energy.rnsc import RNSC as CPU_RNSC
from yescape_grad.cpu.energy.rnsc import (
    _check_df_owner,
    _check_source_state,
    _check_xc,
    _clear_target_state,
    _make_source_signature,
    _record_source_signature,
)
from yescape_grad.gpu._df import default_auxbasis


def _check_gpu_contract(mf):
    '''Validate the molecular GPU restricted non-SCF energy capability.

    Args:
        mf : yescape_grad.gpu.energy.rnsc.RNSC
            Active GPU fixed-density workflow.

    Notes:
        The GPU workflow accepts NLC-free functionals implemented by the
        active GPU4PySCF RKS energy and gradient providers. It accepts either
        a conventional source/target pair or one shared exchange-capable
        molecular DF provider; mixed representations are rejected.
    '''
    _check_xc(mf)
    target_df = hasattr(mf, 'with_df')
    source_df = hasattr(mf.mf_source, 'with_df')
    if target_df != source_df:
        raise ValueError('restricted source SCF and target RKS must use the same representation')
    if target_df:
        if mf.with_df is not mf.mf_source.with_df:
            raise ValueError('restricted source SCF and target RKS must share one GPU with_df')
        if mf.only_dfj or mf.mf_source.only_dfj:
            raise NotImplementedError('GPU restricted non-SCF currently requires only_dfj=False')


class RNSC(SourceAPI, rks.RKS):
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
        Public source-neutral name: RNSC. The historical class names are
        retained as compatibility aliases. Target SCF is never performed. Configure
        source grids on mf_source.grids, independently of target self.grids.
    '''

    _keys = rks.RKS._keys | {
        'mf_source', 'e_source', 'e_dft', 'e_d2c', 'dm_source',
        'dispersion', 'd2c_result', '_source_signature',
    }
    _source_options = CPU_RNSC._source_options

    def __init__(self, mol, xc='LDA,VWN', mf_source=None, *,
                 source_xc=None, mf_hf=None):
        rks.RKS.__init__(self, mol, xc=xc)
        mf_source, supplied_source = make_source(
            mol, hf.RHF, rks.RKS, mf_hf, mf_source, source_xc)

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

    @property
    def device(self):
        '''Return the explicit backend owner for this workflow.'''
        return 'gpu'

    def kernel_source(self, dm0=None, **kwargs):
        '''Run restricted GPU source SCF and synchronize its canonical orbital state.

        Kwargs:
            dm0 : cupy.ndarray or None
                Initial AO density passed to the GPU source SCF.
            **kwargs
                Additional keyword arguments forwarded to the source ``kernel``.

        Returns:
            float
                Converged restricted source SCF energy in Hartree.
        '''
        _check_gpu_contract(self)
        _check_xc(self)
        _clear_target_state(self)
        self.mf_source.converged = False
        self._copy_source_options(self, self.mf_source)
        self._prepare_source_grid()
        e_source = self.mf_source.kernel(dm0=dm0, **kwargs)
        if not self.mf_source.converged:
            raise RuntimeError('restricted GPU source SCF calculation did not converge')
        self._sync_source_results()
        _record_source_signature(self)
        return float(e_source)

    def run_source(self, dm0=None, **kwargs):
        '''Run restricted GPU source SCF and return this workflow object.'''
        self.kernel_source(dm0=dm0, **kwargs)
        return self

    def make_source_rdm1(self):
        '''Return the restricted density from the attached source orbitals.

        Returns:
            cupy.ndarray of shape (nao, nao)
                Closed-shell density including occupation factor two.
        '''
        _check_source_state(self)
        return self.mf_source.make_rdm1()

    @log_kernel_time('Energy workflow')
    def kernel(self, dm0=None, run_source=True, *, run_hf=None, **kwargs):
        '''Run restricted source SCF, then evaluate fixed-density target XC on the GPU.

        Kwargs:
            dm0 : cupy.ndarray or None
                Initial restricted source-SCF density.  Valid only with ``run_source=True``.
            run_source : bool
                Reuse attached orbitals when false; the caller is responsible
                for source stationarity.
            **kwargs
                Additional keyword arguments passed only to restricted GPU source SCF.

        Returns:
            float
                ``E_target[P_source]`` in Hartree using the active representation.
        '''
        if run_hf is not None:
            if run_source is not True and run_source != run_hf:
                raise ValueError("Conflicting run_source and run_hf arguments")
            run_source = run_hf
        _check_gpu_contract(self)
        if run_source:
            self.kernel_source(dm0=dm0, **kwargs)
        else:
            if dm0 is not None or kwargs:
                raise ValueError('dm0/kwargs require run_source=True')
            _check_source_state(self)
            self._sync_source_results()

        self.dm_source = self.make_source_rdm1()
        self.e_tot = float(self.energy_tot(dm=self.dm_source))
        from yescape_grad.tools.logger import energy_summary
        energy_summary(self)
        return self.e_tot

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        '''Evaluate target non-SCF energy and optional D2C dispersion correction.'''
        _check_gpu_contract(self)
        if dm is None:
            dm = self.make_source_rdm1()
            self._sync_source_results()
        self.e_dft = float(rks.RKS.energy_tot(self, dm, h1e, vhf))
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
        '''Reset source and target GPU objects to the same molecule.'''
        rks.RKS.reset(self, mol)
        self.mf_source.reset(mol)
        _clear_target_state(self)
        return self

    def density_fit(self, auxbasis=None, with_df=None, only_dfj=False):
        '''Apply one GPU4PySCF DF provider to source and target.

        Args:
            auxbasis : str, basis dict, or None
                Molecular auxiliary basis passed to GPU4PySCF. When omitted,
                PySCF's matching HF/JK-fit basis is selected from the orbital
                basis, with its normal even-tempered fallback.

        Kwargs:
            with_df : gpu4pyscf.df.df.DF or None
                Optional existing GPU DF provider.
            only_dfj : bool
                Coulomb-only fitting is outside the first GPU capability and
                is rejected.

        Returns:
            RNSC
                Copied workflow with a shared source/target GPU DF provider.
        '''
        if only_dfj:
            raise NotImplementedError('GPU restricted non-SCF currently requires only_dfj=False')
        _check_df_owner(self, with_df)
        source_was_converged = bool(self.mf_source.converged)
        auxbasis = default_auxbasis(
            self.mol, auxbasis=auxbasis, with_df=with_df)
        mf_source = df_jk.density_fit(
            self.mf_source, auxbasis=auxbasis,
            with_df=with_df,
            only_dfj=False,
        )
        mf = _DFRNSC(self, mf_source.with_df, False)
        mf.mf_source = mf_source
        _clear_target_state(mf)
        if not source_was_converged:
            _record_source_signature(mf)
        return mf

    def nuc_grad_method(self):
        '''Return the conventional or density-fitted GPU gradient driver.'''
        _check_gpu_contract(self)
        if hasattr(self, 'with_df'):
            from yescape_grad.gpu.df.grad.rnsc import Gradients
        else:
            from yescape_grad.gpu.grad.rnsc import Gradients

        return Gradients(self)

    Gradients = nuc_grad_method

    def Hessian(self):
        raise NotImplementedError('GPU restricted non-SCF Hessians are not implemented')

    def as_scanner(self):
        from yescape_grad.tools.scanner import energy_scanner
        return energy_scanner(self)

    def to_cpu(self):
        raise NotImplementedError(
            'GPU-to-CPU RNSC state conversion is not implemented; '
            'use scalar energy output or an explicit CPU workflow'
        )

    def to_gpu(self, device=None):
        '''Return this already GPU-owned workflow.'''
        return self

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
        self.cycles = getattr(mf_source, 'cycles', None)

    def _copy_source_options(self, source, target):
        for key in self._source_options:
            if hasattr(source, key) and hasattr(target, key):
                setattr(target, key, getattr(source, key))

    scf = kernel


class _DFRNSC(df_jk._DFHF, RNSC):
    '''GPU4PySCF density-fitting mixin for the restricted non-SCF workflow.'''

    # GPU4PySCF's generic DF mixin owns ordinary SCF derivative dispatch and
    # infers ``device`` from the class module name.  Rebind these methods so a
    # YEScape-Grad DF object cannot silently expose target-RKS derivatives or call
    # itself a CPU object merely because its module starts with ``yescape_grad``.
    device = RNSC.device
    nuc_grad_method = RNSC.nuc_grad_method
    Gradients = RNSC.nuc_grad_method
    Hessian = RNSC.Hessian
    to_cpu = RNSC.to_cpu
    to_gpu = RNSC.to_gpu

    def undo_df(self):
        '''Remove density fitting from source and target workflow objects.'''
        source_was_converged = bool(self.mf_source.converged)
        mf = df_jk._DFHF.undo_df(self)
        mf.mf_source = self.mf_source.undo_df()
        _clear_target_state(mf)
        if not source_was_converged:
            _record_source_signature(mf)
        return mf


# Compatibility name; internal code uses the source-neutral class.
RHFDFT = RNSC
