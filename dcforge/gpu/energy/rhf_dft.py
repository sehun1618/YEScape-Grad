# DC-Forge
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

'''Restricted conventional and density-fitted HF-DFT energy on GPU.'''

from gpu4pyscf.df import df_jk
from gpu4pyscf.dft import rks
from gpu4pyscf.scf import hf

from dcforge.tools.logger import log_kernel_time
from dcforge.cpu.disp.d2c import evaluate_d2c, normalize_dispersion
from dcforge.cpu.energy.rhf_dft import RHFDFT as CPU_RHFDFT
from dcforge.cpu.energy.rhf_dft import (
    _check_df_owner,
    _check_source_state,
    _check_xc,
    _clear_target_state,
    _make_source_signature,
    _record_source_signature,
)
from dcforge.gpu._df import default_auxbasis


def _check_gpu_contract(mf):
    '''Validate the molecular GPU RHF-DFT energy capability.

    Args:
        mf : dcforge.gpu.energy.rhf_dft.RHFDFT
            Active GPU fixed-density workflow.

    Notes:
        The GPU workflow accepts NLC-free functionals implemented by the
        active GPU4PySCF RKS energy and gradient providers. It accepts either
        a conventional source/target pair or one shared exchange-capable
        molecular DF provider; mixed representations are rejected.
    '''
    _check_xc(mf)
    target_df = hasattr(mf, 'with_df')
    source_df = hasattr(mf.mf_hf, 'with_df')
    if target_df != source_df:
        raise ValueError('source RHF and target RKS must use the same representation')
    if target_df:
        if mf.with_df is not mf.mf_hf.with_df:
            raise ValueError('source RHF and target RKS must share one GPU with_df')
        if mf.only_dfj or mf.mf_hf.only_dfj:
            raise NotImplementedError('GPU RHF-DFT currently requires only_dfj=False')


class RHFDFT(rks.RKS):
    '''Evaluate an NLC-free target XC on a GPU RHF density.

    Args:
        mol : pyscf.gto.Mole
            Molecule shared by the source GPU RHF and target GPU RKS object.

    Kwargs:
        xc : str
            NLC-free target functional understood by GPU4PySCF RKS.
        mf_hf : gpu4pyscf.scf.hf.RHF or None
            Optional conventional GPU RHF source object.  Density fitting is
            applied afterward through :meth:`density_fit`.

    Notes:
        This class inherits GPU4PySCF RKS integration and supplied-density
        energy kernels.  Its :meth:`kernel` runs source RHF and then calls
        ``RKS.energy_tot(dm=dm_hf)``; target RKS SCF is never performed.
    '''

    _keys = rks.RKS._keys | {
        'mf_hf', 'e_hf', 'e_dft', 'e_d2c', 'dm_hf',
        'dispersion', 'd2c_result', '_source_signature',
    }
    _hf_options = CPU_RHFDFT._hf_options

    def __init__(self, mol, xc='LDA,VWN', mf_hf=None):
        rks.RKS.__init__(self, mol, xc=xc)
        supplied_mf_hf = mf_hf is not None
        if mf_hf is None:
            mf_hf = hf.RHF(mol)
        if type(mf_hf) is not hf.RHF:
            raise TypeError(
                'mf_hf must be a conventional gpu4pyscf.scf.hf.RHF object'
            )
        if mf_hf.mol is not mol:
            raise ValueError('mf_hf and GPU RHFDFT must reference the same Mole object')

        self.mf_hf = mf_hf
        if supplied_mf_hf:
            self._copy_hf_options(mf_hf, self)
        self.e_hf = None
        self.e_dft = None
        self.e_d2c = None
        self.dm_hf = None
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

    def kernel_hf(self, dm0=None, **kwargs):
        '''Run source GPU RHF and synchronize its canonical orbital state.

        Kwargs:
            dm0 : cupy.ndarray or None
                Initial AO density passed to GPU4PySCF RHF.
            **kwargs
                Additional keyword arguments forwarded to ``RHF.kernel``.

        Returns:
            float
                Converged source RHF energy in Hartree.
        '''
        _check_gpu_contract(self)
        self._copy_hf_options(self, self.mf_hf)
        e_hf = self.mf_hf.kernel(dm0=dm0, **kwargs)
        if not self.mf_hf.converged:
            raise RuntimeError('source GPU RHF calculation did not converge')
        self._sync_hf_results()
        _record_source_signature(self)
        return float(e_hf)

    def run_hf(self, dm0=None, **kwargs):
        '''Run source GPU RHF and return this workflow object.'''
        self.kernel_hf(dm0=dm0, **kwargs)
        return self

    def make_hf_rdm1(self):
        '''Return the converged GPU RHF AO density.

        Returns:
            cupy.ndarray of shape (nao, nao)
                Closed-shell density including occupation factor two.
        '''
        _check_source_state(self)
        return self.mf_hf.make_rdm1()

    @log_kernel_time('Energy workflow')
    def kernel(self, dm0=None, run_hf=True, **kwargs):
        '''Run source RHF, then evaluate fixed-density target XC on the GPU.

        Kwargs:
            dm0 : cupy.ndarray or None
                Initial source-RHF density.  Valid only with ``run_hf=True``.
            run_hf : bool
                Reuse supplied source orbitals when false; an unconverged
                flag warns but does not block explicit reuse.
            **kwargs
                Additional keyword arguments passed only to source GPU RHF.

        Returns:
            float
                ``E_DFA[P_RHF]`` in Hartree using the active representation.
        '''
        _check_gpu_contract(self)
        if run_hf:
            self.kernel_hf(dm0=dm0, **kwargs)
        else:
            if dm0 is not None or kwargs:
                raise ValueError('dm0/kwargs require run_hf=True')
            _check_source_state(self)
            self._sync_hf_results()

        self.dm_hf = self.make_hf_rdm1()
        self.e_tot = float(self.energy_tot(dm=self.dm_hf))
        from dcforge.tools.logger import energy_summary
        energy_summary(self)
        return self.e_tot

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        '''Evaluate target HF-DFT energy and optional D2C dispersion correction.'''
        _check_gpu_contract(self)
        if dm is None:
            dm = self.make_hf_rdm1()
            self._sync_hf_results()
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
        self.mf_hf.reset(mol)
        self.e_hf = self.e_dft = self.e_d2c = self.dm_hf = None
        self.d2c_result = None
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
            RHFDFT
                Copied workflow with a shared source/target GPU DF provider.
        '''
        if only_dfj:
            raise NotImplementedError('GPU RHF-DFT currently requires only_dfj=False')
        _check_df_owner(self, with_df)
        source_was_converged = bool(self.mf_hf.converged)
        auxbasis = default_auxbasis(
            self.mol, auxbasis=auxbasis, with_df=with_df)
        mf_hf = df_jk.density_fit(
            self.mf_hf, auxbasis=auxbasis,
            with_df=with_df,
            only_dfj=False,
        )
        mf = _DFRHFDFT(self, mf_hf.with_df, False)
        mf.mf_hf = mf_hf
        _clear_target_state(mf)
        if not source_was_converged:
            _record_source_signature(mf)
        return mf

    def nuc_grad_method(self):
        '''Return the conventional or density-fitted GPU gradient driver.'''
        _check_gpu_contract(self)
        if hasattr(self, 'with_df'):
            from dcforge.gpu.df.grad.rhf_dft import Gradients
        else:
            from dcforge.gpu.grad.rhf_dft import Gradients

        return Gradients(self)

    Gradients = nuc_grad_method

    def Hessian(self):
        raise NotImplementedError('GPU RHF-DFT Hessians are not implemented')

    def as_scanner(self):
        from dcforge.tools.scanner import energy_scanner
        return energy_scanner(self)

    def to_cpu(self):
        raise NotImplementedError(
            'GPU-to-CPU RHFDFT state conversion is not implemented; '
            'use scalar energy output or an explicit CPU workflow'
        )

    def to_gpu(self, device=None):
        '''Return this already GPU-owned workflow.'''
        return self

    def _sync_hf_results(self):
        mf_hf = self.mf_hf
        self.e_hf = float(mf_hf.e_tot)
        self.e_dft = self.e_d2c = self.dm_hf = None
        self.d2c_result = None
        self.e_tot = self.e_hf
        self.mo_energy = mf_hf.mo_energy
        self.mo_coeff = mf_hf.mo_coeff
        self.mo_occ = mf_hf.mo_occ
        self.converged = mf_hf.converged
        self.cycles = getattr(mf_hf, 'cycles', None)

    def _copy_hf_options(self, source, target):
        for key in self._hf_options:
            if hasattr(source, key) and hasattr(target, key):
                setattr(target, key, getattr(source, key))

    scf = kernel


class _DFRHFDFT(df_jk._DFHF, RHFDFT):
    '''GPU4PySCF density-fitting mixin for the RHF-DFT workflow.'''

    # GPU4PySCF's generic DF mixin owns ordinary SCF derivative dispatch and
    # infers ``device`` from the class module name.  Rebind these methods so a
    # DC-Forge DF object cannot silently expose target-RKS derivatives or call
    # itself a CPU object merely because its module starts with ``dcforge``.
    device = RHFDFT.device
    nuc_grad_method = RHFDFT.nuc_grad_method
    Gradients = RHFDFT.nuc_grad_method
    Hessian = RHFDFT.Hessian
    to_cpu = RHFDFT.to_cpu
    to_gpu = RHFDFT.to_gpu

    def undo_df(self):
        '''Remove density fitting from source and target workflow objects.'''
        source_was_converged = bool(self.mf_hf.converged)
        mf = df_jk._DFHF.undo_df(self)
        mf.mf_hf = self.mf_hf.undo_df()
        _clear_target_state(mf)
        if not source_was_converged:
            _record_source_signature(mf)
        return mf
