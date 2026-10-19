'''Unrestricted conventional and density-fitted HF-DFT energy on GPU.'''

from gpu4pyscf.df import df_jk
from gpu4pyscf.dft import uks
from gpu4pyscf.scf import uhf

from dcforge.cpu.disp.d2c import evaluate_d2c, normalize_dispersion
from dcforge.cpu.energy.rhf_dft import (
    _check_df_owner,
    _check_source_state,
    _check_xc,
    _clear_target_state,
    _make_source_signature,
    _record_source_signature,
)
from dcforge.gpu._df import default_auxbasis
from dcforge.gpu.energy.rhf_dft import RHFDFT as GPU_RHFDFT


def _check_gpu_contract(mf):
    '''Validate the molecular GPU UHF-DFT energy capability.

    Args:
        mf : dcforge.gpu.energy.uhf_dft.UHFDFT
            Active GPU fixed-density workflow.

    Notes:
        This capability accepts NLC-free functionals implemented by the
        active GPU4PySCF UKS providers, with either conventional objects or
        one shared exchange-capable DF provider.
    '''
    _check_xc(mf)
    target_df = hasattr(mf, 'with_df')
    source_df = hasattr(mf.mf_hf, 'with_df')
    if target_df != source_df:
        raise ValueError('source UHF and target UKS must use the same representation')
    if target_df:
        if mf.with_df is not mf.mf_hf.with_df:
            raise ValueError('source UHF and target UKS must share one GPU with_df')
        if mf.only_dfj or mf.mf_hf.only_dfj:
            raise NotImplementedError('GPU UHF-DFT currently requires only_dfj=False')


class UHFDFT(uks.UKS):
    '''Evaluate an NLC-free target XC on a GPU UHF density.

    Args:
        mol : pyscf.gto.Mole
            Molecule shared by the source GPU UHF and target GPU UKS object.

    Kwargs:
        xc : str
            NLC-free target functional understood by GPU4PySCF UKS.
        mf_hf : gpu4pyscf.scf.uhf.UHF or None
            Optional conventional GPU UHF source object. Density fitting is
            attached afterward through :meth:`density_fit`.

    Notes:
        The inherited UKS numerical integration and supplied-density energy
        kernels are reused. :meth:`kernel` runs source UHF and then calls
        ``UKS.energy_tot(dm=dm_hf)``; target UKS SCF is never executed.
    '''

    _keys = uks.UKS._keys | {
        'mf_hf', 'e_uhf', 'e_dft', 'e_d2c', 'dm_hf',
        'dispersion', 'd2c_result', '_source_signature',
        '_source_state_signature',
    }
    _hf_options = GPU_RHFDFT._hf_options

    def __init__(self, mol, xc='LDA,VWN', mf_hf=None):
        uks.UKS.__init__(self, mol, xc=xc)
        supplied_mf_hf = mf_hf is not None
        if mf_hf is None:
            mf_hf = uhf.UHF(mol)
        if type(mf_hf) is not uhf.UHF:
            raise TypeError(
                'mf_hf must be a conventional gpu4pyscf.scf.uhf.UHF object'
            )
        if mf_hf.mol is not mol:
            raise ValueError('mf_hf and GPU UHFDFT must reference the same Mole object')

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

    @property
    def device(self):
        '''Return the explicit backend owner for this workflow.'''
        return 'gpu'

    def kernel_uhf(self, dm0=None, **kwargs):
        '''Run source GPU UHF and synchronize its canonical orbital state.

        Kwargs:
            dm0 : cupy.ndarray, tuple of cupy.ndarray, or None
                Initial alpha/beta density passed to GPU4PySCF UHF.
            **kwargs
                Additional keyword arguments forwarded to ``UHF.kernel``.

        Returns:
            float
                Converged source UHF energy in Hartree.
        '''
        _check_gpu_contract(self)
        self._copy_hf_options(self, self.mf_hf)
        e_uhf = self.mf_hf.kernel(dm0=dm0, **kwargs)
        if not self.mf_hf.converged:
            raise RuntimeError('source GPU UHF calculation did not converge')
        self._sync_hf_results()
        _record_source_signature(self)
        return float(e_uhf)

    def run_uhf(self, dm0=None, **kwargs):
        '''Run source GPU UHF and return this workflow object.'''
        self.kernel_uhf(dm0=dm0, **kwargs)
        return self

    def make_uhf_rdm1(self):
        '''Return the converged alpha/beta GPU UHF AO densities.'''
        _check_source_state(self)
        return self.mf_hf.make_rdm1()

    def kernel(self, dm0=None, run_hf=True, **kwargs):
        '''Run source UHF, then evaluate fixed-density target XC on the GPU.

        Kwargs:
            dm0 : cupy.ndarray, tuple of cupy.ndarray, or None
                Initial source-UHF density. Valid only with ``run_hf=True``.
            run_hf : bool
                Reuse a source converged with the active DF provider when
                false.
            **kwargs
                Additional keyword arguments passed only to source GPU UHF.

        Returns:
            float
                ``E_DFA[P_UHF^alpha,P_UHF^beta]`` in the active representation.
        '''
        _check_gpu_contract(self)
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
        '''Evaluate target UHF-DFT energy and optional D2C-D4 correction.'''
        _check_gpu_contract(self)
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
        '''Reset source and target GPU objects to the same molecule.'''
        uks.UKS.reset(self, mol)
        self.mf_hf.reset(mol)
        self.e_uhf = self.e_dft = self.e_d2c = self.dm_hf = None
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
                Coulomb-only fitting is not supported by this gradient
                capability.

        Returns:
            UHFDFT
                Copied workflow with one shared source/target GPU DF provider.
        '''
        if only_dfj:
            raise NotImplementedError('GPU UHF-DFT currently requires only_dfj=False')
        _check_df_owner(self, with_df)
        source_was_converged = bool(self.mf_hf.converged)
        auxbasis = default_auxbasis(
            self.mol, auxbasis=auxbasis, with_df=with_df)
        mf_hf = df_jk.density_fit(
            self.mf_hf, auxbasis=auxbasis,
            with_df=with_df,
            only_dfj=False,
        )
        mf = _DFUHFDFT(self, mf_hf.with_df, False)
        mf.mf_hf = mf_hf
        _clear_target_state(mf)
        if not source_was_converged:
            _record_source_signature(mf)
        return mf

    def nuc_grad_method(self):
        '''Return the conventional or density-fitted GPU UHF gradient driver.'''
        _check_gpu_contract(self)
        if hasattr(self, 'with_df'):
            from dcforge.gpu.df.grad.uhf_dft import Gradients
        else:
            from dcforge.gpu.grad.uhf_dft import Gradients

        return Gradients(self)

    Gradients = nuc_grad_method

    def Hessian(self):
        raise NotImplementedError('GPU UHF-DFT Hessians are not implemented')

    def as_scanner(self):
        raise NotImplementedError('GPU UHF-DFT scanner support is not implemented')

    def to_cpu(self):
        raise NotImplementedError(
            'GPU-to-CPU UHFDFT state conversion is not implemented; '
            'use scalar energy output or an explicit CPU workflow'
        )

    def to_gpu(self, device=None):
        '''Return this already GPU-owned workflow.'''
        return self

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
        self.cycles = getattr(mf_hf, 'cycles', None)

    def _copy_hf_options(self, source, target):
        for key in self._hf_options:
            if hasattr(source, key) and hasattr(target, key):
                setattr(target, key, getattr(source, key))

    scf = kernel


class _DFUHFDFT(df_jk._DFHF, UHFDFT):
    '''GPU4PySCF density-fitting mixin for the UHF-DFT workflow.'''

    device = UHFDFT.device
    nuc_grad_method = UHFDFT.nuc_grad_method
    Gradients = UHFDFT.nuc_grad_method
    Hessian = UHFDFT.Hessian
    to_cpu = UHFDFT.to_cpu
    to_gpu = UHFDFT.to_gpu

    def undo_df(self):
        '''Remove density fitting from source and target workflow objects.'''
        source_was_converged = bool(self.mf_hf.converged)
        mf = df_jk._DFHF.undo_df(self)
        mf.mf_hf = self.mf_hf.undo_df()
        _clear_target_state(mf)
        if not source_was_converged:
            _record_source_signature(mf)
        return mf
