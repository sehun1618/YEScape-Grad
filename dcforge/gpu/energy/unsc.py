# DC-Forge
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

'''Unrestricted conventional and density-fitted non-SCF energy on GPU.'''

from gpu4pyscf.df import df_jk
from gpu4pyscf.dft import uks
from gpu4pyscf.scf import uhf

from dcforge.tools.logger import log_kernel_time
from dcforge.tools.source import SourceAPI, make_source
from dcforge.cpu.disp.d2c import evaluate_d2c, normalize_dispersion
from dcforge.cpu.energy.rnsc import (
    _check_df_owner,
    _check_source_state,
    _check_xc,
    _clear_target_state,
    _make_source_signature,
    _record_source_signature,
)
from dcforge.gpu._df import default_auxbasis
from dcforge.gpu.energy.rnsc import RNSC as GPU_RNSC


def _check_gpu_contract(mf):
    '''Validate the molecular GPU unrestricted non-SCF energy capability.

    Args:
        mf : dcforge.gpu.energy.unsc.UNSC
            Active GPU fixed-density workflow.

    Notes:
        This capability accepts NLC-free functionals implemented by the
        active GPU4PySCF UKS providers, with either conventional objects or
        one shared exchange-capable DF provider.
    '''
    _check_xc(mf)
    target_df = hasattr(mf, 'with_df')
    source_df = hasattr(mf.mf_source, 'with_df')
    if target_df != source_df:
        raise ValueError('unrestricted source SCF and target UKS must use the same representation')
    if target_df:
        if mf.with_df is not mf.mf_source.with_df:
            raise ValueError('unrestricted source SCF and target UKS must share one GPU with_df')
        if mf.only_dfj or mf.mf_source.only_dfj:
            raise NotImplementedError('GPU unrestricted non-SCF currently requires only_dfj=False')


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
    _source_options = GPU_RNSC._source_options

    def __init__(self, mol, xc='LDA,VWN', mf_source=None, *,
                 source_xc=None, mf_hf=None):
        uks.UKS.__init__(self, mol, xc=xc)
        mf_source, supplied_source = make_source(
            mol, uhf.UHF, uks.UKS, mf_hf, mf_source, source_xc)

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
        '''Run unrestricted GPU source SCF and synchronize its canonical orbital state.

        Kwargs:
            dm0 : cupy.ndarray, tuple of cupy.ndarray, or None
                Initial alpha/beta density passed to the GPU source SCF.
            **kwargs
                Additional keyword arguments forwarded to the source ``kernel``.

        Returns:
            float
                Converged unrestricted source SCF energy in Hartree.
        '''
        _check_gpu_contract(self)
        _check_xc(self)
        _clear_target_state(self)
        self.mf_source.converged = False
        self._copy_source_options(self, self.mf_source)
        self._prepare_source_grid()
        e_source = self.mf_source.kernel(dm0=dm0, **kwargs)
        if not self.mf_source.converged:
            raise RuntimeError('unrestricted GPU source SCF calculation did not converge')
        self._sync_source_results()
        _record_source_signature(self)
        return float(e_source)

    def run_source(self, dm0=None, **kwargs):
        '''Run unrestricted GPU source SCF and return this workflow object.'''
        self.kernel_source(dm0=dm0, **kwargs)
        return self

    def make_source_rdm1(self):
        '''Return alpha/beta densities from the attached source orbitals.'''
        _check_source_state(self)
        return self.mf_source.make_rdm1()

    @log_kernel_time('Energy workflow')
    def kernel(self, dm0=None, run_source=True, *, run_hf=None, **kwargs):
        '''Run unrestricted source SCF, then evaluate fixed-density target XC on the GPU.

        Kwargs:
            dm0 : cupy.ndarray, tuple of cupy.ndarray, or None
                Initial unrestricted source-SCF density. Valid only with ``run_source=True``.
            run_source : bool
                Reuse attached orbitals when false; the caller is responsible
                for source stationarity.
            **kwargs
                Additional keyword arguments passed only to unrestricted GPU source SCF.

        Returns:
            float
                ``E_target[P_source^alpha,P_source^beta]`` in the active representation.
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
        from dcforge.tools.logger import energy_summary
        energy_summary(self, unrestricted=True)
        return self.e_tot

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        '''Evaluate target non-SCF energy and optional D2C dispersion correction.'''
        _check_gpu_contract(self)
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
        '''Reset source and target GPU objects to the same molecule.'''
        uks.UKS.reset(self, mol)
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
                Coulomb-only fitting is not supported by this gradient
                capability.

        Returns:
            UNSC
                Copied workflow with one shared source/target GPU DF provider.
        '''
        if only_dfj:
            raise NotImplementedError('GPU unrestricted non-SCF currently requires only_dfj=False')
        _check_df_owner(self, with_df)
        source_was_converged = bool(self.mf_source.converged)
        auxbasis = default_auxbasis(
            self.mol, auxbasis=auxbasis, with_df=with_df)
        mf_source = df_jk.density_fit(
            self.mf_source, auxbasis=auxbasis,
            with_df=with_df,
            only_dfj=False,
        )
        mf = _DFUNSC(self, mf_source.with_df, False)
        mf.mf_source = mf_source
        _clear_target_state(mf)
        if not source_was_converged:
            _record_source_signature(mf)
        return mf

    def nuc_grad_method(self):
        '''Return the conventional or density-fitted GPU unrestricted gradient driver.'''
        _check_gpu_contract(self)
        if hasattr(self, 'with_df'):
            from dcforge.gpu.df.grad.unsc import Gradients
        else:
            from dcforge.gpu.grad.unsc import Gradients

        return Gradients(self)

    Gradients = nuc_grad_method

    def Hessian(self):
        raise NotImplementedError('GPU unrestricted non-SCF Hessians are not implemented')

    def as_scanner(self):
        from dcforge.tools.scanner import energy_scanner
        return energy_scanner(self)

    def to_cpu(self):
        raise NotImplementedError(
            'GPU-to-CPU UNSC state conversion is not implemented; '
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


class _DFUNSC(df_jk._DFHF, UNSC):
    '''GPU4PySCF density-fitting mixin for the unrestricted non-SCF workflow.'''

    device = UNSC.device
    nuc_grad_method = UNSC.nuc_grad_method
    Gradients = UNSC.nuc_grad_method
    Hessian = UNSC.Hessian
    to_cpu = UNSC.to_cpu
    to_gpu = UNSC.to_gpu

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
UHFDFT = UNSC
