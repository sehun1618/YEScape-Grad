'''Gamma-point GPU semilocal-DFA@RHF energy with one explicit DF provider.

The public API intentionally follows
``dcforge.cpu.pbc.energy.rhf_dft``.  The source calculation is GPU4PySCF RHF;
the target evaluates a supported DFA on that converged density and never
executes a target RKS SCF cycle.
'''

import warnings

import cupy as cp
import numpy as np
from gpu4pyscf.lib.cupy_helper import tag_array
from gpu4pyscf.pbc.dft import multigrid_v2, rks
from gpu4pyscf.pbc.gto import int1e
from gpu4pyscf.pbc.scf import hf
from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import gto

from dcforge.gpu.pbc import multigrid as dc_multigrid
from dcforge.gpu.pbc.df.fft import FFTDF
from dcforge.gpu.pbc.df.gdf import GDF
from dcforge.gpu.pbc.df.gdf.precision import (
    GDFBuildPrecisionWarning,
    GDFMixedPrecisionWarning,
    normalize_profile as normalize_gdf_precision_profile,
)
from dcforge.gpu.pbc.df.rsdf import RSDF

_SUPPORTED_TARGET_XC = ('PBE', 'R2SCAN')


def _target_xc_name(mf):
    '''Return the canonical name of the restricted periodic target DFA.'''
    return str(mf.xc).replace(' ', '').upper()


def _as_float(value):
    '''Return a real scalar from a Python, NumPy, or CuPy scalar.'''
    if value is None:
        return None
    if hasattr(value, 'get'):
        value = value.get()
    return float(np.asarray(value).real)


def _check_input(mf):
    '''Validate the restricted Gamma-point GPU semilocal-DFA@RHF capability.

    Args:
        mf : RHFDFT
            Active fixed-density periodic GPU workflow.

    Raises:
        NotImplementedError
            If the calculation is outside the implemented Stage-1 boundary.
        ValueError
            If source/target ownership or a finite FFT mesh is inconsistent.
    '''
    cell = mf.cell
    if cell.dimension != 3:
        raise NotImplementedError('periodic RHF-DFT currently requires a 3D cell')
    if cell.spin != 0 or cell.nelectron % 2:
        raise NotImplementedError('periodic RHF-DFT requires a closed-shell cell')
    if not cell._pseudo or cell._ecp:
        raise NotImplementedError('periodic RHF-DFT currently requires GTH pseudo')
    if not isinstance(mf.with_df, (FFTDF, GDF, RSDF)):
        raise NotImplementedError('unsupported periodic density-fitting provider')
    if mf.with_df is not mf.mf_hf.with_df:
        raise ValueError('source RHF and target RKS must share one GPU with_df')
    if mf.with_df.cell is not cell:
        raise ValueError('density-fitting provider must own the active Cell')
    if isinstance(mf.with_df, FFTDF) and not np.array_equal(
            np.asarray(mf.with_df.mesh), np.asarray(cell.mesh)):
        raise ValueError(
            'FFTDF energy requires with_df.mesh == cell.mesh'
        )
    if isinstance(mf._numint, multigrid_v2.MultiGridNumInt):
        if cell.ke_cutoff is None or cell.ke_cutoff <= 0:
            raise ValueError(
                'GPU multigrid XC requires a positive cell.ke_cutoff set '
                'before cell.build()')
        if mf._numint.cell is not cell:
            raise ValueError('GPU multigrid XC and RHFDFT must own the same Cell')
        if not np.array_equal(
                np.asarray(mf._numint.mesh), np.asarray(cell.mesh)):
            raise ValueError(
                'GPU multigrid XC mesh must equal the resolved cell.mesh')
    elif mf.multigrid_with_j:
        raise ValueError(
            'multigrid_with_j=True requires multigrid_numint(with_j=True)')
    if mf._numint.libxc.is_hybrid_xc(mf.xc):
        raise NotImplementedError(
            'periodic target functionals must be pure semilocal PBE or R2SCAN')
    if mf._numint.libxc.is_nlc(mf.xc) or mf.nlc not in ('', 0, False):
        raise NotImplementedError('periodic NLC/VV10 is not supported')
    if _target_xc_name(mf) not in _SUPPORTED_TARGET_XC:
        raise NotImplementedError(
            'periodic target functional must be PBE or R2SCAN')


class DFRHF(hf.RHF):
    '''Gamma-point GPU RHF whose ``with_df`` is selected by ``RHFDFT``.'''

    def get_hcore(self, cell=None, kpt=None):
        '''Return the one-electron Hamiltonian owned by the active provider.

        GPU4PySCF's generic RHF method selects a uniform-grid multigrid
        pseudopotential whenever the cell mesh is smaller than ``500^3``.
        A DC-Forge GDF or RSDF calculation instead delegates the
        pseudopotential to its shared Gaussian provider so zero-order local
        part 1 and the analytic derivative consume the same resolved SR/LR
        plan.  FFTDF retains the upstream dispatcher unchanged.
        '''
        if not isinstance(self.with_df, (GDF, RSDF)):
            return hf.RHF.get_hcore(self, cell=cell, kpt=kpt)
        if cell is None:
            cell = self.cell
        if kpt is None:
            kpt = self.kpt
        if cell is not self.with_df.cell:
            raise ValueError(
                'Gaussian Hcore and provider must own the same Cell')
        hcore = self.with_df.get_pp(kpt)
        return hcore + int1e.int1e_kin(cell, kpt)


class RHFDFT(rks.RKS):
    '''Evaluate a target DFA on converged GPU Gamma-point RHF orbitals.

    Args:
        cell : pyscf.pbc.gto.Cell
            Built periodic cell shared by source and target objects.

    Kwargs:
        xc : str
            Target functional.  ``'PBE'`` and ``'R2SCAN'`` are supported.
        mf_hf : gpu4pyscf.pbc.scf.hf.RHF or None
            Optional source RHF object.  It must share the exact active
            DC-Forge GPU periodic provider with this target object.

    Notes:
        ``kernel`` first converges source RHF.  It then calls inherited
        ``RKS.energy_tot(dm=dm_hf)`` exactly once.  Calling ``RKS.kernel`` on
        the target object would incorrectly perform a self-consistent DFA and
        is therefore not part of this workflow.
    '''

    _keys = rks.RKS._keys | {
        'mf_hf', 'e_hf', 'e_dft', 'dm_hf', 'target_fock',
        'target_fock_signature', '_source_converged', 'multigrid_backend',
        'multigrid_with_j', 'source_jk_backend',
        'source_precision_policy', 'source_precision_policy_effective',
        'source_precision_block_size', 'source_precision_copy_block_size',
        'source_precision_energy_normalization',
        'source_precision_adaptive_norm_fraction',
        'source_precision_coarse_minimum_cycle',
        'source_precision_coarse_energy_change_per_unit',
        'source_precision_coarse_density_change_norm',
        'source_precision_coarse_ao_commutator_norm',
        'source_precision_fine_minimum_cycle',
        'source_precision_fine_energy_change_per_unit',
        'source_precision_fine_density_change_norm',
        'source_precision_fine_ao_commutator_norm',
        'source_precision_final_energy_tol_per_unit',
        'source_precision_final_residual_tol',
        'source_precision_fallback', 'source_precision_report',
        'gdf_precision_profile', 'gdf_precision_profile_effective',
        'gdf_precision_report', '_gdf_precision_warning_profile',
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

        if mf_hf is None:
            provider = FFTDF(cell)
            mf_hf = DFRHF(cell, exxdiv=self.exxdiv)
            mf_hf.with_df = provider
        elif not isinstance(mf_hf, hf.RHF):
            raise TypeError('mf_hf must be a gpu4pyscf.pbc.scf.hf.RHF object')
        if mf_hf.cell is not cell:
            raise ValueError('mf_hf and RHFDFT must reference the same Cell')
        if not isinstance(mf_hf.with_df, (FFTDF, GDF, RSDF)):
            raise TypeError('mf_hf.with_df must be a dcforge GPU PBC provider')

        self.mf_hf = mf_hf
        self.with_df = mf_hf.with_df
        self.e_hf = None
        self.e_dft = None
        self.dm_hf = None
        self.target_fock = None
        self.target_fock_signature = None
        self._source_converged = False
        self.multigrid_backend = None
        self.multigrid_with_j = False
        self.source_jk_backend = self.with_df.backend_name
        # Native FP64 remains the default.  The guarded mixed route must be
        # requested explicitly and is limited to positive real-Gamma GDF RHF.
        self.source_precision_policy = 'fp64'
        self.source_precision_policy_effective = 'fp64'
        self.source_precision_block_size = 256
        self.source_precision_copy_block_size = 256
        self.source_precision_energy_normalization = 1
        self.source_precision_adaptive_norm_fraction = .9
        self.source_precision_coarse_minimum_cycle = 2
        self.source_precision_coarse_energy_change_per_unit = 1.e-4
        self.source_precision_coarse_density_change_norm = 1.e-1
        self.source_precision_coarse_ao_commutator_norm = 1.e-1
        self.source_precision_fine_minimum_cycle = 4
        self.source_precision_fine_energy_change_per_unit = 1.e-7
        self.source_precision_fine_density_change_norm = 1.e-3
        self.source_precision_fine_ao_commutator_norm = 1.e-3
        self.source_precision_final_energy_tol_per_unit = 1.e-7
        self.source_precision_final_residual_tol = 1.e-5
        self.source_precision_fallback = True
        self.source_precision_report = None
        self.gdf_precision_profile = 'fp64'
        self.gdf_precision_profile_effective = 'fp64'
        self.gdf_precision_report = None
        self._gdf_precision_warning_profile = None

    @property
    def device(self):
        '''Return the backend owner without moving any state to the host.'''
        return 'gpu'

    def dump_flags(self, verbose=None):
        '''Log fixed-density and GDF precision controls in PySCF style.'''
        rks.RKS.dump_flags(self, verbose)
        logger.info(self, 'gdf_precision_profile = %s',
                    self.gdf_precision_profile)
        logger.info(self, 'gdf_precision_profile_effective = %s',
                    self.gdf_precision_profile_effective)
        logger.info(self, 'source_precision_policy = %s',
                    self.source_precision_policy)
        logger.info(self, 'source_precision_policy_effective = %s',
                    self.source_precision_policy_effective)
        if self.gdf_precision_profile == 'fp32_build':
            logger.warn(
                self, 'fp32_build uses an approximate FP32/complex64 CDERI '
                'transformation; SCF/CPHF residuals do not bound its error '
                'relative to fp64')
        return self

    def kernel_hf(self, dm0=None, **kwargs):
        '''Run source GPU RHF with the active FFTDF/GDF/RSDF provider.

        Kwargs:
            dm0 : cupy.ndarray or None
                Optional initial source-RHF AO density.
            **kwargs
                Keyword arguments forwarded only to source RHF.

        Returns:
            float
                Converged source RHF energy in Hartree.
        '''
        _check_input(self)
        self._apply_gdf_precision_profile()
        self._copy_hf_options(self, self.mf_hf)
        policy = str(self.source_precision_policy).strip().lower()
        if policy not in ('fp64', 'mixed', 'persistent_mixed'):
            raise ValueError(
                'source_precision_policy must be fp64, mixed, or '
                'persistent_mixed')
        self.source_precision_report = None
        if policy == 'fp64':
            e_hf = self.mf_hf.kernel(dm0=dm0, **kwargs)
            self.source_precision_policy_effective = 'fp64'
        else:
            if not isinstance(self.with_df, GDF):
                raise NotImplementedError(
                    'source mixed precision requires GPU real-Gamma GDF')
            from dcforge.gpu.pbc.scf import run_mixed_source_scf

            e_hf, report = run_mixed_source_scf(
                self.mf_hf, dm0=dm0,
                block_size=self.source_precision_block_size,
                copy_block_size=self.source_precision_copy_block_size,
                energy_normalization=(
                    self.source_precision_energy_normalization),
                adaptive_norm_fraction=(
                    self.source_precision_adaptive_norm_fraction),
                thresholds=self._source_precision_thresholds(),
                final_energy_tol_per_unit=(
                    self.source_precision_final_energy_tol_per_unit),
                final_residual_tol=(
                    self.source_precision_final_residual_tol),
                fallback=bool(self.source_precision_fallback),
                kernel_kwargs=kwargs,
                persistent=policy == 'persistent_mixed')
            self.source_precision_report = report
            self.source_precision_policy_effective = report[
                'effective_policy']
        self.gdf_precision_profile_effective = (
            self.with_df.gdf_precision_profile_effective
            if isinstance(self.with_df, GDF) else 'fp64')
        self.gdf_precision_report = (
            None if not isinstance(self.with_df, GDF) else {
                'requested_profile': self.gdf_precision_profile,
                'effective_profile': self.gdf_precision_profile_effective,
                'provider': self.with_df.gdf_precision_report,
                'source_scf': self.source_precision_report,
            })
        if not self.mf_hf.converged:
            raise RuntimeError('periodic source RHF calculation did not converge')
        self._source_converged = True
        self._sync_hf_results()
        return _as_float(e_hf)

    def run_hf(self, dm0=None, **kwargs):
        '''Run source RHF and return this fixed-density workflow.'''
        self.kernel_hf(dm0=dm0, **kwargs)
        return self

    def make_hf_rdm1(self):
        '''Return the converged closed-shell GPU RHF AO density matrix.'''
        if not self.mf_hf.converged or not self._source_converged:
            raise RuntimeError('periodic source RHF must be converged')
        return self.mf_hf.make_rdm1()

    def kernel(self, dm0=None, run_hf=True, **kwargs):
        '''Run RHF and evaluate ``E_DFA[P_HF]`` without target SCF.

        Kwargs:
            dm0 : cupy.ndarray or None
                Initial source-RHF density, valid only if ``run_hf=True``.
            run_hf : bool
                Reuse a source RHF result only if it was converged with the
                current shared provider.
            **kwargs
                Options forwarded only to the source RHF kernel.

        Returns:
            float
                Fixed-density target-DFA@RHF energy in Hartree.
        '''
        _check_input(self)
        if run_hf:
            self.kernel_hf(dm0=dm0, **kwargs)
        else:
            if dm0 is not None or kwargs:
                raise ValueError('dm0/kwargs require run_hf=True')
            if not self.mf_hf.converged or not self._source_converged:
                raise RuntimeError(
                    'run_hf=False requires a source converged with the current GPU representation'
                )
            self._sync_hf_results()

        self.dm_hf = self.make_hf_rdm1()
        h1e = self.get_hcore(self.cell)
        vhf = self.get_veff(self.cell, self.dm_hf)
        self.e_coul = _as_float(getattr(vhf, 'ecoul', None))
        self.e_xc = _as_float(getattr(vhf, 'exc', None))
        self.target_fock = h1e + vhf
        self.target_fock_signature = _target_fock_signature(self, self.dm_hf)
        self.e_dft = _as_float(self.energy_tot(
            dm=self.dm_hf, h1e=h1e, vhf=vhf))
        self.e_tot = self.e_dft
        return self.e_dft

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        '''Evaluate a supplied-density periodic target-DFA energy.'''
        _check_input(self)
        return rks.RKS.energy_tot(self, dm, h1e, vhf)

    def get_veff(self, cell=None, dm=None, dm_last=None, vhf_last=None,
                 hermi=1, kpt=None, kpts_band=None):
        r'''Return the selected target Hartree-plus-XC potential.

        GPU4PySCF multigrid normally owns both Coulomb and XC.  DC-Forge
        ``multigrid_with_j=False`` requests XC only and obtains target Hartree
        from the source FFTDF/GDF/RSDF provider.  ``True``
        requests the fused multigrid J+XC potential and Coulomb energy.
        Source RHF and CPHF keep their original provider/backend in either
        case.

        The arguments mirror GPU4PySCF ``RKS.get_veff``.  Incremental
        ``dm_last``/``vhf_last`` reuse is intentionally not claimed for this
        target split because the fixed-density target is evaluated once.
        '''
        if not isinstance(self._numint, multigrid_v2.MultiGridNumInt):
            return rks.RKS.get_veff(
                self, cell=cell, dm=dm, dm_last=dm_last,
                vhf_last=vhf_last, hermi=hermi, kpt=kpt,
                kpts_band=kpts_band)
        if cell is None:
            cell = self.cell
        if dm is None:
            dm = self.make_rdm1()
        if kpt is None:
            kpt = self.kpt
        ni = self._numint
        with_j = bool(self.multigrid_with_j)
        _nelec, exc, vxc = ni.nr_rks(
            cell, self.grids, self.xc, dm, 0, hermi, kpt, kpts_band,
            with_j=with_j, verbose=self.verbose)
        if with_j:
            return vxc
        # The source SCF object is the authoritative J/K dispatcher and
        # delegates target Hartree to the shared FFTDF/GDF/RSDF provider.
        vj, _vk = self.mf_hf.get_jk(
            cell=cell, dm=cp.asarray(dm), hermi=hermi, kpt=kpt,
            kpts_band=kpts_band, with_j=True, with_k=False)
        if vj is None:
            raise RuntimeError('active DF provider did not return target Hartree J')
        vj = cp.asarray(vj)
        ecoul = .5 * cp.einsum('ij,ji->', cp.asarray(dm), vj).real
        return tag_array(cp.asarray(vxc) + vj, ecoul=ecoul, exc=exc)

    def get_hcore(self, cell=None, kpt=None):
        '''Return the one-electron Hamiltonian used by source RHF.

        GPU4PySCF may choose an internal multigrid helper for a one-electron
        pseudopotential integral.  Delegating to the source object ensures
        source RHF and the fixed-density target differentiate the same
        finite representation; it does not invoke target RKS SCF.
        '''
        return self.mf_hf.get_hcore(cell=cell, kpt=kpt)

    def fft_density_fit(self, mesh=None):
        '''Replace source and target by one explicit GPU FFTDF provider.'''
        return self._set_provider(FFTDF(self.cell, mesh=mesh))

    def density_fit(self, auxbasis=None, with_df=None):
        '''Replace source and target by one shared Gamma-point GPU GDF provider.'''
        if with_df is None:
            with_df = GDF(self.cell, auxbasis=auxbasis)
        elif not isinstance(with_df, GDF):
            raise TypeError('with_df must be dcforge GPU PBC GDF')
        return self._set_provider(with_df)

    def set_gdf_precision_profile(
            self, profile='fp64', block_size=256,
            energy_normalization=1):
        r'''Select the real-Gamma GDF precision policy.

        Args:
            profile : str
                ``'fp64'`` retains the reference GPU4PySCF GDF build,
                storage, source SCF, Z-vector and derivative contractions.
                ``'mixed'`` builds CDERI in FP64, stores it in FP32, and uses
                staged mixed-precision SCF/Z/derivative contractions.
                ``'fp32_build'`` additionally evaluates the large CDERI
                metric transformation and reciprocal contraction in
                FP32/complex64.
            block_size : int
                Auxiliary rows transferred for each source J/K contraction.
            energy_normalization : int
                Positive physical count used only by SCF precision-transition
                gates.  For a water-only Cell this is normally ``nwater``.

        Returns:
            RHFDFT
                This workflow, following PySCF's chainable style.

        Notes:
            ``mixed`` is an opt-in moderate approximation.  It discards CDERI
            bits only after the native FP64 transformation and must be checked
            against ``fp64`` on a representative subset.

            ``fp32_build`` is an aggressive experimental approximation.  FP32
            roundoff is accumulated while CDERI is formed, so a converged SCF
            or CPHF residual proves stationarity only for that approximate
            Hamiltonian.  It does not bound the error relative to ``fp64``.
            Recovering the reference result requires rebuilding CDERI.

            Both accelerated profiles currently require a positive-metric,
            single-GPU, three-dimensional real-Gamma GDF calculation with
            mapped-host CDERI.  Sampled k-points, RSDF and stress are excluded.
        '''
        profile = normalize_gdf_precision_profile(profile)
        if not isinstance(self.with_df, GDF):
            raise NotImplementedError(
                'GDF precision profiles require density_fit/GDF')
        if int(block_size) < 1:
            raise ValueError('GDF precision block_size must be positive')
        if int(energy_normalization) < 1:
            raise ValueError('energy_normalization must be positive')
        self.gdf_precision_profile = profile
        self.gdf_precision_profile_effective = None
        self.gdf_precision_report = None
        self.source_precision_block_size = int(block_size)
        self.source_precision_copy_block_size = int(block_size)
        self.source_precision_energy_normalization = int(
            energy_normalization)
        self._apply_gdf_precision_profile(force_warning=True)
        return self

    def source_mixed_precision(
            self, block_size=256, copy_block_size=256,
            energy_normalization=1, adaptive_norm_fraction=.9,
            fallback=True):
        '''Enable guarded mixed-to-native FP64 source RHF.

        Args:
            block_size : int
                Auxiliary rows per FP32 J/K contraction.
            copy_block_size : int
                Auxiliary rows converted while building the mapped-host FP32
                sidecar.
            energy_normalization : int
                Positive physical count used only to normalize SCF transition
                gates.  For a water-only Cell this is normally ``nwater``.
            adaptive_norm_fraction : float
                Factor-norm fraction whose adaptive K outer product is FP64.
            fallback : bool
                Resume from the current density with native FP64 when an
                allocation or the mandatory final audit fails.

        Returns:
            RHFDFT
                This workflow, following PySCF's chainable configuration style.
        '''
        if not isinstance(self.with_df, GDF):
            raise NotImplementedError(
                'source_mixed_precision requires density_fit/GDF')
        if int(block_size) < 1 or int(copy_block_size) < 1:
            raise ValueError('source mixed-precision block sizes must be positive')
        if int(energy_normalization) < 1:
            raise ValueError('energy_normalization must be positive')
        if not 0. < float(adaptive_norm_fraction) <= 1.:
            raise ValueError('adaptive_norm_fraction must be in (0,1]')
        self.source_precision_policy = 'mixed'
        self.source_precision_block_size = int(block_size)
        self.source_precision_copy_block_size = int(copy_block_size)
        self.source_precision_energy_normalization = int(
            energy_normalization)
        self.source_precision_adaptive_norm_fraction = float(
            adaptive_norm_fraction)
        self.source_precision_fallback = bool(fallback)
        if str(self.with_df.cderi_storage).lower() == 'auto':
            self.with_df.cderi_auto_sidecar_factor = max(
                .5, float(self.with_df.cderi_auto_sidecar_factor))
        return self

    def rs_density_fit(self, auxbasis=None, with_df=None):
        '''Replace source and target by one shared Gamma-point GPU RSDF provider.'''
        if with_df is None:
            with_df = RSDF(self.cell, auxbasis=auxbasis)
        elif not isinstance(with_df, RSDF):
            raise TypeError('with_df must be dcforge GPU PBC RSDF')
        return self._set_provider(with_df)

    def multigrid_numint(self, mesh=None, ntasks=None, ke_ratio=None,
                         rel_cutoff=None, with_j=False):
        r'''Return a copy whose target XC optionally includes multigrid J.

        Source RHF and source CPHF always remain owned by the shared GPU
        provider.  ``with_j=False`` delegates only target XC to multigrid;
        ``with_j=True`` delegates fixed-density target J+XC and its explicit
        derivative together.

        Args:
            mesh : None
                A separate target mesh is intentionally unsupported.  Set
                ``cell.ke_cutoff`` before ``cell.build()`` so source/provider
                and target-XC mesh ownership remain explicit.
            ntasks, ke_ratio, rel_cutoff : None
                Retained for the CPU-compatible public signature.  GPU4PySCF
                1.8.1 multigrid-v2 does not expose these hierarchy controls;
                passing them is rejected instead of silently ignoring a
                requested numerical representation.
            with_j : bool
                Select fused target multigrid J+XC instead of XC only.

        Returns:
            RHFDFT
                A shallow copied fixed-density workflow with the same source
                RHF and DF provider but a new multigrid integrator with the
                requested target-Hartree ownership.
        '''
        if mesh is not None:
            raise ValueError(
                'set cell.ke_cutoff before cell.build(); a separate GPU '
                'multigrid XC mesh is not supported')
        if any(value is not None for value in (ntasks, ke_ratio, rel_cutoff)):
            raise NotImplementedError(
                'GPU4PySCF multigrid-v2 does not expose ntasks, ke_ratio, '
                'or rel_cutoff controls')
        if not isinstance(with_j, (bool, np.bool_)):
            raise TypeError('multigrid with_j must be boolean')
        with_j = bool(with_j)
        if self.cell.ke_cutoff is None or self.cell.ke_cutoff <= 0:
            raise ValueError(
                'multigrid_numint requires a positive cell.ke_cutoff set '
                'before cell.build()')
        mf = self.copy()
        ni = dc_multigrid.MultiGridNumInt(self.cell)
        ni.mesh = np.asarray(self.cell.mesh, dtype=int).copy()
        mf._numint = ni
        mf.multigrid_with_j = with_j
        mf.multigrid_backend = (
            'gpu4pyscf_multigrid_v2_j_xc' if with_j
            else 'gpu4pyscf_multigrid_v2_xc_only')
        mf.e_dft = mf.dm_hf = mf.target_fock = None
        mf.target_fock_signature = None
        if mf.e_hf is not None:
            mf.e_tot = mf.e_hf
        return mf

    def reset(self, cell=None):
        '''Reset geometry-dependent state and fixed-density results.

        Args:
            cell : pyscf.pbc.gto.Cell or None
                New periodic cell.  ``None`` invalidates intermediates while
                retaining the current geometry.

        Returns:
            RHFDFT
                This workflow after the source, target, provider, grid, and
                numerical-integration reset hooks have been applied.

        Notes:
            Source RHF and target RKS share one provider, so the two inherited
            reset calls may invalidate that provider twice.  Both operations
            are idempotent and preserve the shared-object identity.  The
            multigrid backend label describes the selected integrator and
            survives reset; only its geometry-dependent tasks are discarded.
        '''
        rks.RKS.reset(self, cell)
        self.mf_hf.reset(cell)
        self.with_df = self.mf_hf.with_df
        self.e_hf = self.e_dft = self.dm_hf = self.target_fock = None
        self.target_fock_signature = None
        self._source_converged = False
        self.source_precision_policy_effective = (
            'fp64' if self.source_precision_policy == 'fp64' else None)
        self.source_precision_report = None
        self.gdf_precision_profile_effective = (
            'fp64' if self.gdf_precision_profile == 'fp64' else None)
        self.gdf_precision_report = None
        self.source_jk_backend = getattr(
            self.mf_hf, 'dcforge_source_jk_backend',
            self.with_df.backend_name)
        self.multigrid_backend = (
            ('gpu4pyscf_multigrid_v2_j_xc' if self.multigrid_with_j
             else 'gpu4pyscf_multigrid_v2_xc_only')
            if isinstance(self._numint, multigrid_v2.MultiGridNumInt)
            else None)
        return self

    def nuc_grad_method(self):
        '''Return the Gamma FFTDF/GDF/RSDF gradient driver.'''
        if getattr(self.mf_hf, 'dcforge_experimental_source', None):
            raise NotImplementedError(
                'the active experimental source RHF has no analytic gradient')
        if not isinstance(self.with_df, (FFTDF, GDF, RSDF)):
            raise NotImplementedError('unsupported GPU PBC analytic-gradient provider')
        from dcforge.gpu.pbc.grad.rhf_dft import Gradients

        mf_grad = Gradients(self)
        if self.gdf_precision_profile in ('mixed', 'fp32_build'):
            mf_grad.precision_policy = 'mixed'
            mf_grad.gdf_derivative_precision = 'mixed'
        return mf_grad

    Gradients = nuc_grad_method

    def Hessian(self):
        raise NotImplementedError('GPU PBC RHF-DFT Hessians are not implemented')

    def as_scanner(self):
        '''Return a stateful fixed-density energy scanner in PySCF style.'''
        return as_scanner(self)

    def to_cpu(self):
        raise NotImplementedError(
            'GPU-to-CPU PBC RHFDFT state conversion is not implemented; '
            'construct an explicit CPU workflow'
        )

    def to_gpu(self, device=None):
        '''Return this already GPU-owned workflow.'''
        return self

    def _sync_hf_results(self):
        mf_hf = self.mf_hf
        self.e_hf = _as_float(mf_hf.e_tot)
        self.e_dft = self.dm_hf = self.target_fock = None
        self.target_fock_signature = None
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

    def _source_precision_thresholds(self):
        '''Return the scalar mixed-to-native transition ledger for one run.'''
        return {
            'coarse': {
                'minimum_cycle': self.source_precision_coarse_minimum_cycle,
                'energy_change_per_unit_ha': (
                    self.source_precision_coarse_energy_change_per_unit),
                'density_change_norm': (
                    self.source_precision_coarse_density_change_norm),
                'ao_commutator_norm': (
                    self.source_precision_coarse_ao_commutator_norm),
            },
            'fine': {
                'minimum_cycle': self.source_precision_fine_minimum_cycle,
                'energy_change_per_unit_ha': (
                    self.source_precision_fine_energy_change_per_unit),
                'density_change_norm': (
                    self.source_precision_fine_density_change_norm),
                'ao_commutator_norm': (
                    self.source_precision_fine_ao_commutator_norm),
            },
        }

    def _apply_gdf_precision_profile(self, force_warning=False):
        '''Synchronize one public profile across provider and source SCF.'''
        profile = normalize_gdf_precision_profile(
            self.gdf_precision_profile)
        if profile != 'fp64' and not isinstance(self.with_df, GDF):
            raise NotImplementedError(
                'GDF precision profiles require density_fit/GDF')
        if isinstance(self.with_df, GDF):
            built = self.with_df._cderi is not None
            built_profile = self.with_df.gdf_precision_profile_effective
            if built and built_profile != profile:
                # A factor cannot change precision by relabeling or casting.
                # Invalidate source orbitals and rebuild from primitives under
                # the requested profile, matching PySCF's reset-on-Hamiltonian
                # change semantics.
                self.reset()
            self.with_df.gdf_precision_profile = profile
            self.with_df.gdf_derivative_precision = (
                'mixed' if profile != 'fp64' else 'fp64')
        if profile != 'fp64':
            self.source_precision_policy = 'persistent_mixed'
            # There is no separate sidecar: source SCF and response reuse the
            # provider-owned persistent FP32 factor.
            self.with_df.cderi_auto_sidecar_factor = 0.
            if (force_warning
                    or self._gdf_precision_warning_profile != profile):
                if profile == 'mixed':
                    warnings.warn(
                        "gdf_precision_profile='mixed' stores the completed "
                        'FP64-built CDERI in FP32 and uses mixed SCF/Z/gradient '
                        'contractions. Validate energy and force errors against '
                        "'fp64' on a representative subset.",
                        GDFMixedPrecisionWarning, stacklevel=3)
                else:
                    warnings.warn(
                        "gdf_precision_profile='fp32_build' is an aggressive "
                        'experimental approximation: the CDERI transformation '
                        'itself uses FP32/complex64 and accumulated roundoff can '
                        'bias energies and forces. SCF/CPHF residuals do not '
                        "measure that bias. Rebuild with 'fp64' for an oracle.",
                        GDFBuildPrecisionWarning, stacklevel=3)
                self._gdf_precision_warning_profile = profile
        elif self.source_precision_policy == 'persistent_mixed':
            self.source_precision_policy = 'fp64'
        self.gdf_precision_profile_effective = (
            'fp64' if profile == 'fp64' else None)

    def _set_provider(self, provider):
        if provider.cell is not self.cell:
            raise ValueError('density-fitting provider must own the active Cell')
        if (self.gdf_precision_profile != 'fp64'
                and not isinstance(provider, GDF)):
            raise NotImplementedError(
                'non-FP64 GDF precision profiles cannot be transferred to '
                'FFTDF or RSDF')
        if getattr(self.mf_hf, 'dcforge_experimental_source', None):
            mf_hf = DFRHF(self.cell, exxdiv=self.exxdiv)
            self._copy_hf_options(self, mf_hf)
            self.mf_hf = mf_hf
        self.with_df = self.mf_hf.with_df = provider
        # A source object formerly run with FFTDF can retain GPU4PySCF's
        # auxiliary numerical-integrator helper.  It belongs to the previous
        # provider state and must not leak into a new GDF calculation.
        self.mf_hf._numint = None
        self.e_hf = self.e_dft = self.dm_hf = self.target_fock = None
        self.target_fock_signature = None
        self._source_converged = False
        self.source_jk_backend = provider.backend_name
        self.source_precision_policy_effective = (
            'fp64' if self.source_precision_policy == 'fp64' else None)
        self.source_precision_report = None
        if isinstance(provider, GDF):
            provider.gdf_precision_profile = self.gdf_precision_profile
            provider.gdf_derivative_precision = (
                'mixed' if self.gdf_precision_profile != 'fp64' else 'fp64')
        self.gdf_precision_profile_effective = (
            'fp64' if self.gdf_precision_profile == 'fp64' else None)
        self.gdf_precision_report = None
        self.converged = False
        return self

    scf = kernel


def as_scanner(mf):
    '''Create a callable GPU periodic ``E_DFA[P_RHF]`` energy scanner.

    Args:
        mf : RHFDFT
            Fixed-density Gamma-point GPU workflow.

    Returns:
        RHFDFTScanner
            Callable scanner.  ``scanner(cell_or_geom, **kwargs)`` returns the
            target DFA energy in Hartree.

    Notes:
        Like PySCF's SCF scanner, this object has side effects: the active
        ``Cell``, source RHF state, target state, provider factors, grids, and
        multigrid tasks are replaced or reset at every call.  If the AO layout
        is unchanged, the preceding source-RHF density is supplied as ``dm0``;
        this is an initial guess only and source RHF is reconverged before the
        fixed-density target energy is evaluated.
    '''
    if isinstance(mf, lib.SinglePointScanner):
        return mf
    if not isinstance(mf, RHFDFT):
        raise TypeError('GPU PBC energy scanner requires RHFDFT')

    logger.info(mf, 'Create scanner for %s', mf.__class__)
    name = mf.__class__.__name__ + RHFDFTScanner.__name_mixin__
    return lib.set_class(
        RHFDFTScanner(mf), (RHFDFTScanner, mf.__class__), name)


class RHFDFTScanner(lib.SinglePointScanner):
    '''PySCF-style scanner that reconverges RHF at each periodic geometry.'''

    def __init__(self, mf):
        self.__dict__.update(mf.__dict__)
        self._last_cell_ao_loc = np.asarray(mf.cell.ao_loc).copy()

    def __call__(self, cell_or_geom, **kwargs):
        '''Evaluate one geometry and retain its source density as the next guess.

        Args:
            cell_or_geom : pyscf.pbc.gto.Cell or geometry-like
                A built periodic cell, or any geometry accepted by
                ``Cell.set_geom_`` while retaining the current lattice,
                basis, pseudopotential, precision, and cutoff settings.

        Kwargs:
            dm0 : cupy.ndarray or None
                Explicit source-RHF initial density.  When omitted, the
                preceding converged density is reused only if ``ao_loc`` is
                unchanged.
            **kwargs
                Additional options forwarded to :meth:`RHFDFT.kernel` and
                therefore to the source RHF calculation.

        Returns:
            float
                Fixed-density target-DFA energy in Hartree.
        '''
        if isinstance(cell_or_geom, gto.Cell):
            cell = cell_or_geom
        else:
            cell = self.cell.set_geom_(cell_or_geom, inplace=False)

        previous_coeff = self.mf_hf.mo_coeff
        previous_occ = self.mf_hf.mo_occ
        self.reset(cell)

        if 'dm0' in kwargs:
            dm0 = kwargs.pop('dm0')
        elif (previous_coeff is not None and previous_occ is not None
              and np.array_equal(self._last_cell_ao_loc, cell.ao_loc)):
            dm0 = self.mf_hf.make_rdm1(previous_coeff, previous_occ)
        else:
            dm0 = None

        # RHFDFT owns a target facade and a separate source SCF object.  Clear
        # both MO handles so a Newton/SOSCF wrapper cannot consume orbitals
        # from the preceding geometry; ``dm0`` above is the sole intended
        # geometry-to-geometry state transfer.
        self.mo_coeff = None
        self.mf_hf.mo_coeff = None
        energy = self.kernel(dm0=dm0, **kwargs)
        self._last_cell_ao_loc = np.asarray(cell.ao_loc).copy()
        return energy


def energy_tot(mf_hf, mf_dft):
    '''Evaluate periodic target DFT energy with a converged GPU RHF density.'''
    if not mf_hf.converged:
        raise RuntimeError('periodic source RHF calculation must be converged')
    if mf_hf.cell is not mf_dft.cell:
        raise ValueError('source and target must reference the same Cell')
    return _as_float(mf_dft.energy_tot(dm=mf_hf.make_rdm1()))


def _target_fock_signature(mf, dm):
    '''Return immutable state that defines a reusable GPU target Fock.'''
    provider = mf.with_df
    values = [
        str(mf.xc), id(provider), bool(mf.multigrid_with_j),
        np.asarray(mf.cell.atom_coords()).tobytes(),
        np.asarray(mf.cell.lattice_vectors()).tobytes(),
        float(mf.cell.ke_cutoff) if mf.cell.ke_cutoff is not None else None,
        float(mf.cell.precision),
        np.asarray(mf.cell.mesh, dtype=int).tobytes(),
        tuple(dm.shape),
    ]
    for name in ('mesh', 'auxbasis'):
        value = getattr(provider, name, None)
        if value is None:
            values.append(None)
        elif isinstance(value, str):
            values.append(value)
        else:
            values.append(np.asarray(value).tobytes())
    if isinstance(mf._numint, multigrid_v2.MultiGridNumInt):
        values.extend((
            mf.multigrid_backend,
            np.asarray(mf._numint.mesh, dtype=int).tobytes(),
        ))
    else:
        values.append('uniform_grid')
    return tuple(values)
