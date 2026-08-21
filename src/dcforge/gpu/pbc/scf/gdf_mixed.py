r'''Guarded mixed-precision source RHF for positive real-Gamma GDF.

The large repeated three-center J/K contractions use an FP32 copy or the
provider-owned persistent FP32 form of the completed GDF factor.  Density
matrices, DIIS, diagonalization, returned J/K, between-block sums, Ewald
correction, final source orbitals and the accepted source energy remain FP64.
The legacy guarded operator changes monotonically

``FP32 block J/K -> adaptive FP64 K outer product -> native FP64 GDF``.

Every transition clears DIIS and forces the following Fock build to use the
full density.  A final native-FP64 Fock audit is mandatory.  If the mixed route
does not reach and iterate with the native operator, or the physical residual
fails its gate, the calculation resumes from the current density with the
ordinary GPU4PySCF FP64 GDF operator.

The public ``mixed`` and ``fp32_build`` GDF profiles instead retain FP32
storage throughout and finish with blockwise FP64 J/K arithmetic over that
quantized factor.  This checks convergence of the selected approximate
Hamiltonian; it cannot recreate or audit against the discarded FP64 CDERI.
'''

import time

import cupy as cp
import numpy as np
from gpu4pyscf.lib.cupy_helper import contract
from gpu4pyscf.pbc.df.fft_jk import _ewald_exxdiv_for_G0
from pyscf import lib

from dcforge.gpu.pbc.df.cderi import copy_gamma_cderi_fp32


FP32_BLOCK = 'fp32_block_fp64_sum'
FP32_RHOK = 'fp32_rhok_fp64_outer'
FP32_ADAPTIVE = 'adaptive_rhok_fp64_outer'
FP32_STORAGE_FP64 = 'fp32_storage_fp64_compute'
NATIVE_FP64 = 'native_fp64'

DEFAULT_THRESHOLDS = {
    'coarse': {
        'minimum_cycle': 2,
        'energy_change_per_unit_ha': 1.e-4,
        'density_change_norm': 1.e-1,
        'ao_commutator_norm': 1.e-1,
    },
    'fine': {
        'minimum_cycle': 4,
        'energy_change_per_unit_ha': 1.e-7,
        'density_change_norm': 1.e-3,
        'ao_commutator_norm': 1.e-3,
    },
}


def _scalar(value):
    if value is None:
        return None
    if hasattr(value, 'get'):
        value = value.get()
    return float(np.asarray(value).real)


def _maximum(value):
    value = cp.asarray(value)
    return float(cp.max(cp.abs(value)).get()) if value.size else 0.


def _clear_diis(diis):
    '''Clear GPU4PySCF DIIS history without replacing its configured class.'''
    if diis is None:
        return False
    for name in ('_buffer', '_bookkeep'):
        value = getattr(diis, name, None)
        if value is not None:
            value.clear()
    for name, value in (
            ('_head', 0), ('_H', None), ('_xprev', None),
            ('_err_vec_touched', False)):
        if hasattr(diis, name):
            setattr(diis, name, value)
    return True


class GammaFP32JK:
    r'''Evaluate GDF J/K from mapped-host FP32 compressed Gamma factors.

    Args:
        provider : dcforge.gpu.pbc.df.gdf.GDF
            Built positive-metric real-Gamma provider.
        storage : numpy.ndarray
            Persistent mapped-host FP32 factor with shape ``(naux,npair)``.

    Kwargs:
        block_size : int
            Maximum number of auxiliary rows transferred per contraction.
        policy : str
            ``fp32_block_fp64_sum`` keeps both large contractions in FP32.
            ``adaptive_rhok_fp64_outer`` promotes the K outer product for the
            highest-norm blocks covering ``adaptive_norm_fraction``.
        adaptive_norm_fraction : float
            Static squared-factor-norm fraction assigned the FP64 K outer
            product in the adaptive tier.
    '''

    backend = 'dcforge_gdf_mixed_gamma_jk'

    def __init__(self, provider, storage, *, block_size, policy,
                 adaptive_norm_fraction=.9):
        if not bool(getattr(provider, 'is_gamma_point', False)):
            raise NotImplementedError(
                'mixed source J/K requires a real-Gamma GDF provider')
        if np.dtype(storage.dtype) != np.dtype(np.float32):
            raise TypeError('mixed source J/K requires FP32 CDERI storage')
        if getattr(provider, '_cderip', None):
            raise NotImplementedError(
                'mixed source J/K requires a positive 3D GDF metric')
        if set(provider._cderi) != {0}:
            raise NotImplementedError(
                'mixed source J/K requires one positive Gamma factor')
        if policy not in (
                FP32_BLOCK, FP32_RHOK, FP32_ADAPTIVE,
                FP32_STORAGE_FP64):
            raise ValueError('unknown mixed source J/K policy')
        if int(block_size) < 1:
            raise ValueError('mixed source block_size must be positive')
        if not 0. < float(adaptive_norm_fraction) <= 1.:
            raise ValueError('adaptive_norm_fraction must be in (0,1]')
        self.provider = provider
        self.storage = storage
        self.block_size = min(int(block_size), int(storage.shape[0]))
        self.policy = policy
        self.adaptive_norm_fraction = float(adaptive_norm_fraction)
        self.calls = self.j_calls = self.k_calls = 0
        self.block_count = self.host_to_device_bytes = 0
        self.wall_seconds = 0.

        nao = int(provider.nao)
        pair_address = cp.asarray(provider._cderi_idx[0], dtype=cp.int64)
        self.rows, self.cols = divmod(pair_address, nao)
        self.diag = cp.asarray(provider._cderi_idx[1], dtype=cp.int64)
        self._dense_buffers = {}
        self._sensitive_blocks = self._select_sensitive_blocks()

    def _working_dtype(self):
        '''Return the device arithmetic dtype for the current tier.'''
        return (
            cp.float64 if self.policy == FP32_STORAGE_FP64 else cp.float32)

    def _dense_buffer(self, dtype):
        key = np.dtype(dtype).str
        if key not in self._dense_buffers:
            self._dense_buffers[key] = cp.zeros(
                (self.block_size, int(self.provider.nao),
                 int(self.provider.nao)), dtype=dtype)
        return self._dense_buffers[key]

    def _select_sensitive_blocks(self):
        if self.policy != FP32_ADAPTIVE:
            return frozenset()
        norms = []
        for index, p0 in enumerate(
                range(0, int(self.storage.shape[0]), self.block_size)):
            p1 = min(int(self.storage.shape[0]), p0 + self.block_size)
            block = np.asarray(self.storage[p0:p1], dtype=np.float32)
            norms.append((
                float(np.sum(np.square(block), dtype=np.float64)), index))
        target = self.adaptive_norm_fraction * sum(
            value for value, _index in norms)
        selected = []
        accumulated = 0.
        for value, index in sorted(norms, reverse=True):
            selected.append(index)
            accumulated += value
            if accumulated >= target:
                break
        return frozenset(selected)

    def _use_fp64_outer(self, block_index):
        return (
            self.policy in (FP32_RHOK, FP32_STORAGE_FP64)
            or (self.policy == FP32_ADAPTIVE
                and block_index in self._sensitive_blocks))

    def _density_sparse(self, dms, hermi, dtype):
        sparse = dms[:, self.rows, self.cols].copy()
        if hermi == 1:
            sparse *= 2
        else:
            sparse += dms[:, self.cols, self.rows]
        sparse[:, self.diag] *= .5
        return sparse.astype(dtype, copy=False)

    @staticmethod
    def _occupied_coefficients(dm, nset, nao, dtype):
        coeff = getattr(dm, 'mo_coeff', None)
        occupation = getattr(dm, 'mo_occ', None)
        if coeff is None or occupation is None:
            return None
        coeff = cp.asarray(coeff).reshape(nset, nao, -1)
        occupation = cp.asarray(occupation).reshape(nset, -1)
        return [
            (values[:, occ > 0] * cp.sqrt(occ[occ > 0])).astype(
                dtype, copy=False)
            for values, occ in zip(coeff, occupation)]

    def _add_k_from_orbitals(
            self, dense, occupied, accumulator, block_index):
        promote = self._use_fp64_outer(block_index)
        for index, coeff in enumerate(occupied):
            rhok = contract('Lji,jk->Lki', dense, coeff).reshape(
                -1, dense.shape[-1])
            if promote:
                rhok = rhok.astype(cp.float64)
            accumulator[index] += cp.dot(rhok.T, rhok).astype(
                cp.float64, copy=False)

    def _add_k_from_density(self, dense, dms, accumulator, block_index):
        promote = self._use_fp64_outer(block_index)
        right = dense.astype(cp.float64) if promote else dense
        for index, dm in enumerate(dms):
            rhok = contract('Lij,jk->Lki', dense, dm)
            if promote:
                rhok = rhok.astype(cp.float64)
            accumulator[index] += contract(
                'Lki,Lkj->ij', rhok, right).astype(cp.float64)

    def __call__(self, dm, hermi=1, kpts=None, kpts_band=None,
                 with_j=True, with_k=True, omega=None, exxdiv=None):
        '''Return FP64 J/K through GPU4PySCF's provider-level contract.'''
        if not (with_j or with_k):
            raise ValueError('at least one of with_j and with_k is required')
        if kpts_band is not None:
            raise NotImplementedError('mixed source J/K does not support bands')
        if kpts is not None and np.max(np.abs(np.asarray(kpts))) > 1.e-12:
            raise NotImplementedError('mixed source J/K requires Gamma')
        if omega not in (None, 0):
            raise NotImplementedError('mixed source J/K excludes RSH')

        started = time.perf_counter()
        original_shape = dm.shape
        nao = int(original_shape[-1])
        dms = cp.asarray(dm)
        if (nao != int(self.provider.nao) or dms.dtype != cp.float64
                or cp.iscomplexobj(dms)):
            raise TypeError(
                'mixed source J/K requires a real FP64 AO density matching GDF')
        dms = dms.reshape(-1, nao, nao)
        nset = int(dms.shape[0])
        working_dtype = self._working_dtype()
        occupied = (
            self._occupied_coefficients(dm, nset, nao, working_dtype)
            if with_k else None)
        dms_work = (
            dms.astype(working_dtype, copy=False)
            if with_k and occupied is None else None)
        sparse = (
            self._density_sparse(dms, hermi, working_dtype)
            if with_j else None)
        vj_packed = (
            cp.zeros((nset, int(self.storage.shape[1])), dtype=cp.float64)
            if with_j else None)
        vk = cp.zeros((nset, nao, nao), dtype=cp.float64) if with_k else None

        for block_index, (p0, p1) in enumerate(lib.prange(
                0, int(self.storage.shape[0]), self.block_size)):
            packed = cp.asarray(self.storage[p0:p1], dtype=cp.float32)
            self.host_to_device_bytes += int(packed.nbytes)
            self.block_count += 1
            if working_dtype == cp.float64:
                packed = packed.astype(cp.float64)
            if with_j:
                rho = cp.dot(sparse, packed.T)
                vj_packed += cp.dot(rho, packed).astype(cp.float64)
            if with_k:
                dense = self._dense_buffer(working_dtype)[:p1-p0]
                dense[...] = 0.
                dense[:, self.rows, self.cols] = packed
                dense[:, self.cols, self.rows] = packed
                if occupied is None:
                    self._add_k_from_density(
                        dense, dms_work, vk, block_index)
                else:
                    self._add_k_from_orbitals(
                        dense, occupied, vk, block_index)

        vj = None
        if with_j:
            vj = cp.zeros((nset, nao, nao), dtype=cp.float64)
            vj[:, self.cols, self.rows] = vj_packed
            vj[:, self.rows, self.cols] = vj_packed
            vj = vj.reshape(original_shape)
        if with_k:
            if exxdiv == 'ewald':
                _ewald_exxdiv_for_G0(
                    self.provider.cell, np.zeros(3), dms, vk)
            vk = vk.reshape(original_shape)
        self.calls += 1
        self.j_calls += int(with_j)
        self.k_calls += int(with_k)
        self.wall_seconds += time.perf_counter() - started
        return vj, vk

    def profile(self):
        '''Return the explicit precision and transfer ledger.'''
        return {
            'backend': self.backend,
            'policy': self.policy,
            'storage_dtype': 'float32',
            'large_contraction_dtype': (
                'float64 after device-block promotion from FP32 storage'
                if self.policy == FP32_STORAGE_FP64 else 'float32'),
            'adaptive_k_outer_dtype': (
                'all float64' if self.policy == FP32_RHOK else
                'selected float64' if self.policy == FP32_ADAPTIVE else None),
            'between_block_accumulation_dtype': 'float64',
            'output_dtype': 'float64',
            'block_size': self.block_size,
            'adaptive_norm_fraction': self.adaptive_norm_fraction,
            'adaptive_fp64_outer_blocks': len(self._sensitive_blocks),
            'calls': self.calls,
            'j_calls': self.j_calls,
            'k_calls': self.k_calls,
            'auxiliary_block_visits': self.block_count,
            'host_to_device_cderi_bytes': self.host_to_device_bytes,
            'wall_seconds': self.wall_seconds,
        }


class SourcePrecisionController:
    '''Install a monotonic mixed-to-native source-RHF GDF operator.'''

    def __init__(self, mf_hf, storage, *, block_size,
                 energy_normalization, adaptive_norm_fraction, thresholds,
                 persistent=False):
        if int(energy_normalization) < 1:
            raise ValueError('energy_normalization must be positive')
        self.mf_hf = mf_hf
        self.provider = mf_hf.with_df
        self.storage = storage
        self.energy_normalization = int(energy_normalization)
        self.thresholds = thresholds
        self.persistent = bool(persistent)
        self.sequence = (
            (FP32_BLOCK, FP32_RHOK, FP32_STORAGE_FP64)
            if self.persistent else
            (FP32_BLOCK, FP32_ADAPTIVE, NATIVE_FP64))
        self.owners = {
            policy: GammaFP32JK(
                self.provider, storage, block_size=block_size, policy=policy,
                adaptive_norm_fraction=adaptive_norm_fraction)
            for policy in self.sequence if policy != NATIVE_FP64}
        self.current_policy = self.sequence[0]
        self.tier = 0
        self.original_get_jk = self.original_get_veff = None
        self.original_callback = None
        self.pending_full_rebuild = False
        self.calls = []
        self.cycles = []
        self.transitions = []
        self.full_rebuilds = []

    def __enter__(self):
        self.original_get_jk = self.provider.get_jk
        self.original_get_veff = self.mf_hf.get_veff
        self.original_callback = self.mf_hf.callback
        self.provider.get_jk = self.get_jk
        self.mf_hf.get_veff = self.get_veff
        self.mf_hf.callback = self.callback
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.provider.get_jk = self.original_get_jk
        self.mf_hf.get_veff = self.original_get_veff
        self.mf_hf.callback = self.original_callback
        return False

    def get_jk(self, dm, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, exxdiv=None):
        '''Dispatch the current tier from the very first source J/K call.'''
        started = time.perf_counter()
        if self.current_policy == NATIVE_FP64:
            output = self.original_get_jk(
                dm, hermi=hermi, kpts=kpts, kpts_band=kpts_band,
                with_j=with_j, with_k=with_k, omega=omega, exxdiv=exxdiv)
        else:
            output = self.owners[self.current_policy](
                dm, hermi=hermi, kpts=kpts, kpts_band=kpts_band,
                with_j=with_j, with_k=with_k, omega=omega, exxdiv=exxdiv)
        self.calls.append({
            'call': len(self.calls) + 1,
            'policy': self.current_policy,
            'with_j': bool(with_j),
            'with_k': bool(with_k),
            'python_interval_seconds': time.perf_counter() - started,
        })
        return output

    def get_veff(self, mol=None, dm=None, dm_last=None, vhf_last=None,
                 hermi=1):
        '''Discard incremental Fock state immediately after a tier change.'''
        if self.pending_full_rebuild:
            dm_last = vhf_last = None
            self.pending_full_rebuild = False
            self.full_rebuilds.append({
                'before_cycle': len(self.cycles) + 1,
                'policy': self.current_policy,
            })
        return self.original_get_veff(mol, dm, dm_last, vhf_last, hermi)

    @staticmethod
    def _ao_commutator_norm(envs):
        fock = cp.asarray(envs['fock'])
        dm = cp.asarray(envs['dm'])
        overlap = cp.asarray(envs['s1e'])
        return float(cp.linalg.norm(
            fock @ dm @ overlap - overlap @ dm @ fock).get())

    def callback(self, envs):
        '''Record one cycle and move only toward a more accurate operator.'''
        cycle = int(envs['cycle']) + 1
        energy = _scalar(envs['e_tot'])
        previous = _scalar(envs.get('last_hf_e'))
        energy_change = (
            float('inf') if previous is None else abs(energy - previous))
        record = {
            'cycle': cycle,
            'policy_used': self.current_policy,
            'energy_ha': energy,
            'energy_change_ha': energy_change,
            'energy_change_per_unit_ha': (
                energy_change / self.energy_normalization),
            'density_change_norm': _scalar(envs['norm_ddm']),
            'orbital_gradient_norm': _scalar(envs['norm_gorb']),
            'ao_commutator_norm': self._ao_commutator_norm(envs),
        }
        self.cycles.append(record)
        gate = (
            self.thresholds['coarse'] if self.tier == 0 else
            self.thresholds['fine'] if self.tier == 1 else None)
        if gate is not None and (
                cycle >= int(gate['minimum_cycle'])
                and record['energy_change_per_unit_ha']
                <= float(gate['energy_change_per_unit_ha'])
                and record['density_change_norm']
                <= float(gate['density_change_norm'])
                and record['ao_commutator_norm']
                <= float(gate['ao_commutator_norm'])):
            old = self.current_policy
            self.tier += 1
            self.current_policy = self.sequence[self.tier]
            cleared = _clear_diis(envs.get('mf_diis'))
            self.pending_full_rebuild = True
            self.transitions.append({
                'after_cycle': cycle,
                'after_call': len(self.calls),
                'from': old,
                'to': self.current_policy,
                'trigger': record,
                'gate': gate,
                'diis_history_cleared': cleared,
                'next_fock_full_rebuild_required': True,
            })
        if callable(self.original_callback):
            self.original_callback(envs)

    def profile(self):
        final_policy = self.sequence[-1]
        final_transition = next((
            item for item in self.transitions if item['to'] == final_policy),
            None)
        final_after = 0
        if final_transition is not None:
            final_after = sum(
                item['policy'] == final_policy
                for item in self.calls[final_transition['after_call']:])
        return {
            'mode': (
                'persistent_fp32_monotonic' if self.persistent else
                'monotonic_block_adaptive_native'),
            'final_policy': self.current_policy,
            'required_final_policy': final_policy,
            'energy_normalization': self.energy_normalization,
            'calls': self.calls,
            'cycles': self.cycles,
            'transitions': self.transitions,
            'full_rebuilds': self.full_rebuilds,
            'final_calls_after_transition': int(final_after),
            'native_calls_after_transition': (
                int(final_after) if final_policy == NATIVE_FP64 else 0),
            'owners': {
                name: owner.profile() for name, owner in self.owners.items()},
        }

    def force_final(self, *, reason):
        '''Move to the final arithmetic tier before a cleanup SCF pass.'''
        final_policy = self.sequence[-1]
        if self.current_policy == final_policy:
            return False
        old = self.current_policy
        self.tier = len(self.sequence) - 1
        self.current_policy = final_policy
        self.pending_full_rebuild = True
        self.transitions.append({
            'after_cycle': len(self.cycles),
            'after_call': len(self.calls),
            'from': old,
            'to': final_policy,
            'trigger': {'reason': str(reason)},
            'gate': None,
            'diis_history_cleared': False,
            'next_fock_full_rebuild_required': True,
            'forced_by_cleanup': True,
        })
        return True


def _final_fp64_audit(mf_hf, previous_energy, definition=None):
    '''Rebuild one full Fock and measure source stationarity.'''
    started = time.perf_counter()
    dm = mf_hf.make_rdm1()
    hcore = mf_hf.get_hcore()
    vhf = mf_hf.get_veff(mf_hf.cell, dm, None, None)
    fock = hcore + vhf
    overlap = mf_hf.get_ovlp()
    commutator = fock @ dm @ overlap - overlap @ dm @ fock
    gradient = mf_hf.get_grad(mf_hf.mo_coeff, mf_hf.mo_occ, fock)
    energy = _scalar(mf_hf.energy_tot(dm=dm, h1e=hcore, vhf=vhf))
    return {
        'definition': (
            'one full native-FP64 GDF Fock at accepted density'
            if definition is None else str(definition)),
        'energy_ha': energy,
        'energy_change_ha': abs(energy - float(previous_energy)),
        'ao_commutator_max_abs': _maximum(commutator),
        'orbital_gradient_max_abs': _maximum(gradient),
        'wall_seconds': time.perf_counter() - started,
    }


def run_mixed_source_scf(
        mf_hf, dm0=None, *, block_size=256, copy_block_size=256,
        energy_normalization=1, adaptive_norm_fraction=.9,
        thresholds=None, final_energy_tol_per_unit=1.e-7,
        final_residual_tol=1.e-5, fallback=True, kernel_kwargs=None,
        persistent=False):
    '''Run one guarded mixed source RHF and return ``(energy, report)``.

    With ``persistent=False`` the established path ends at the native FP64
    provider and may fall back to it.  With ``persistent=True`` the source uses
    the provider-owned FP32 factor through all tiers and ends with FP64
    arithmetic over FP32 storage; no FP64-CDERI oracle exists without a fresh
    rebuild.  A provider build occurs inside this call when needed.
    '''
    provider = mf_hf.with_df
    if provider.__class__.__name__ != 'GDF' or not getattr(
            provider, 'is_gamma_point', False):
        raise NotImplementedError(
            'source mixed precision currently supports real-Gamma GDF RHF')
    if float(final_energy_tol_per_unit) <= 0. or float(final_residual_tol) <= 0.:
        raise ValueError('final FP64 audit tolerances must be positive')
    thresholds = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    kernel_kwargs = {} if kernel_kwargs is None else dict(kernel_kwargs)
    started = time.perf_counter()
    report = {
        'requested_policy': (
            'persistent_mixed' if persistent else 'mixed'),
        'effective_policy': None,
        'fallback': None,
        'storage': None,
        'schedule': None,
        'final_fp64_audit': None,
        'native_cleanup_seconds': 0.,
    }
    # Auto storage must account for both the persistent FP64 factor and the
    # persistent FP32 sidecar before the native builder allocates either.
    if (not persistent
            and str(provider.cderi_storage).lower() == 'auto'):
        provider.cderi_auto_sidecar_factor = max(
            .5, float(provider.cderi_auto_sidecar_factor))

    storage = controller = None
    try:
        if provider._cderi is None:
            provider.build(j_only=False)
        storage, storage_report = copy_gamma_cderi_fp32(
            provider, block_size=copy_block_size)
        report['storage'] = storage_report
        controller = SourcePrecisionController(
            mf_hf, storage, block_size=block_size,
            energy_normalization=energy_normalization,
            adaptive_norm_fraction=adaptive_norm_fraction,
            thresholds=thresholds, persistent=persistent)
        with controller:
            energy = mf_hf.kernel(dm0=dm0, **kernel_kwargs)
            if persistent:
                interim = controller.profile()
                needs_final = (
                    not bool(mf_hf.converged)
                    or interim['final_policy']
                    != interim['required_final_policy']
                    or interim['final_calls_after_transition'] < 1)
                if needs_final:
                    controller.force_final(
                        reason='mandatory persistent-FP32 cleanup tier')
                    cleanup_dm = (
                        mf_hf.make_rdm1()
                        if mf_hf.mo_coeff is not None else dm0)
                    mf_hf.converged = False
                    cleanup_start = time.perf_counter()
                    energy = mf_hf.kernel(
                        dm0=cleanup_dm, **kernel_kwargs)
                    report['native_cleanup_seconds'] = (
                        time.perf_counter() - cleanup_start)
        report['schedule'] = controller.profile()
    except (MemoryError, cp.cuda.memory.OutOfMemoryError) as error:
        if not fallback or persistent:
            raise
        report['fallback'] = {
            'reason': f'{type(error).__name__}: {error}',
            'stage': 'provider_build_or_fp32_sidecar',
        }
        energy = mf_hf.kernel(dm0=dm0, **kernel_kwargs)

    schedule = report['schedule']
    required_final_policy = (
        FP32_STORAGE_FP64 if persistent else NATIVE_FP64)
    needs_cleanup = (
        not bool(mf_hf.converged)
        or schedule is None
        or schedule['final_policy'] != required_final_policy
        or schedule['final_calls_after_transition'] < 1)
    if not needs_cleanup:
        audit = _final_fp64_audit(
            mf_hf, energy,
            definition=(
                'FP64 arithmetic over persistent FP32 CDERI at accepted '
                'density' if persistent else None))
        report['final_fp64_audit'] = audit
        needs_cleanup = (
            audit['energy_change_ha'] / int(energy_normalization)
            > float(final_energy_tol_per_unit)
            or audit['ao_commutator_max_abs'] > float(final_residual_tol)
            or audit['orbital_gradient_max_abs'] > float(final_residual_tol))

    if needs_cleanup:
        if persistent:
            raise RuntimeError(
                'persistent-FP32 source did not satisfy its mandatory final '
                'FP64-arithmetic gate; rebuild with gdf_precision_profile='
                "'fp64' for a true FP64 fallback")
        if not fallback:
            raise RuntimeError(
                'mixed source did not satisfy the final native-FP64 gate')
        if report['fallback'] is None:
            report['fallback'] = {
                'reason': 'mixed source did not satisfy the final FP64 gate',
                'stage': 'final_fp64_audit',
            }
        cleanup_dm = (
            mf_hf.make_rdm1() if mf_hf.mo_coeff is not None else dm0)
        cleanup_start = time.perf_counter()
        mf_hf.converged = False
        energy = mf_hf.kernel(dm0=cleanup_dm, **kernel_kwargs)
        report['native_cleanup_seconds'] = time.perf_counter() - cleanup_start
        if not mf_hf.converged:
            raise RuntimeError('native FP64 source cleanup did not converge')
        audit = _final_fp64_audit(mf_hf, energy)
        report['final_fp64_audit'] = audit
        if (audit['energy_change_ha'] / int(energy_normalization)
                > float(final_energy_tol_per_unit)
                or audit['ao_commutator_max_abs'] > float(final_residual_tol)
                or audit['orbital_gradient_max_abs'] > float(final_residual_tol)):
            raise RuntimeError('native FP64 source cleanup failed its audit')
        report['effective_policy'] = 'native_fp64_fallback'
    else:
        report['effective_policy'] = (
            'persistent_fp32_mixed' if persistent else
            'mixed_to_native_fp64')

    mf_hf.e_tot = report['final_fp64_audit']['energy_ha']
    report['wall_seconds'] = time.perf_counter() - started
    report['accepted_energy_ha'] = _scalar(mf_hf.e_tot)
    # The mapped sidecar is intentionally not attached to the provider.  Its
    # lifetime ends here, before target XC and analytic nuclear response.
    storage = controller = None
    return _scalar(mf_hf.e_tot), report


__all__ = [
    'DEFAULT_THRESHOLDS',
    'GammaFP32JK',
    'SourcePrecisionController',
    'run_mixed_source_scf',
]
