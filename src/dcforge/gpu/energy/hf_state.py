'''Validated molecular GPU RHF/UHF source-state restoration.'''

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import cupy as cp

from dcforge.hf_state import (
    HFState,
    HFStateInvalidatedError,
    HFStateMismatchError,
    HFStateValidationError,
    canonical_sha256,
    load_hf_state,
    molecule_fingerprint,
    representation_identity,
    source_state_fingerprint,
)


@dataclass(frozen=True)
class HFSourceValidation:
    '''Immutable evidence for one active molecular GPU source state.'''

    origin: str
    capability: str
    reference: str
    molecule_fingerprint: str
    representation_fingerprint: str
    state_fingerprint: str
    source_energy_ha: float
    checkpoint_sha256: str | None
    orthogonality_max_abs: float | None
    electron_number_error: float | None
    density_max_abs_error: float | None
    source_energy_error_ha: float | None
    fock_vo_max_abs: float | None
    generalized_eigen_residual_relative: float | None

    def to_dict(self):
        '''Return JSON-serializable validation evidence.'''
        return asdict(self)


class GPUHFStateMixin:
    '''Public save/restore state machine shared by GPU RHFDFT and UHFDFT.'''

    _hf_state_reference = None

    @classmethod
    def from_hf_state(
            cls, path, *, xc='PBE', validation='gradient',
            with_df=None, device=None):
        '''Construct the final GPU representation and restore its HF state.'''
        if validation != 'gradient':
            raise ValueError("validation must be 'gradient'")
        state = load_hf_state(path)
        if state.reference != cls._hf_state_reference:
            raise HFStateMismatchError(
                f'{cls.__name__} requires {cls._hf_state_reference}, '
                f'checkpoint contains {state.reference}')

        def build():
            workflow = cls(state.mol, xc=xc)
            representation = state.metadata['representation']
            kind = representation.get('kind')
            if kind == 'density_fitted':
                workflow = workflow.density_fit(
                    auxbasis=representation.get('auxbasis'),
                    with_df=with_df,
                    only_dfj=False,
                )
            elif kind == 'conventional':
                if with_df is not None:
                    raise HFStateMismatchError(
                        'with_df cannot be supplied for a conventional state')
            else:
                raise HFStateMismatchError(
                    f'unsupported source representation: {kind!r}')
            workflow.restore_hf_state(state, validation=validation)
            return workflow

        if device is None:
            return build()
        with cp.cuda.Device(int(device)):
            return build()

    def restore_hf_state(self, state_or_path, *, validation='gradient'):
        '''Restore and independently validate one canonical GPU HF state.'''
        if validation != 'gradient':
            raise ValueError("validation must be 'gradient'")
        state = (
            state_or_path if isinstance(state_or_path, HFState)
            else load_hf_state(state_or_path)
        )
        reference = self._hf_state_reference
        if state.reference != reference:
            raise HFStateMismatchError(
                f'{type(self).__name__} requires {reference}, checkpoint '
                f'contains {state.reference}')
        if molecule_fingerprint(self.mol) != state.metadata[
                'molecule_fingerprint']:
            raise HFStateMismatchError(
                'checkpoint molecule does not match the active GPU workflow')
        _check_runtime_provider_contract(self)
        _check_representation(self.mf_hf, state.metadata['representation'])

        source = self.mf_hf
        names = ('mo_energy', 'mo_coeff', 'mo_occ', 'e_tot',
                 'converged', 'cycles')
        missing = object()
        previous = {
            name: getattr(source, name, missing)
            for name in names
        }
        old_receipt = getattr(self, '_source_state_receipt', None)
        old_source_converged = bool(getattr(self, '_source_converged', False))
        try:
            source.mo_energy = cp.asarray(state.mo_energy)
            source.mo_coeff = cp.asarray(state.mo_coeff)
            source.mo_occ = cp.asarray(state.mo_occ)
            source.e_tot = float(state.e_tot)
            source.converged = False
            cycles = state.metadata['convergence_claim'].get('cycles')
            if cycles is not None:
                source.cycles = int(cycles)

            metrics = _validate_active_gpu_state(self, state)
            current_fingerprint = _current_state_fingerprint(self)
            if current_fingerprint != state.metadata['state_fingerprint']:
                raise HFStateValidationError(
                    'GPU materialization changed the canonical source arrays')
            active_representation = representation_identity(source)
            representation_fingerprint = canonical_sha256(
                active_representation)
            source.converged = True
            self._source_converged = True
            self._source_state_receipt = HFSourceValidation(
                origin='restored', capability='gradient_ready',
                reference=reference,
                molecule_fingerprint=molecule_fingerprint(self.mol),
                representation_fingerprint=representation_fingerprint,
                state_fingerprint=current_fingerprint,
                source_energy_ha=float(state.e_tot),
                checkpoint_sha256=state.checkpoint_sha256,
                **metrics,
            )
            self._sync_hf_results()
            return self._source_state_receipt
        except Exception:
            for name, value in previous.items():
                if value is missing:
                    source.__dict__.pop(name, None)
                else:
                    setattr(source, name, value)
            self._source_state_receipt = old_receipt
            self._source_converged = old_source_converged
            raise

    @property
    def source_validation(self):
        '''Return the current immutable source-state receipt or ``None``.'''
        return getattr(self, '_source_state_receipt', None)

    def has_valid_source_state(self, capability='gradient_ready'):
        '''Return whether the current arrays still match their receipt.'''
        try:
            self._assert_valid_source_state(capability=capability)
        except (HFStateInvalidatedError, HFStateValidationError):
            return False
        return True

    def _assert_valid_source_state(self, capability='gradient_ready'):
        if capability != 'gradient_ready':
            raise ValueError("capability must be 'gradient_ready'")
        receipt = getattr(self, '_source_state_receipt', None)
        if (receipt is None or not self._source_converged
                or not self.mf_hf.converged):
            raise HFStateValidationError(
                'a validated current GPU source state is required')
        try:
            _check_runtime_provider_contract(self)
        except HFStateMismatchError as error:
            self._invalidate_source_state()
            raise HFStateInvalidatedError(
                'GPU source/target provider contract changed after validation'
            ) from error
        current = (
            molecule_fingerprint(self.mol),
            canonical_sha256(representation_identity(self.mf_hf)),
            _current_state_fingerprint(self),
        )
        expected = (
            receipt.molecule_fingerprint,
            receipt.representation_fingerprint,
            receipt.state_fingerprint,
        )
        energy = float(self.mf_hf.e_tot)
        if current != expected or energy != receipt.source_energy_ha:
            self._invalidate_source_state()
            raise HFStateInvalidatedError(
                'GPU source molecule, representation, or canonical arrays '
                'changed after validation')
        return receipt

    def _record_native_source_state(self):
        '''Record a source produced by the active GPU SCF kernel.'''
        _check_runtime_provider_contract(self)
        self._source_state_receipt = HFSourceValidation(
            origin='native_scf', capability='gradient_ready',
            reference=self._hf_state_reference,
            molecule_fingerprint=molecule_fingerprint(self.mol),
            representation_fingerprint=canonical_sha256(
                representation_identity(self.mf_hf)),
            state_fingerprint=_current_state_fingerprint(self),
            source_energy_ha=float(self.mf_hf.e_tot),
            checkpoint_sha256=None,
            orthogonality_max_abs=None,
            electron_number_error=None,
            density_max_abs_error=None,
            source_energy_error_ha=None,
            fock_vo_max_abs=None,
            generalized_eigen_residual_relative=None,
        )
        return self._source_state_receipt

    def _invalidate_source_state(self):
        self._source_converged = False
        self._source_state_receipt = None


def _validate_active_gpu_state(workflow, state):
    source = workflow.mf_hf
    reference = state.reference
    coeff = cp.asarray(source.mo_coeff)
    energy = cp.asarray(source.mo_energy)
    occ = cp.asarray(source.mo_occ)
    saved_dm = cp.asarray(state.dm)
    overlap = cp.asarray(source.get_ovlp())
    dm = cp.asarray(source.make_rdm1(coeff, occ))

    density_error = _max_abs(dm - saved_dm)
    orthogonality = 0.0
    electron_error = 0.0
    if reference == 'RHF':
        identity = cp.eye(coeff.shape[1], dtype=coeff.dtype)
        orthogonality = _max_abs(
            coeff.conj().T @ overlap @ coeff - identity)
        observed = _real_scalar(cp.einsum('ij,ji->', dm, overlap))
        electron_error = abs(observed - float(source.mol.nelectron))
    else:
        for spin in range(2):
            coeff_s = coeff[spin]
            identity = cp.eye(coeff_s.shape[1], dtype=coeff_s.dtype)
            orthogonality = max(
                orthogonality,
                _max_abs(coeff_s.conj().T @ overlap @ coeff_s - identity),
            )
            observed = _real_scalar(cp.einsum(
                'ij,ji->', dm[spin], overlap))
            electron_error = max(
                electron_error,
                abs(observed - float(source.nelec[spin])),
            )

    hcore = source.get_hcore()
    vhf = source.get_veff(source.mol, dm)
    fock = source.get_fock(
        h1e=hcore, s1e=overlap, vhf=vhf, dm=dm)
    active_energy = float(source.energy_tot(dm=dm, h1e=hcore, vhf=vhf))
    energy_error = abs(active_energy - float(state.e_tot))

    fvo_max = 0.0
    eigen_relative = 0.0
    if reference == 'RHF':
        fvo_max, eigen_relative = _orbital_residuals(
            fock, overlap, coeff, energy, occ)
    else:
        for spin in range(2):
            fvo, eigen = _orbital_residuals(
                fock[spin], overlap, coeff[spin], energy[spin], occ[spin])
            fvo_max = max(fvo_max, fvo)
            eigen_relative = max(eigen_relative, eigen)

    conv_tol_grad = state.metadata['convergence_claim'].get('conv_tol_grad')
    fvo_tolerance = max(
        1e-7,
        10 * (0.0 if conv_tol_grad is None else float(conv_tol_grad)),
    )
    checks = {
        'orthogonality': (orthogonality, 1e-10),
        'electron number': (electron_error, 1e-9),
        'stored density': (density_error, 1e-10),
        'source energy': (energy_error, 1e-9),
        'source Fock VO': (fvo_max, fvo_tolerance),
        'generalized eigen residual': (eigen_relative, 1e-8),
    }
    failures = [
        f'{name}={value:.3e}>{tolerance:.3e}'
        for name, (value, tolerance) in checks.items()
        if not math.isfinite(value) or value > tolerance
    ]
    if failures:
        raise HFStateValidationError(
            'restored GPU HF state failed active validation: '
            + ', '.join(failures))

    _check_representation(source, state.metadata['representation'])
    return {
        'orthogonality_max_abs': orthogonality,
        'electron_number_error': electron_error,
        'density_max_abs_error': density_error,
        'source_energy_error_ha': energy_error,
        'fock_vo_max_abs': fvo_max,
        'generalized_eigen_residual_relative': eigen_relative,
    }


def _orbital_residuals(fock, overlap, coeff, energy, occ):
    fock_coeff = fock @ coeff
    residual = fock_coeff - overlap @ coeff * energy[None, :]
    denominator = max(_norm(fock_coeff), 1.0)
    relative = _norm(residual) / denominator
    occupied = occ > 0
    virtual = occ == 0
    fvo = coeff[:, virtual].conj().T @ fock @ coeff[:, occupied]
    return _max_abs(fvo), relative


def _check_runtime_provider_contract(workflow):
    target_df = hasattr(workflow, 'with_df')
    source_df = hasattr(workflow.mf_hf, 'with_df')
    if target_df != source_df:
        raise HFStateMismatchError(
            'source and target GPU representations differ')
    if target_df:
        if workflow.with_df is not workflow.mf_hf.with_df:
            raise HFStateMismatchError(
                'source and target must share one GPU DF provider')
        provider_mol = getattr(workflow.with_df, 'mol', None)
        if (provider_mol is not None
                and molecule_fingerprint(provider_mol)
                != molecule_fingerprint(workflow.mol)):
            raise HFStateMismatchError(
                'GPU DF provider molecule does not match the workflow')
        if bool(workflow.only_dfj) or bool(workflow.mf_hf.only_dfj):
            raise HFStateMismatchError(
                'gradient-ready GPU DF source requires only_dfj=False')


def _check_representation(source, expected):
    observed = representation_identity(source)
    if observed.get('kind') != expected.get('kind'):
        raise HFStateMismatchError(
            'checkpoint and active source representations differ')
    if observed.get('kind') == 'density_fitted':
        if observed.get('only_dfj') or expected.get('only_dfj'):
            raise HFStateMismatchError('GPU DF restore requires only_dfj=False')
        if canonical_sha256(observed.get('auxbasis')) != canonical_sha256(
                expected.get('auxbasis')):
            raise HFStateMismatchError(
                'checkpoint and active auxiliary basis specifications differ')
        expected_basis = expected.get('auxbasis_fingerprint')
        observed_basis = observed.get('auxbasis_fingerprint')
        if (expected_basis is not None and observed_basis is not None
                and expected_basis != observed_basis):
            raise HFStateMismatchError(
                'checkpoint and active resolved auxiliary bases differ')


def _current_state_fingerprint(workflow):
    source = workflow.mf_hf
    return source_state_fingerprint(
        workflow.mol, workflow._hf_state_reference,
        cp.asnumpy(cp.asarray(source.mo_energy)),
        cp.asnumpy(cp.asarray(source.mo_coeff)),
        cp.asnumpy(cp.asarray(source.mo_occ)),
    )


def _max_abs(value):
    value = cp.asarray(value)
    return float(cp.max(cp.abs(value)).get()) if value.size else 0.0


def _norm(value):
    return float(cp.linalg.norm(cp.asarray(value)).get())


def _real_scalar(value):
    value = cp.asarray(value)
    return float(value.real.get())


__all__ = ['GPUHFStateMixin', 'HFSourceValidation']
