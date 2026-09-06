'''Versioned, gradient-ready periodic RHF/KRHF source checkpoints.

PySCF owns the standard ``cell`` and ``scf`` checkpoint groups.  DC-Forge
adds only the state needed to prove that those canonical orbitals belong to
the same periodic Hamiltonian used by a later HF-DFT gradient.  CDERI is not
stored: it is process-local provider state and is rebuilt after restoration.
'''

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from pyscf.lib import chkfile as lib_chkfile
from pyscf.pbc.scf import chkfile as pbc_chkfile

from dcforge.cpu.pbc.kpts import validate_insulating_occupations

_GROUP = 'dcforge_pbc_hf_state'
_SCHEMA_VERSION = 1


class PBCStateError(RuntimeError):
    '''A periodic source checkpoint is invalid or incompatible.'''


@dataclass(frozen=True)
class PBCHFState:
    '''Host-owned canonical source arrays and checkpoint metadata.'''

    path: str
    cell: object
    mo_energy: np.ndarray
    mo_coeff: np.ndarray
    mo_occ: np.ndarray
    dm: np.ndarray
    e_tot: float
    metadata: dict
    checkpoint_sha256: str


@dataclass(frozen=True)
class PBCSourceReceipt:
    '''Immutable evidence attached to one live periodic source state.'''

    origin: str
    capability: str
    source_fingerprint: str
    checkpoint_sha256: str | None = None
    orthogonality_max_abs: float | None = None
    electron_number_error: float | None = None
    density_max_abs_error: float | None = None
    source_energy_error_ha: float | None = None
    fock_vo_max_abs: float | None = None
    generalized_eigen_residual_relative: float | None = None

    def to_dict(self):
        '''Return JSON-compatible validation evidence.'''
        return asdict(self)


def save_pbc_hf_state(path, mf, *, extra_provenance=None):
    '''Atomically save one validated DC-Forge periodic source state.

    Args:
        path : str or pathlib.Path
            Destination PySCF-compatible HDF5 checkpoint.
        mf : RHFDFT or KRHFDFT
            CPU or GPU periodic workflow with a current source-state receipt.

    Kwargs:
        extra_provenance : dict or None
            Optional JSON-compatible caller metadata.  It is descriptive and
            never substitutes for restore-time physical validation.

    Returns:
        dict
            The complete DC-Forge manifest stored in the checkpoint.
    '''
    receipt = assert_pbc_source_state(mf)
    source = mf.mf_hf
    mo_energy = _host(source.mo_energy)
    mo_coeff = _host(source.mo_coeff)
    mo_occ = _host(source.mo_occ)
    dm = _host(source.make_rdm1(mo_coeff, mo_occ))
    sampling = sampling_identity(mf)
    _validate_array_contract(
        mf.cell, sampling['kind'], mo_energy, mo_coeff, mo_occ, dm)
    e_tot = float(source.e_tot)
    if not np.isfinite(e_tot):
        raise PBCStateError('source HF energy is not finite')

    cell_data = cell_identity(mf.cell)
    provider_data = provider_identity(mf.with_df)
    arrays = {
        name: _array_manifest(value)
        for name, value in {
            'mo_energy': mo_energy,
            'mo_coeff': mo_coeff,
            'mo_occ': mo_occ,
            'dm': dm,
        }.items()
    }
    metadata = {
        'schema_version': _SCHEMA_VERSION,
        'complete': True,
        'capability': 'gradient_ready',
        'cell_identity': cell_data,
        'cell_fingerprint': canonical_sha256(cell_data),
        'sampling': sampling,
        'provider': provider_data,
        'source_fingerprint': receipt.source_fingerprint,
        'source_energy_ha': e_tot,
        'convergence_claim': {
            'converged': True,
            'conv_tol': _optional_float(getattr(source, 'conv_tol', None)),
            'conv_tol_grad': _optional_float(
                getattr(source, 'conv_tol_grad', None)),
            'cycles': _optional_int(getattr(source, 'cycles', None)),
        },
        'arrays': arrays,
        'extra_provenance': _canonicalize(
            {} if extra_provenance is None else extra_provenance),
    }

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.tmp-{os.getpid()}')
    try:
        temporary.unlink()
    except FileNotFoundError:
        pass
    try:
        pbc_chkfile.dump_scf(
            mf.cell, str(temporary), e_tot,
            mo_energy, mo_coeff, mo_occ, overwrite_mol=True)
        lib_chkfile.save(str(temporary), _GROUP, {
            'metadata_json': json.dumps(
                metadata, sort_keys=True, separators=(',', ':'),
                allow_nan=False),
            'dm': dm,
        })
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return metadata


def load_pbc_hf_state(path):
    '''Load and checksum one DC-Forge periodic source checkpoint.'''
    path = Path(path).expanduser().resolve()
    try:
        cell, scf_data = pbc_chkfile.load_scf(str(path))
        extension = lib_chkfile.load(str(path), _GROUP)
        raw_metadata = extension['metadata_json']
        if isinstance(raw_metadata, bytes):
            raw_metadata = raw_metadata.decode()
        metadata = json.loads(raw_metadata)
        dm = np.asarray(extension['dm'])
        mo_energy = np.asarray(scf_data['mo_energy'])
        mo_coeff = np.asarray(scf_data['mo_coeff'])
        mo_occ = np.asarray(scf_data['mo_occ'])
        e_tot = float(scf_data['e_tot'])
    except PBCStateError:
        raise
    except Exception as error:
        raise PBCStateError(
            f'cannot read DC-Forge PBC source state from {path}') from error

    if metadata.get('schema_version') != _SCHEMA_VERSION:
        raise PBCStateError(
            'unsupported PBC source-state schema: '
            f'{metadata.get("schema_version")!r}')
    if (not metadata.get('complete', False)
            or metadata.get('capability') != 'gradient_ready'):
        raise PBCStateError('PBC source-state checkpoint is incomplete')
    observed_cell = cell_identity(cell)
    if (metadata.get('cell_identity') != observed_cell
            or metadata.get('cell_fingerprint')
            != canonical_sha256(observed_cell)):
        raise PBCStateError('checkpoint Cell fingerprint mismatch')
    if float(metadata.get('source_energy_ha')) != e_tot:
        raise PBCStateError('standard and DC-Forge source energies differ')

    sampling = metadata.get('sampling')
    provider = metadata.get('provider')
    if not isinstance(sampling, dict) or not isinstance(provider, dict):
        raise PBCStateError('sampling/provider metadata are missing')
    _validate_array_contract(
        cell, sampling.get('kind'), mo_energy, mo_coeff, mo_occ, dm)
    values = {
        'mo_energy': mo_energy,
        'mo_coeff': mo_coeff,
        'mo_occ': mo_occ,
        'dm': dm,
    }
    expected_arrays = metadata.get('arrays', {})
    for name, value in values.items():
        if _array_manifest(value) != expected_arrays.get(name):
            raise PBCStateError(f'checkpoint array mismatch: {name}')

    fingerprint = source_fingerprint(
        observed_cell, sampling, provider,
        mo_energy, mo_coeff, mo_occ, e_tot)
    if fingerprint != metadata.get('source_fingerprint'):
        raise PBCStateError('checkpoint source-state fingerprint mismatch')
    for value in values.values():
        value.setflags(write=False)
    return PBCHFState(
        path=str(path), cell=cell,
        mo_energy=mo_energy, mo_coeff=mo_coeff, mo_occ=mo_occ,
        dm=dm, e_tot=e_tot, metadata=metadata,
        checkpoint_sha256=_sha256_file(path),
    )


def configure_pbc_provider(mf, state):
    '''Reconstruct the saved FFTDF/GDF representation on one workflow.'''
    expected = state.metadata['provider']
    exxdiv = state.metadata['sampling'].get('exxdiv')
    if exxdiv is not None:
        mf.exxdiv = mf.mf_hf.exxdiv = exxdiv
    device = getattr(mf.with_df, 'device', 'cpu')
    if expected.get('device') != device:
        raise PBCStateError(
            'checkpoint and active workflow use different CPU/GPU backends')
    kind = expected.get('kind')
    if kind == 'fftdf':
        mf = mf.fft_density_fit(mesh=expected.get('mesh'))
    elif kind == 'gdf':
        mf = mf.density_fit(auxbasis=expected.get('auxbasis'))
        if hasattr(mf, 'gdf_precision_profile'):
            mf.gdf_precision_profile = expected.get(
                'gdf_precision_profile') or 'fp64'
        provider = mf.with_df
        for name in (
                'mesh', 'omega', 'metric_mesh', 'linear_dep_threshold',
                'exp_to_discard', 'j2c_eig_always',
                'gdf_precision_profile'):
            value = expected.get(name)
            if value is not None and hasattr(provider, name):
                if name in ('mesh', 'metric_mesh'):
                    value = np.asarray(value, dtype=int)
                setattr(provider, name, value)
    else:
        raise PBCStateError(f'unsupported PBC DF provider: {kind!r}')
    return mf


def restore_pbc_source_state(mf, state_or_path):
    '''Restore and independently validate a canonical periodic source state.'''
    state = (
        state_or_path if isinstance(state_or_path, PBCHFState)
        else load_pbc_hf_state(state_or_path)
    )
    expected_sampling = state.metadata['sampling']
    if canonical_sha256(cell_identity(mf.cell)) != state.metadata[
            'cell_fingerprint']:
        raise PBCStateError('checkpoint Cell does not match the active workflow')
    if sampling_identity(mf) != expected_sampling:
        raise PBCStateError(
            'checkpoint k-points, weights or sampling mode do not match')
    _check_provider_configuration(mf.with_df, state.metadata['provider'])

    source = mf.mf_hf
    names = ('mo_energy', 'mo_coeff', 'mo_occ', 'e_tot', 'converged', 'cycles')
    missing = object()
    previous = {name: getattr(source, name, missing) for name in names}
    previous_receipt = getattr(mf, '_source_state_receipt', None)
    previous_source_converged = getattr(mf, '_source_converged', None)
    try:
        source.mo_energy = _backend_array(source, state.mo_energy)
        source.mo_coeff = _backend_array(source, state.mo_coeff)
        source.mo_occ = _backend_array(source, state.mo_occ)
        source.e_tot = float(state.e_tot)
        source.converged = False
        cycles = state.metadata['convergence_claim'].get('cycles')
        if cycles is not None:
            source.cycles = int(cycles)

        # Recreate process-local SCF helpers without running SCF.  In
        # particular, GPU4PySCF attaches the FFTDF MultiGrid J/Hcore helper
        # in SCF.build(); it is deliberately absent from the portable
        # checkpoint but required by the corresponding analytic derivative.
        source.build()

        metrics = _validate_active_state(mf, state)
        source.converged = True
        if hasattr(mf, '_source_converged'):
            mf._source_converged = True
        mf._sync_hf_results()
        fingerprint = current_source_fingerprint(mf)
        if fingerprint != state.metadata['source_fingerprint']:
            raise PBCStateError(
                'active backend changed the restored source-state identity')
        mf._source_state_receipt = PBCSourceReceipt(
            origin='restored', capability='gradient_ready',
            source_fingerprint=fingerprint,
            checkpoint_sha256=state.checkpoint_sha256,
            **metrics,
        )
        return mf._source_state_receipt
    except Exception:
        for name, value in previous.items():
            if value is missing:
                source.__dict__.pop(name, None)
            else:
                setattr(source, name, value)
        mf._source_state_receipt = previous_receipt
        if previous_source_converged is not None:
            mf._source_converged = previous_source_converged
        raise


def record_pbc_source_state(mf):
    '''Record source orbitals produced by the active periodic SCF kernel.'''
    if not bool(getattr(mf.mf_hf, 'converged', False)):
        raise PBCStateError('periodic source HF must be converged')
    fingerprint = current_source_fingerprint(mf)
    mf._source_state_receipt = PBCSourceReceipt(
        origin='native_scf', capability='gradient_ready',
        source_fingerprint=fingerprint,
    )
    return mf._source_state_receipt


def assert_pbc_source_state(mf):
    '''Require the live Cell, provider and canonical arrays to match receipt.'''
    receipt = getattr(mf, '_source_state_receipt', None)
    if receipt is None or not bool(getattr(mf.mf_hf, 'converged', False)):
        raise PBCStateError(
            'a validated periodic source state is required; run kernel_hf() '
            'or restore_hf_state() first')
    if current_source_fingerprint(mf) != receipt.source_fingerprint:
        clear_pbc_source_state(mf)
        raise PBCStateError(
            'periodic Cell, provider or canonical source state changed after '
            'validation; rerun kernel_hf() or restore the checkpoint again')
    return receipt


def clear_pbc_source_state(mf):
    '''Remove reusable-source authority without changing target settings.'''
    mf._source_state_receipt = None
    if hasattr(mf, '_source_converged'):
        mf._source_converged = False


def current_source_fingerprint(mf):
    '''Return the complete identity of one live periodic source state.'''
    source = mf.mf_hf
    values = tuple(
        getattr(source, name, None)
        for name in ('mo_energy', 'mo_coeff', 'mo_occ')
    )
    if any(value is None for value in values):
        raise PBCStateError('periodic source has no canonical orbital state')
    e_tot = float(source.e_tot)
    return source_fingerprint(
        cell_identity(mf.cell), sampling_identity(mf),
        provider_identity(mf.with_df),
        *map(_host, values), e_tot,
    )


def cell_identity(cell):
    '''Return the physical periodic Cell and AO-ordering identity.'''
    return _canonicalize({
        'atom_symbols': [cell.atom_symbol(i) for i in range(cell.natm)],
        'atom_charges': np.asarray(cell.atom_charges()),
        'atom_coords_bohr': np.asarray(cell.atom_coords()),
        'lattice_vectors_bohr': np.asarray(cell.lattice_vectors()),
        'dimension': int(cell.dimension),
        'low_dim_ft_type': getattr(cell, 'low_dim_ft_type', None),
        'charge': int(cell.charge),
        'spin': int(cell.spin),
        'nelectron': int(cell.nelectron),
        'cart': bool(cell.cart),
        'basis': getattr(cell, '_basis', {}),
        'pseudo': getattr(cell, '_pseudo', {}),
        'ecp': getattr(cell, '_ecp', {}),
        'precision': float(cell.precision),
        'ke_cutoff': _optional_float(cell.ke_cutoff),
        'mesh': np.asarray(cell.mesh, dtype=int),
        'nao': int(cell.nao_nr()),
        'ao_labels': list(map(str, cell.ao_labels())),
    })


def sampling_identity(mf):
    '''Return Gamma or explicit full-BZ sampling identity.'''
    sampled = hasattr(mf, 'kpts_weights')
    if not sampled:
        kpts = np.zeros((1, 3), dtype=float)
        weights = np.ones(1, dtype=float)
        wrap = None
        kind = 'gamma'
    else:
        kpts = np.asarray(mf.kpts, dtype=float)
        weights = np.asarray(mf.kpts_weights, dtype=float)
        wrap = getattr(mf, 'kpts_wrap_around', None)
        kind = 'sampled_k'
    return _canonicalize({
        'kind': kind,
        'kpts_bohr_inverse': kpts,
        'weights': weights,
        'wrap_around': wrap,
        'exxdiv': getattr(mf.mf_hf, 'exxdiv', None),
    })


def provider_identity(provider):
    '''Return Hamiltonian-defining FFTDF/GDF provider settings.'''
    kind = str(getattr(provider, 'backend_name', '')).lower()
    if kind not in ('fftdf', 'gdf'):
        name = type(provider).__name__.lower()
        kind = 'fftdf' if 'fft' in name else 'gdf' if 'gdf' in name else name
    auxcell = getattr(provider, 'auxcell', None)
    auxbasis_data = None if auxcell is None else getattr(auxcell, '_basis', None)
    return _canonicalize({
        'kind': kind,
        'device': str(getattr(provider, 'device', 'cpu')),
        'mesh': getattr(provider, 'mesh', None),
        'auxbasis': getattr(provider, 'auxbasis', None),
        'resolved_auxbasis_fingerprint': (
            None if auxbasis_data is None
            else canonical_sha256(auxbasis_data)),
        'omega': getattr(provider, 'omega', None),
        'metric_mesh': getattr(provider, 'metric_mesh', None),
        'resolved_omega': getattr(provider, 'resolved_omega', None),
        'resolved_source_mesh': getattr(
            provider, 'resolved_source_mesh', None),
        'resolved_metric_mesh': getattr(
            provider, 'resolved_metric_mesh', None),
        'linear_dep_threshold': getattr(
            provider, 'linear_dep_threshold', None),
        'exp_to_discard': getattr(provider, 'exp_to_discard', None),
        'j2c_eig_always': getattr(provider, 'j2c_eig_always', None),
        'gdf_precision_profile': getattr(
            provider, 'gdf_precision_profile', None),
        'gdf_precision_profile_effective': getattr(
            provider, 'gdf_precision_profile_effective', None),
    })


def source_fingerprint(
        cell_data, sampling, provider, mo_energy, mo_coeff, mo_occ, e_tot):
    '''Fingerprint the Hamiltonian, canonical arrays and source energy.'''
    payload = {
        'cell': cell_data,
        'sampling': sampling,
        'provider': provider,
        'mo_energy': _array_manifest(mo_energy),
        'mo_coeff': _array_manifest(mo_coeff),
        'mo_occ': _array_manifest(mo_occ),
        'source_energy_ha': float(e_tot),
    }
    return canonical_sha256(payload)


def canonical_sha256(value):
    '''Return deterministic SHA256 for JSON-compatible scientific data.'''
    payload = json.dumps(
        _canonicalize(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_active_state(mf, state):
    '''Rebuild the source Fock and return independent restore diagnostics.'''
    source = mf.mf_hf
    dm = source.make_rdm1(source.mo_coeff, source.mo_occ)
    overlap = source.get_ovlp()
    hcore = source.get_hcore()
    vhf = source.get_veff(source.cell, dm)
    fock = source.get_fock(h1e=hcore, s1e=overlap, vhf=vhf, dm=dm)
    active_energy = float(np.asarray(_host(
        source.energy_tot(dm=dm, h1e=hcore, vhf=vhf))).real)

    coeff = _host(source.mo_coeff)
    energy = _host(source.mo_energy)
    occ = _host(source.mo_occ)
    dm_host = _host(dm)
    overlap = _host(overlap)
    fock = _host(fock)
    blocks = 1 if coeff.ndim == 2 else coeff.shape[0]
    coeff_blocks = (coeff,) if blocks == 1 else tuple(coeff)
    energy_blocks = (energy,) if blocks == 1 else tuple(energy)
    occ_blocks = (occ,) if blocks == 1 else tuple(occ)
    overlap_blocks = (overlap,) if blocks == 1 else tuple(overlap)
    fock_blocks = (fock,) if blocks == 1 else tuple(fock)
    dm_blocks = (dm_host,) if blocks == 1 else tuple(dm_host)

    orthogonality = electron_error = fvo_max = eigen_relative = 0.0
    for c, e, occupation, s, f, density in zip(
            coeff_blocks, energy_blocks, occ_blocks,
            overlap_blocks, fock_blocks, dm_blocks, strict=True):
        identity = np.eye(c.shape[1], dtype=c.dtype)
        orthogonality = max(
            orthogonality,
            _max_abs(c.conj().T @ s @ c - identity))
        electron_error = max(
            electron_error,
            abs(float(np.einsum('ij,ji->', density, s).real)
                - float(source.cell.nelectron)))
        is_occ = occupation == 2
        is_vir = occupation == 0
        fvo_max = max(
            fvo_max,
            _max_abs(c[:, is_vir].conj().T @ f @ c[:, is_occ]))
        fc = f @ c
        residual = fc - s @ (c * e)
        eigen_relative = max(
            eigen_relative,
            float(np.linalg.norm(residual))
            / max(float(np.linalg.norm(fc)), 1.0))

    density_error = _max_abs(dm_host - state.dm)
    energy_error = abs(active_energy - state.e_tot)
    conv_grad = state.metadata['convergence_claim'].get('conv_tol_grad')
    orbital_tolerance = max(
        1e-7, 10 * (0.0 if conv_grad is None else float(conv_grad)))
    checks = {
        'orthogonality': (orthogonality, 1e-9),
        'electron number': (electron_error, 1e-8),
        'stored density': (density_error, 1e-10),
        'source energy': (energy_error, 1e-8),
        'source Fock VO': (fvo_max, orbital_tolerance),
        'generalized eigen residual': (eigen_relative, orbital_tolerance),
    }
    failures = [
        f'{name}={value:.3e}>{tolerance:.3e}'
        for name, (value, tolerance) in checks.items()
        if not math.isfinite(value) or value > tolerance
    ]
    if failures:
        raise PBCStateError(
            'restored periodic source failed physical validation: '
            + ', '.join(failures))
    if provider_identity(mf.with_df) != state.metadata['provider']:
        raise PBCStateError(
            'rebuilt provider does not match the checkpoint representation')
    return {
        'orthogonality_max_abs': orthogonality,
        'electron_number_error': electron_error,
        'density_max_abs_error': density_error,
        'source_energy_error_ha': energy_error,
        'fock_vo_max_abs': fvo_max,
        'generalized_eigen_residual_relative': eigen_relative,
    }


def _check_provider_configuration(provider, expected):
    observed = provider_identity(provider)
    keys = (
        'kind', 'device', 'mesh', 'auxbasis', 'omega', 'metric_mesh',
        'linear_dep_threshold', 'exp_to_discard', 'j2c_eig_always',
        'gdf_precision_profile',
    )
    if any(observed.get(key) != expected.get(key) for key in keys):
        raise PBCStateError(
            'checkpoint and active FFTDF/GDF configurations differ')


def _validate_array_contract(
        cell, sampling, mo_energy, mo_coeff, mo_occ, dm):
    arrays = tuple(map(np.asarray, (mo_energy, mo_coeff, mo_occ, dm)))
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise PBCStateError('periodic source arrays contain NaN/Inf')
    nao = int(cell.nao_nr())
    if sampling == 'gamma':
        if (mo_coeff.ndim != 2 or mo_coeff.shape[0] != nao
                or dm.shape != (nao, nao)):
            raise PBCStateError('Gamma source array shapes are inconsistent')
        nmo = mo_coeff.shape[1]
        if mo_energy.shape != (nmo,) or mo_occ.shape != (nmo,):
            raise PBCStateError('Gamma MO array shapes are inconsistent')
        if not np.all((mo_occ == 2) | (mo_occ == 0)):
            raise PBCStateError('Gamma RHF occupations must be exactly 2 or 0')
    elif sampling == 'sampled_k':
        if (mo_coeff.ndim != 3 or mo_coeff.shape[1] != nao
                or dm.shape != (mo_coeff.shape[0], nao, nao)):
            raise PBCStateError('sampled-k source array shapes are inconsistent')
        nkpts, _, nmo = mo_coeff.shape
        if mo_energy.shape != (nkpts, nmo) or mo_occ.shape != (nkpts, nmo):
            raise PBCStateError('sampled-k MO array shapes are inconsistent')
        validate_insulating_occupations(mo_energy, mo_occ)
    else:
        raise PBCStateError(f'unsupported periodic sampling mode: {sampling!r}')


def _backend_array(source, value):
    if type(source).__module__.startswith(('gpu4pyscf', 'dcforge.gpu')):
        import cupy as cp
        return cp.asarray(value)
    return np.asarray(value).copy()


def _host(value):
    getter = getattr(value, 'get', None)
    if callable(getter):
        value = getter()
    return np.asarray(value)


def _array_manifest(value):
    array = np.ascontiguousarray(_host(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode('ascii'))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return {
        'shape': list(array.shape),
        'dtype': array.dtype.str,
        'sha256': digest.hexdigest(),
    }


def _canonicalize(value):
    if isinstance(value, dict):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, np.ndarray):
        return _canonicalize(value.tolist())
    if isinstance(value, np.generic):
        return _canonicalize(value.item())
    if isinstance(value, complex):
        return {'real': float(value.real), 'imag': float(value.imag)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _max_abs(value):
    value = np.asarray(value)
    return float(np.max(np.abs(value))) if value.size else 0.0


def _optional_float(value):
    return None if value is None else float(value)


def _optional_int(value):
    return None if value is None else int(value)


def _sha256_file(path, block_bytes=16 * 1024**2):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        while True:
            block = stream.read(block_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    'PBCHFState', 'PBCSourceReceipt', 'PBCStateError',
    'assert_pbc_source_state', 'clear_pbc_source_state',
    'configure_pbc_provider', 'load_pbc_hf_state',
    'record_pbc_source_state', 'restore_pbc_source_state',
    'save_pbc_hf_state',
]
