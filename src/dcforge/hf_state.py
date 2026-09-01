'''Versioned molecular RHF/UHF source-state checkpoints.

The checkpoint keeps PySCF's standard ``mol`` and ``scf`` groups and adds a
small DC-Forge manifest plus an independently stored AO density.  It stores
no CDERI: provider factors and canonical source orbitals have different
identities and lifecycles.
'''

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform

import h5py
import numpy as np
import pyscf
from pyscf.scf import chkfile


_SCHEMA_VERSION = 1
_GROUP = 'dcforge_hf_state'


class HFStateError(RuntimeError):
    '''Base class for molecular HF source-state failures.'''


class HFStateSchemaError(HFStateError):
    '''The checkpoint is incomplete, corrupt, or has an unsupported schema.'''


class HFStateMismatchError(HFStateError):
    '''The checkpoint and requested molecular workflow are incompatible.'''


class HFStateValidationError(HFStateError):
    '''The restored arrays fail an independent physical validation.'''


class HFStateInvalidatedError(HFStateError):
    '''An accepted source state changed after validation.'''


@dataclass(frozen=True)
class HFState:
    '''Host-owned source arrays and verified checkpoint metadata.'''

    path: str
    mol: object
    reference: str
    mo_energy: np.ndarray
    mo_coeff: np.ndarray
    mo_occ: np.ndarray
    dm: np.ndarray
    e_tot: float
    metadata: dict
    checkpoint_sha256: str


def save_hf_state(path, mf_hf, *, extra_provenance=None):
    '''Atomically save one converged molecular canonical RHF/UHF state.

    Args:
        path : str or pathlib.Path
            Destination PySCF-compatible HDF5 checkpoint.
        mf_hf : pyscf or gpu4pyscf RHF/UHF object
            Converged finite-molecular source object. ROHF is deliberately
            outside schema version 1.

    Kwargs:
        extra_provenance : dict or None
            Optional JSON-serializable caller metadata. It is provenance and
            never substitutes for restore-time physical validation.

    Returns:
        dict
            Complete checkpoint manifest.
    '''
    if hasattr(mf_hf.mol, 'pbc_intor'):
        raise NotImplementedError('HF source-state schema 1 is molecular only')
    if not bool(getattr(mf_hf, 'converged', False)):
        raise HFStateValidationError('source HF object must be converged')

    mo_coeff = _as_host_array(mf_hf.mo_coeff)
    mo_occ = _as_host_array(mf_hf.mo_occ)
    mo_energy = _as_host_array(mf_hf.mo_energy)
    reference = infer_reference(mo_coeff, mo_occ)
    _validate_array_contract(
        reference, mf_hf.mol, mo_energy, mo_coeff, mo_occ)
    dm = _as_host_array(mf_hf.make_rdm1())
    _validate_density_shape(reference, mf_hf.mol, dm)
    e_tot = float(mf_hf.e_tot)
    if not np.isfinite(e_tot):
        raise HFStateValidationError('source HF energy is not finite')

    mol_identity = molecule_identity(mf_hf.mol)
    representation = representation_identity(mf_hf)
    if (representation.get('kind') == 'density_fitted'
            and representation.get('only_dfj')):
        raise NotImplementedError(
            'gradient-ready RHF/UHF checkpoints require only_dfj=False')
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
        'reference': reference,
        'molecule_identity': mol_identity,
        'molecule_fingerprint': canonical_sha256(mol_identity),
        'representation': representation,
        'representation_fingerprint': canonical_sha256(representation),
        'state_fingerprint': source_state_fingerprint(
            mf_hf.mol, reference, mo_energy, mo_coeff, mo_occ),
        'source_energy_ha': e_tot,
        'convergence_claim': {
            'converged': True,
            'conv_tol': _optional_float(getattr(mf_hf, 'conv_tol', None)),
            'conv_tol_grad': _optional_float(
                getattr(mf_hf, 'conv_tol_grad', None)),
            'cycles': _optional_int(getattr(mf_hf, 'cycles', None)),
        },
        'versions': _versions(mf_hf),
        'arrays': arrays,
        'extra_provenance': (
            {} if extra_provenance is None
            else _canonicalize(extra_provenance)),
    }

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.tmp-{os.getpid()}')
    try:
        temporary.unlink()
    except FileNotFoundError:
        pass
    try:
        chkfile.dump_scf(
            mf_hf.mol, str(temporary), e_tot,
            mo_energy, mo_coeff, mo_occ,
        )
        with h5py.File(temporary, 'a') as handle:
            group = handle.create_group(_GROUP)
            group.attrs['complete'] = True
            group.attrs['metadata_json'] = json.dumps(
                metadata, sort_keys=True, separators=(',', ':'),
                allow_nan=False)
            group.create_dataset('dm', data=dm)
            handle.flush()
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return metadata


def inspect_hf_state(path):
    '''Return a checksum-verified manifest without allocating GPU arrays.'''
    return load_hf_state(path).metadata


def load_hf_state(path):
    '''Load and checksum one molecular RHF/UHF source-state checkpoint.'''
    path = Path(path).expanduser().resolve()
    try:
        with h5py.File(path, 'r') as handle:
            group = handle[_GROUP]
            if not bool(group.attrs.get('complete', False)):
                raise HFStateSchemaError('HF source-state checkpoint is incomplete')
            metadata = json.loads(group.attrs['metadata_json'])
            dm = np.asarray(group['dm'])
    except HFStateSchemaError:
        raise
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise HFStateSchemaError(
            f'cannot read DC-Forge HF source state from {path}') from error

    if metadata.get('schema_version') != _SCHEMA_VERSION:
        raise HFStateSchemaError(
            'unsupported HF source-state schema: '
            f'{metadata.get("schema_version")!r}')
    if not metadata.get('complete', False):
        raise HFStateSchemaError('HF source-state metadata is incomplete')
    if metadata.get('capability') != 'gradient_ready':
        raise HFStateSchemaError('schema 1 requires capability=gradient_ready')

    try:
        mol = chkfile.load_mol(str(path))
        scf_data = chkfile.load(str(path), 'scf')
        mo_energy = np.asarray(scf_data['mo_energy'])
        mo_coeff = np.asarray(scf_data['mo_coeff'])
        mo_occ = np.asarray(scf_data['mo_occ'])
        e_tot = float(scf_data['e_tot'])
    except Exception as error:
        raise HFStateSchemaError('standard PySCF SCF data are incomplete') from error

    reference = metadata.get('reference')
    if reference not in ('RHF', 'UHF'):
        raise HFStateSchemaError(
            f'unsupported molecular reference in schema 1: {reference!r}')
    try:
        observed_reference = infer_reference(mo_coeff, mo_occ)
        _validate_array_contract(
            reference, mol, mo_energy, mo_coeff, mo_occ)
        _validate_density_shape(reference, mol, dm)
    except (HFStateValidationError, NotImplementedError) as error:
        raise HFStateSchemaError(
            'checkpoint RHF/UHF array contract is invalid') from error
    if observed_reference != reference:
        raise HFStateSchemaError(
            f'reference metadata says {reference}, arrays describe '
            f'{observed_reference}')

    observed_mol = molecule_identity(mol)
    if canonical_sha256(observed_mol) != metadata.get('molecule_fingerprint'):
        raise HFStateSchemaError('checkpoint molecule fingerprint mismatch')
    if metadata.get('molecule_identity') != observed_mol:
        raise HFStateSchemaError('checkpoint molecule identity is inconsistent')
    if float(metadata.get('source_energy_ha')) != e_tot:
        raise HFStateSchemaError('standard and DC-Forge source energies differ')

    values = {
        'mo_energy': mo_energy,
        'mo_coeff': mo_coeff,
        'mo_occ': mo_occ,
        'dm': dm,
    }
    expected_arrays = metadata.get('arrays', {})
    for name, value in values.items():
        if name not in expected_arrays:
            raise HFStateSchemaError(f'missing array manifest: {name}')
        if _array_manifest(value) != expected_arrays[name]:
            raise HFStateSchemaError(f'checkpoint array mismatch: {name}')

    fingerprint = source_state_fingerprint(
        mol, reference, mo_energy, mo_coeff, mo_occ)
    if fingerprint != metadata.get('state_fingerprint'):
        raise HFStateSchemaError('checkpoint source-state fingerprint mismatch')
    representation = metadata.get('representation')
    if not isinstance(representation, dict):
        raise HFStateSchemaError('checkpoint representation metadata is missing')
    if canonical_sha256(representation) != metadata.get(
            'representation_fingerprint'):
        raise HFStateSchemaError('checkpoint representation fingerprint mismatch')

    for value in (mo_energy, mo_coeff, mo_occ, dm):
        value.setflags(write=False)

    return HFState(
        path=str(path), mol=mol, reference=reference,
        mo_energy=mo_energy, mo_coeff=mo_coeff, mo_occ=mo_occ,
        dm=dm, e_tot=e_tot, metadata=metadata,
        checkpoint_sha256=_sha256_file(path),
    )


def infer_reference(mo_coeff, mo_occ):
    '''Return RHF/UHF from array shape and exact occupations; reject ROHF.'''
    coeff = np.asarray(mo_coeff)
    occ = np.asarray(mo_occ)
    if coeff.ndim == 2 and occ.ndim == 1:
        if np.any(occ == 1):
            raise NotImplementedError('ROHF source-state checkpoints are excluded')
        if not np.all(np.isin(occ, (0, 2))):
            raise HFStateValidationError('RHF occupations must be exact 2/0')
        return 'RHF'
    if coeff.ndim == 3 and coeff.shape[0] == 2 and occ.ndim == 2:
        if not np.all(np.isin(occ, (0, 1))):
            raise HFStateValidationError('UHF occupations must be exact 1/0')
        return 'UHF'
    raise HFStateValidationError('source arrays are neither RHF nor UHF')


def molecule_identity(mol):
    '''Return stable, physical finite-molecule input and AO ordering.'''
    return _canonicalize({
        'boundary': 'molecular',
        'atom_symbols': [mol.atom_symbol(i) for i in range(mol.natm)],
        'atom_charges': np.asarray(mol.atom_charges()),
        'atom_coords_bohr': np.asarray(mol.atom_coords()),
        'charge': int(mol.charge),
        'spin': int(mol.spin),
        'nelectron': int(mol.nelectron),
        'cart': bool(mol.cart),
        'symmetry': bool(mol.symmetry),
        'groupname': str(getattr(mol, 'groupname', 'C1')),
        'basis': getattr(mol, '_basis', {}),
        'ecp': getattr(mol, '_ecp', {}),
        'nucmod': getattr(mol, '_nucmod', {}),
        'nao': int(mol.nao),
        'ao_labels': list(map(str, mol.ao_labels())),
    })


def molecule_fingerprint(mol):
    '''Return the SHA256 of the physical molecular identity.'''
    return canonical_sha256(molecule_identity(mol))


def representation_identity(mf_hf):
    '''Return the scientific conventional/JK-DF representation identity.'''
    if not hasattr(mf_hf, 'with_df'):
        return {'kind': 'conventional'}
    with_df = mf_hf.with_df
    auxmol = getattr(with_df, 'auxmol', None)
    auxbasis_data = None if auxmol is None else getattr(auxmol, '_basis', None)
    return _canonicalize({
        'kind': 'density_fitted',
        'provider_contract': 'molecular_jk_df',
        'auxbasis': getattr(with_df, 'auxbasis', None),
        'auxbasis_fingerprint': (
            None if auxbasis_data is None
            else canonical_sha256(auxbasis_data)),
        'only_dfj': bool(getattr(mf_hf, 'only_dfj', False)),
    })


def source_state_fingerprint(
        mol, reference, mo_energy, mo_coeff, mo_occ):
    '''Fingerprint molecule, reference and canonical response arrays.'''
    digest = hashlib.sha256()
    digest.update(molecule_fingerprint(mol).encode())
    digest.update(str(reference).encode())
    for value in (mo_energy, mo_coeff, mo_occ):
        digest.update(array_sha256(value).encode())
    return digest.hexdigest()


def array_sha256(value):
    '''Return a shape/dtype/content-sensitive array SHA256.'''
    array = np.ascontiguousarray(_as_host_array(value))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.dtype.str.encode())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def canonical_sha256(value):
    '''Return a deterministic SHA256 for JSON-compatible scientific data.'''
    payload = json.dumps(
        _canonicalize(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_array_contract(reference, mol, mo_energy, mo_coeff, mo_occ):
    arrays = tuple(map(np.asarray, (mo_energy, mo_coeff, mo_occ)))
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise HFStateValidationError('source orbital arrays contain NaN/Inf')
    if np.iscomplexobj(mo_coeff):
        raise NotImplementedError('complex molecular source orbitals are excluded')
    nao = int(mol.nao)
    if reference == 'RHF':
        if mo_coeff.ndim != 2 or mo_coeff.shape[0] != nao:
            raise HFStateValidationError('RHF mo_coeff must have nao AO rows')
        nmo = mo_coeff.shape[1]
        if mo_energy.shape != (nmo,) or mo_occ.shape != (nmo,):
            raise HFStateValidationError('RHF energy/occupation shape mismatch')
        if not np.all(np.isin(mo_occ, (0, 2))):
            raise HFStateValidationError('RHF occupations must be exact 2/0')
        nocc = int(np.count_nonzero(mo_occ))
        if not np.all(mo_occ[:nocc] == 2) or not np.all(mo_occ[nocc:] == 0):
            raise HFStateValidationError('RHF occupied orbitals must precede virtuals')
    elif reference == 'UHF':
        if (mo_coeff.ndim != 3 or mo_coeff.shape[0] != 2
                or mo_coeff.shape[1] != nao):
            raise HFStateValidationError('UHF mo_coeff must have 2 spin/nao AO rows')
        nmo = mo_coeff.shape[2]
        if mo_energy.shape != (2, nmo) or mo_occ.shape != (2, nmo):
            raise HFStateValidationError('UHF energy/occupation shape mismatch')
        if not np.all(np.isin(mo_occ, (0, 1))):
            raise HFStateValidationError('UHF occupations must be exact 1/0')
        for occ in mo_occ:
            nocc = int(np.count_nonzero(occ))
            if not np.all(occ[:nocc] == 1) or not np.all(occ[nocc:] == 0):
                raise HFStateValidationError(
                    'UHF occupied orbitals must precede virtuals')
    else:
        raise NotImplementedError(f'unsupported reference: {reference}')


def _validate_density_shape(reference, mol, dm):
    expected = (mol.nao, mol.nao) if reference == 'RHF' else (
        2, mol.nao, mol.nao)
    if np.asarray(dm).shape != expected:
        raise HFStateValidationError(
            f'{reference} density shape must be {expected}')
    if not np.all(np.isfinite(dm)):
        raise HFStateValidationError('source density contains NaN/Inf')


def _array_manifest(value):
    array = np.asarray(value)
    return {
        'shape': list(array.shape),
        'dtype': array.dtype.str,
        'sha256': array_sha256(array),
    }


def _versions(mf_hf):
    versions = {
        'python': platform.python_version(),
        'pyscf': str(pyscf.__version__),
        'source_module': type(mf_hf).__module__,
        'source_class': type(mf_hf).__name__,
    }
    if type(mf_hf).__module__.startswith('gpu4pyscf'):
        import gpu4pyscf
        versions['gpu4pyscf'] = str(gpu4pyscf.__version__)
    return versions


def _as_host_array(value):
    if hasattr(value, 'get'):
        value = value.get()
    return np.asarray(value)


def _canonicalize(value):
    if isinstance(value, dict):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
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
    'HFState',
    'HFStateError',
    'HFStateInvalidatedError',
    'HFStateMismatchError',
    'HFStateSchemaError',
    'HFStateValidationError',
    'array_sha256',
    'canonical_sha256',
    'inspect_hf_state',
    'load_hf_state',
    'molecule_fingerprint',
    'representation_identity',
    'save_hf_state',
    'source_state_fingerprint',
]
