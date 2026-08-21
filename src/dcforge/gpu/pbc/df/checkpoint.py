'''Independent, fingerprinted checkpoints for GPU Gamma Gaussian-DF factors.

This module deliberately stores no molecular orbitals.  A provider factor is
an expensive representation of the Gaussian density-fitting operator, while
an RHF orbital state is the solution of a particular source Hamiltonian.
Keeping their identities separate allows a target-functional change to reuse
the former without silently accepting stale source orbitals.  GDF and RSDF
share the compressed factor layout; RSDF additionally stores the exact
source/metric split and its authoritative central metric/Cholesky factor.
'''

import hashlib
import inspect
import json
import os
import time
from pathlib import Path

import cupy as cp
import gpu4pyscf
import h5py
import numpy as np
import pyscf
from gpu4pyscf.lib.cupy_helper import empty_mapped
from pyscf.pbc.df import df as cpu_df

import dcforge
from dcforge.gpu.pbc.df.cderi import H5CDERI, cderi_memory_report

_SCHEMA_VERSION = 3
_CHECKSUM_BLOCK_BYTES = 16 * 1024**2


def provider_factor_identity(provider):
    '''Return the geometry, basis, metric, and ABI identity of one provider.

    The identity intentionally excludes source orbitals and target XC.  It
    includes a canonical physical Cell identity, AO ordering,
    auxiliary-basis specification, k-points, metric thresholds, and package
    ABIs that define the compressed factor layout.  The canonical identity
    deliberately excludes PySCF's raw ``Cell.dumps()`` representation:
    equivalent cells can place element basis data at different offsets in
    ``Cell._env`` depending on import order, although every physical input and
    AO label is unchanged.

    Args:
        provider : dcforge.gpu.pbc.df.gdf.GDF
            Real-Gamma GDF provider, built or not.

    Returns:
        dict
            JSON-serializable factor identity.
    '''
    cell = provider.cell
    auxcell, representation = _resolved_provider_representation(provider)
    kpts = np.asarray(provider.kpts, dtype=float).reshape(-1, 3)
    labels = list(map(str, cell.ao_labels()))
    cell_identity = _cell_physical_identity(cell)
    provider_source = inspect.getsourcefile(type(provider))
    return {
        'schema_version': _SCHEMA_VERSION,
        'provider_module': type(provider).__module__,
        'provider_class': type(provider).__name__,
        'provider_source_sha256': (
            None if provider_source is None
            else _sha256_bytes(Path(provider_source).read_bytes())),
        'cell_physical_sha256': _canonical_sha256(cell_identity),
        'lattice_bohr': np.asarray(cell.lattice_vectors()).tolist(),
        'atom_coords_bohr': np.asarray(cell.atom_coords()).tolist(),
        'atom_symbols': [cell.atom_symbol(index)
                         for index in range(cell.natm)],
        'basis_sha256': _canonical_sha256(cell._basis),
        'pseudo_sha256': _canonical_sha256(cell._pseudo),
        'auxbasis': _canonicalize(provider.auxbasis),
        'auxbasis_sha256': _canonical_sha256(provider.auxbasis),
        'precision': float(cell.precision),
        'dimension': int(cell.dimension),
        'low_dim_ft_type': str(cell.low_dim_ft_type),
        'cell_omega': float(cell.omega),
        'kpts': kpts.tolist(),
        'linear_dep_threshold': float(provider.linear_dep_threshold),
        'exp_to_discard': (
            None if provider.exp_to_discard is None
            else float(provider.exp_to_discard)),
        'nao': int(cell.nao_nr()),
        'naux': int(auxcell.nao_nr()),
        'ao_labels_sha256': _canonical_sha256(labels),
        'pyscf_version': str(pyscf.__version__),
        'gpu4pyscf_version': str(gpu4pyscf.__version__),
        'dcforge_version': str(dcforge.__version__),
        'storage_dtype': _provider_storage_dtype(provider),
        'representation': representation,
    }


def _cell_physical_identity(cell):
    '''Return stable result-defining Cell data without memory-layout offsets.

    PySCF's serialized ``_bas`` entries contain pointers into ``_env``.  The
    pointers can change when equivalent per-element basis tables are inserted
    in a different order.  Hashing them rejects valid cross-process
    checkpoints, so this identity records the underlying physical data and AO
    ordering explicitly instead.
    '''
    ke_cutoff = getattr(cell, 'ke_cutoff', None)
    mesh = getattr(cell, 'mesh', None)
    magmom = getattr(cell, 'magmom', None)
    return {
        'lattice_bohr': np.asarray(cell.lattice_vectors()).tolist(),
        'atom_coords_bohr': np.asarray(cell.atom_coords()).tolist(),
        'atom_symbols': [cell.atom_symbol(index)
                         for index in range(cell.natm)],
        'atom_charges': np.asarray(cell.atom_charges()).tolist(),
        'basis': _canonicalize(cell._basis),
        'pseudo': _canonicalize(cell._pseudo),
        'charge': int(cell.charge),
        'spin': int(cell.spin),
        'cart': bool(cell.cart),
        'magmom': _canonicalize(magmom),
        'precision': float(cell.precision),
        'dimension': int(cell.dimension),
        'low_dim_ft_type': str(cell.low_dim_ft_type),
        'omega': float(cell.omega),
        'ke_cutoff': None if ke_cutoff is None else float(ke_cutoff),
        'mesh': None if mesh is None else np.asarray(mesh, dtype=int).tolist(),
        'nao': int(cell.nao_nr()),
        'ao_labels': list(map(str, cell.ao_labels())),
    }


def identity_fingerprint(identity):
    '''Return the deterministic SHA256 of one factor identity.'''
    payload = json.dumps(
        identity, sort_keys=True, separators=(',', ':')).encode()
    return _sha256_bytes(payload)


def save_provider_factor_checkpoint(
        path, provider, identity=None, block_size=64):
    '''Atomically save one complete compressed GDF factor, without orbitals.

    Args:
        path : str or pathlib.Path
            Destination HDF5 file.  A sibling ``.tmp`` file is replaced only
            after every dataset checksum and completion marker is written.
        provider : dcforge.gpu.pbc.df.gdf.GDF
            Built real-Gamma GDF provider.

    Kwargs:
        identity : dict or None
            Expected provider identity.  ``None`` derives it from provider.
        block_size : int
            Auxiliary rows copied per transaction.

    Returns:
        dict
            Checkpoint provenance and exact byte/checksum metadata.
    '''
    _validate_built_provider(provider)
    if getattr(provider, 'gdf_precision_profile', 'fp64') != 'fp64':
        raise NotImplementedError(
            'mixed-precision GDF factor checkpoints are not yet supported; '
            'a checkpoint must not erase whether quantization occurred during '
            'or after CDERI transformation')
    rows = max(1, int(block_size))
    identity = provider_factor_identity(provider) if identity is None else identity
    expected = provider_factor_identity(provider)
    if identity != expected:
        raise ValueError(
            f'provided factor identity does not match provider: '
            f'{_identity_differences(identity, expected)}')

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        temporary.unlink()
    except FileNotFoundError:
        pass
    started = time.perf_counter()
    metadata = {
        'schema_version': _SCHEMA_VERSION,
        'provider_identity': identity,
        'provider_fingerprint': identity_fingerprint(identity),
        'provider_j_only': bool(getattr(provider, '_j_only', False)),
        'resolved_omega': getattr(provider, 'resolved_omega', None),
        'resolved_source_mesh': _optional_integer_list(
            getattr(provider, 'resolved_source_mesh', None)),
        'resolved_metric_mesh': _optional_integer_list(
            getattr(provider, 'resolved_metric_mesh', None)),
        'response_state': None,
        'complete': False,
    }
    try:
        with h5py.File(temporary, 'w') as handle:
            handle.attrs['schema_version'] = _SCHEMA_VERSION
            handle.attrs['complete'] = False
            index = handle.create_group('cderi_index')
            pair_address = cp.asnumpy(provider._cderi_idx[0])
            diagonal = cp.asnumpy(provider._cderi_idx[1])
            layout = {
                'pair_address': _write_array(
                    index, 'pair_address', pair_address),
                'diagonal': _write_array(index, 'diagonal', diagonal),
            }
            factors = handle.create_group('cderi')
            factor_metadata = {
                'positive': _write_mapping(
                    factors.create_group('positive'), provider._cderi, rows),
                'negative': None,
            }
            if getattr(provider, '_cderip', None) is not None:
                factor_metadata['negative'] = _write_mapping(
                    factors.create_group('negative'), provider._cderip, rows)
            if getattr(provider, 'backend_name', None) == 'rsdf':
                central_metric = getattr(provider, '_central_metric', None)
                if central_metric is None:
                    raise RuntimeError(
                        'built RSDF checkpoint requires its central metric')
                response = handle.create_group('response_state')
                central_metric = cp.asnumpy(central_metric)
                central_factor = cp.asnumpy(
                    provider.central_metric_factor())
                metadata['response_state'] = {
                    'central_metric': _write_array(
                        response, 'central_metric', central_metric),
                    'central_factor': _write_array(
                        response, 'central_factor', central_factor),
                }
            metadata['layout'] = layout
            metadata['factors'] = factor_metadata
            metadata['factor_fingerprint'] = _factor_fingerprint(
                layout, factor_metadata)
            metadata['complete'] = True
            handle.attrs['metadata_json'] = json.dumps(
                metadata, sort_keys=True)
            handle.attrs['complete'] = True
            handle.flush()
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return {
        **metadata,
        'path': str(path),
        'bytes': int(path.stat().st_size),
        'write_seconds': time.perf_counter() - started,
    }


def load_provider_factor_checkpoint(
        path, provider, expected_identity=None, storage='mapped_host',
        block_size=64):
    '''Restore a validated provider factor without touching source orbitals.

    ``mapped_host`` copies factors once into CUDA-mapped host memory.
    ``hdf5`` leaves them in the user-owned checkpoint and streams slices.
    Dataset checksums are verified before any arrays are installed.
    '''
    if getattr(provider, 'gdf_precision_profile', 'fp64') != 'fp64':
        raise NotImplementedError(
            'mixed-precision GDF profiles currently require a fresh provider '
            'build; provider-factor checkpoint restore is FP64-only')
    if storage not in ('mapped_host', 'hdf5'):
        raise ValueError("storage must be 'mapped_host' or 'hdf5'")
    rows = max(1, int(block_size))
    path = Path(path).expanduser().resolve()
    identity = (
        provider_factor_identity(provider)
        if expected_identity is None else expected_identity)
    expected_fingerprint = identity_fingerprint(identity)
    started = time.perf_counter()
    with h5py.File(path, 'r') as handle:
        if not bool(handle.attrs.get('complete', False)):
            raise ValueError('provider-factor checkpoint is incomplete')
        try:
            metadata = json.loads(handle.attrs['metadata_json'])
        except KeyError as error:
            raise ValueError(
                'provider-factor checkpoint has no metadata') from error
        if metadata.get('schema_version') != _SCHEMA_VERSION:
            raise ValueError(
                'unsupported provider-factor checkpoint schema: '
                f'{metadata.get("schema_version")!r}')
        if not metadata.get('complete', False):
            raise ValueError('provider-factor checkpoint metadata is incomplete')
        observed = metadata.get('provider_fingerprint')
        if observed != expected_fingerprint:
            differences = _identity_differences(
                metadata.get('provider_identity', {}), identity)
            raise ValueError(
                'provider-factor checkpoint identity mismatch: '
                f'checkpoint={observed}, requested={expected_fingerprint}, '
                f'differences={differences}')

        pair_address = _read_verified_array(
            handle['cderi_index/pair_address'],
            metadata['layout']['pair_address'])
        diagonal = _read_verified_array(
            handle['cderi_index/diagonal'],
            metadata['layout']['diagonal'])
        positive = _load_verified_mapping(
            path, handle['cderi/positive'],
            metadata['factors']['positive'], storage, rows)
        negative = None
        if metadata['factors'].get('negative') is not None:
            negative = _load_verified_mapping(
                path, handle['cderi/negative'],
                metadata['factors']['negative'], storage, rows)
        observed_factor = _factor_fingerprint(
            metadata['layout'], metadata['factors'])
        if observed_factor != metadata.get('factor_fingerprint'):
            raise ValueError('provider-factor metadata fingerprint mismatch')
        response_state = None
        if getattr(provider, 'backend_name', None) == 'rsdf':
            response_metadata = metadata.get('response_state')
            if not response_metadata:
                raise ValueError(
                    'RSDF checkpoint has no authoritative response state')
            central_metric = _read_verified_array(
                handle['response_state/central_metric'],
                response_metadata['central_metric'])
            central_factor = _read_verified_array(
                handle['response_state/central_factor'],
                response_metadata['central_factor'])
            if central_metric.shape != central_factor.shape:
                raise ValueError(
                    'RSDF central metric/factor checkpoint shapes differ')
            residual = np.max(np.abs(
                central_factor @ central_factor.T - central_metric))
            tolerance = max(
                1.e-10,
                1.e-10 * float(np.max(np.abs(central_metric))))
            if residual > tolerance:
                raise ValueError(
                    'RSDF checkpoint central factor does not reconstruct '
                    f'its metric: {residual:.3e} > {tolerance:.3e}')
            response_state = (central_metric, central_factor, residual)

    _validate_layout(pair_address, diagonal, identity['nao'])
    provider.auxcell, representation = _resolved_provider_representation(
        provider)
    provider.nao = provider.cell.nao
    provider._omega = (
        None if getattr(provider, 'backend_name', None) == 'rsdf'
        else provider.cell.omega)
    provider.kmesh = [1, 1, 1]
    provider._j_only = bool(metadata['provider_j_only'])
    provider._cderi = positive
    provider._cderip = negative
    provider._cderi_idx = (
        cp.asarray(pair_address), cp.asarray(diagonal))
    provider.cderi_storage = storage
    # A persistent user checkpoint is not provider-owned temporary storage.
    provider._cderi_storage_files = []
    provider.resolved_omega = metadata.get('resolved_omega')
    provider.resolved_source_mesh = _optional_integer_array(
        metadata.get('resolved_source_mesh'))
    provider.resolved_metric_mesh = _optional_integer_array(
        metadata.get('resolved_metric_mesh'))
    if response_state is not None:
        central_metric, central_factor, factor_residual = response_state
        provider._central_metric = cp.asarray(central_metric)
        provider._central_metric_factor = cp.asarray(central_factor)
        provider.central_build_metadata = {
            'builder': 'restored provider-factor checkpoint',
            'checkpoint_path': str(path),
            'source_omega': representation['source_omega'],
            'source_mesh': representation['source_mesh'],
            'metric_omega': representation['metric_omega'],
            'metric_mesh': representation['metric_mesh'],
            'central_factor_reconstruction_max_abs': float(
                factor_residual),
        }
    provider.cderi_memory_metadata = cderi_memory_report(provider)
    return {
        **metadata,
        'path': str(path),
        'bytes': int(path.stat().st_size),
        'load_seconds': time.perf_counter() - started,
        'storage': storage,
        'requested_provider_fingerprint': expected_fingerprint,
        'cderi_memory': provider.cderi_memory_metadata,
    }


def _validate_built_provider(provider):
    if np.asarray(provider.kpts).reshape(-1, 3).shape != (1, 3):
        raise NotImplementedError(
            'provider-factor checkpoints currently require one Gamma point')
    if np.max(np.abs(np.asarray(provider.kpts))) >= 1.e-12:
        raise NotImplementedError(
            'provider-factor checkpoints currently require real Gamma')
    if provider._cderi is None or provider._cderi_idx is None:
        raise RuntimeError('GDF provider factors are not built')


def _write_array(group, name, value):
    value = np.ascontiguousarray(value)
    dataset = group.create_dataset(name, data=value)
    return _dataset_metadata(dataset, value)


def _write_mapping(group, mapping, block_size):
    result = {}
    for key, value in mapping.items():
        name = str(int(key))
        shape = tuple(map(int, value.shape))
        rows = max(1, min(block_size, shape[0]))
        columns = max(1, min(8192, shape[1]))
        dataset = group.create_dataset(
            name, shape=shape, dtype=value.dtype,
            chunks=(rows, columns))
        digest = hashlib.sha256()
        for p0 in range(0, shape[0], rows):
            p1 = min(shape[0], p0 + rows)
            block = np.ascontiguousarray(value[p0:p1])
            dataset[p0:p1] = block
            digest.update(block.view(np.uint8))
        result[name] = {
            'shape': list(shape),
            'dtype': np.dtype(value.dtype).str,
            'sha256': digest.hexdigest(),
            'bytes': int(np.prod(shape, dtype=np.int64)
                         * np.dtype(value.dtype).itemsize),
        }
    return result


def _read_verified_array(dataset, metadata):
    value = np.ascontiguousarray(dataset[...])
    _verify_array_metadata(value, metadata, dataset.name)
    return value


def _load_verified_mapping(
        path, group, metadata, storage, block_size):
    if set(group) != set(metadata):
        raise ValueError(
            f'factor dataset keys differ: file={sorted(group)}, '
            f'metadata={sorted(metadata)}')
    result = {}
    for name, expected in metadata.items():
        dataset = group[name]
        shape = tuple(map(int, dataset.shape))
        dtype = np.dtype(dataset.dtype)
        if shape != tuple(expected['shape']) or dtype.str != expected['dtype']:
            raise ValueError(f'factor dataset {dataset.name} shape/dtype mismatch')
        rows = max(1, min(block_size, shape[0]))
        digest = hashlib.sha256()
        output = None
        if storage == 'mapped_host':
            output = empty_mapped(shape, dtype=dtype)
        for p0 in range(0, shape[0], rows):
            p1 = min(shape[0], p0 + rows)
            block = np.ascontiguousarray(dataset[p0:p1])
            digest.update(block.view(np.uint8))
            if output is not None:
                output[p0:p1] = block
        if digest.hexdigest() != expected['sha256']:
            raise ValueError(
                f'factor dataset checksum mismatch: {dataset.name}')
        result[int(name)] = (
            output if output is not None else H5CDERI(
                path, dataset.name.lstrip('/'), shape, dtype))
    return result


def _dataset_metadata(dataset, value):
    return {
        'shape': list(map(int, value.shape)),
        'dtype': value.dtype.str,
        'sha256': _sha256_bytes(value.view(np.uint8)),
        'bytes': int(value.nbytes),
    }


def _verify_array_metadata(value, metadata, name):
    if list(value.shape) != metadata['shape']:
        raise ValueError(f'checkpoint array shape mismatch: {name}')
    if value.dtype.str != metadata['dtype']:
        raise ValueError(f'checkpoint array dtype mismatch: {name}')
    if _sha256_bytes(value.view(np.uint8)) != metadata['sha256']:
        raise ValueError(f'checkpoint array checksum mismatch: {name}')


def _factor_fingerprint(layout, factors):
    return _canonical_sha256({'layout': layout, 'factors': factors})


def _validate_layout(pair_address, diagonal, nao):
    pair_address = np.asarray(pair_address)
    diagonal = np.asarray(diagonal)
    if pair_address.ndim != 1 or diagonal.ndim != 1:
        raise ValueError('GDF pair layout arrays must be one-dimensional')
    if len(pair_address) < 1:
        raise ValueError('GDF pair layout is empty')
    if np.any(pair_address < 0) or np.any(pair_address >= int(nao)**2):
        raise ValueError('GDF pair address is outside the AO square')


def _provider_storage_dtype(provider):
    if getattr(provider, '_cderi', None):
        first = next(iter(provider._cderi.values()))
        return np.dtype(first.dtype).str
    kpts = np.asarray(provider.kpts).reshape(-1, 3)
    return np.dtype(
        np.float64 if len(kpts) == 1 and np.max(np.abs(kpts)) < 1.e-12
        else np.complex128).str


def _resolved_provider_representation(provider):
    '''Return auxiliary Cell and result-defining Gaussian representation.'''
    backend = getattr(provider, 'backend_name', None)
    if backend != 'rsdf':
        from gpu4pyscf.pbc.df import rsdf_builder

        from dcforge.gpu.pbc.df.gdf.gdf import resolve_partition_omega

        auxcell = provider.auxcell
        if auxcell is None:
            auxcell = cpu_df.make_auxcell(
                provider.cell, provider.auxbasis, provider.exp_to_discard)
        physical_omega = abs(float(getattr(provider.cell, 'omega', 0.)))
        partition_omega = resolve_partition_omega(
            provider.cell, getattr(provider, 'omega', None))
        if physical_omega < partition_omega:
            source_cutoff = rsdf_builder.estimate_ke_cutoff_for_omega(
                provider.cell, partition_omega)
            source_mesh = provider.cell.symmetrize_mesh(
                provider.cell.cutoff_to_mesh(source_cutoff))
            metric_cutoff = rsdf_builder.estimate_ke_cutoff_for_omega(
                auxcell, partition_omega, auxcell.precision * 1.e-6)
            metric_mesh = auxcell.cutoff_to_mesh(metric_cutoff)
        else:
            source_mesh = metric_mesh = np.ones(3, dtype=int)
        return auxcell, {
            'backend': backend,
            'source_omega': float(partition_omega),
            'source_mesh': np.asarray(source_mesh, dtype=int).tolist(),
            'source_precision': float(provider.cell.precision),
            'metric_omega': float(partition_omega),
            'metric_mesh': np.asarray(metric_mesh, dtype=int).tolist(),
            'metric_precision': float(auxcell.precision),
        }

    required = ('omega', 'mesh_compact', 'omega_j2c', 'mesh_j2c')
    if provider.auxcell is None or any(
            getattr(provider, name, None) is None for name in required):
        witness = provider._build_parameter_witness()
        provider._adopt_witness_settings(witness)
        auxcell = witness.auxcell
        owner = witness
    else:
        auxcell = provider.auxcell
        owner = provider
    return auxcell, {
        'backend': 'rsdf',
        'source_omega': float(owner.omega),
        'source_mesh': np.asarray(owner.mesh_compact, dtype=int).tolist(),
        'source_ke_cutoff': (
            None if owner.ke_cutoff is None else float(owner.ke_cutoff)),
        'source_precision': float(owner.precision_R),
        'reciprocal_precision': float(owner.precision_G),
        'metric_omega': float(owner.omega_j2c),
        'metric_mesh': np.asarray(owner.mesh_j2c, dtype=int).tolist(),
        'metric_precision': float(owner.precision_j2c),
        'npw_max': int(owner.npw_max),
        'use_bvk': bool(owner.use_bvk),
        'j2c_eig_always': bool(owner.j2c_eig_always),
    }


def _canonicalize(value):
    if isinstance(value, dict):
        return {str(key): _canonicalize(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _canonical_sha256(value):
    payload = json.dumps(
        _canonicalize(value), sort_keys=True,
        separators=(',', ':')).encode()
    return _sha256_bytes(payload)


def _sha256_bytes(value):
    return hashlib.sha256(memoryview(value)).hexdigest()


def _identity_differences(observed, expected):
    names = set(observed) | set(expected)
    return {
        name: {'checkpoint': observed.get(name), 'requested': expected.get(name)}
        for name in sorted(names)
        if observed.get(name) != expected.get(name)
    }


def _optional_integer_list(value):
    if value is None:
        return None
    return np.asarray(value, dtype=int).tolist()


def _optional_integer_array(value):
    if value is None:
        return None
    return np.asarray(value, dtype=int)


__all__ = [
    'identity_fingerprint',
    'load_provider_factor_checkpoint',
    'provider_factor_identity',
    'save_provider_factor_checkpoint',
]
