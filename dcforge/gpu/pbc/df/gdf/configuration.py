r'''Experiment-facing configuration helpers for GPU real-Gamma GDF.

The helpers in this module expose numerical representation controls which are
already owned by PySCF/GPU4PySCF.  They do not invent a second GDF builder and
they do not change a provider unless the caller explicitly supplies an option.
In particular, ``partition_omega`` is the algorithmic split

.. math::

   r^{-1}=\operatorname{erfc}(\omega r)r^{-1}
          +\operatorname{erf}(\omega r)r^{-1},

not a physical range-separated Hamiltonian parameter.
'''

import numpy as np
from gpu4pyscf.pbc.df import rsdf_builder
from pyscf.pbc.df.rsdf_builder import (
    estimate_ke_cutoff_for_omega as estimate_source_ke_cutoff,
)

from dcforge.gpu.pbc.df.compatibility import make_auxcell

_UNSET = object()


def configure_gdf_candidate(
        provider, *, partition_omega=_UNSET, cell_precision=_UNSET,
        auxbasis=_UNSET, aux_metric_eigenvalue_threshold=_UNSET,
        exp_to_discard=_UNSET, device_residency=_UNSET,
        device_residency_fraction=_UNSET, transfer_profile=_UNSET):
    r'''Configure one opt-in GDF experiment before CDERI construction.

    Args:
        provider : dcforge.gpu.pbc.df.gdf.GDF
            Unbuilt GPU GDF provider.

    Kwargs:
        partition_omega : float or None
            Positive algorithmic SR/LR partition.  ``None`` restores the
            upstream system-dependent resolver.
        cell_precision : float
            Positive PySCF ``Cell.precision`` used by the source integral
            representation.  A provider receives an already-built ``Cell``;
            changing its precision here cannot rebuild geometry-dependent
            cutoffs such as ``Cell.rcut``.  Consequently this argument is
            accepted only as an equality assertion.  Set ``cell.precision``
            before ``cell.build()`` to run a different-precision candidate.
        auxbasis : str, dict, or None
            Auxiliary basis.  ``None`` requests PySCF's normal matching/ETB
            policy.
        aux_metric_eigenvalue_threshold : float
            Positive eigenvalue cutoff passed as GPU4PySCF
            ``linear_dep_threshold`` when the auxiliary Coulomb metric is
            factorized.
        exp_to_discard : float or None
            Optional PySCF auxiliary primitive exponent cutoff.
        device_residency : str
            ``'off'``, ``'auto'`` or ``'required'``.  See
            :mod:`dcforge.gpu.pbc.df.gdf.residency`.
        device_residency_fraction : float
            Requested fraction of auxiliary rows retained on device.  This is
            distinct from the free-VRAM budget fraction.
        transfer_profile : bool
            Enable synchronized CDERI load/unpack telemetry.  Profiling is
            diagnostics-only and may slow an SCF calculation.

    Returns:
        GDF
            The same provider, allowing PySCF-style chained setup.

    Notes:
        Omitted arguments are not the same as explicit ``None``.  This lets a
        caller restore automatic ``partition_omega`` or automatic ``auxbasis``
        without the helper silently resetting unrelated provider options.
        Result-defining options may not be changed after CDERI construction.
    '''
    _check_gdf_provider(provider)
    if getattr(provider, '_cderi', None) is not None:
        raise RuntimeError(
            'GDF candidate configuration must be set before CDERI build; '
            'call provider.reset() first')

    if partition_omega is not _UNSET:
        if partition_omega is None:
            provider.omega = None
        else:
            provider.omega = _positive_finite(
                partition_omega, 'partition_omega')
    if cell_precision is not _UNSET:
        value = _positive_finite(cell_precision, 'cell_precision')
        current = float(provider.cell.precision)
        if value != current:
            raise RuntimeError(
                'cell_precision cannot be changed on an already-built Cell; '
                'set Cell.precision before Cell.build() and construct a new '
                'RHFDFT/GDF provider')
    if auxbasis is not _UNSET:
        provider.auxbasis = auxbasis
        # ``preflight_cderi_memory`` may have constructed a temporary auxcell
        # without building CDERI.  Never retain it across an auxbasis change.
        provider.auxcell = None
    if aux_metric_eigenvalue_threshold is not _UNSET:
        provider.linear_dep_threshold = _positive_finite(
            aux_metric_eigenvalue_threshold,
            'aux_metric_eigenvalue_threshold')
    if exp_to_discard is not _UNSET:
        if exp_to_discard is None:
            provider.exp_to_discard = None
        else:
            value = float(exp_to_discard)
            if not np.isfinite(value) or value < 0.:
                raise ValueError(
                    'exp_to_discard must be non-negative and finite or None')
            provider.exp_to_discard = value
        provider.auxcell = None
    if device_residency is not _UNSET:
        provider.cderi_device_residency = _residency_mode(device_residency)
    if device_residency_fraction is not _UNSET:
        value = float(device_residency_fraction)
        if not np.isfinite(value) or not 0. < value <= 1.:
            raise ValueError('device_residency_fraction must be in (0,1]')
        provider.cderi_device_residency_fraction = value
    if transfer_profile is not _UNSET:
        provider.cderi_transfer_profile = bool(transfer_profile)
    return provider


def gdf_candidate_configuration(provider):
    r'''Return the requested and resolved GDF representation as plain data.

    This is a read-only planning helper.  It may construct the small PySCF
    auxiliary ``Cell`` witness when the provider has not yet been built, but
    it never constructs CDERI or modifies the provider.
    '''
    _check_gdf_provider(provider)
    auxcell = getattr(provider, 'auxcell', None)
    if auxcell is None:
        auxcell = make_auxcell(
            provider.cell, provider.auxbasis, provider.exp_to_discard)
    resolved = getattr(provider, 'resolved_omega', None)
    if resolved is None:
        from dcforge.gpu.pbc.df.gdf.gdf import resolve_partition_omega

        resolved = resolve_partition_omega(
            provider.cell, getattr(provider, 'omega', None))
    source_mesh, metric_mesh = _representation_meshes(
        provider.cell, auxcell, resolved)
    return {
        'partition_omega_requested': getattr(provider, 'omega', None),
        'partition_omega_resolved': float(resolved),
        'source_mesh': source_mesh.tolist(),
        'metric_mesh': metric_mesh.tolist(),
        'cell_precision': float(provider.cell.precision),
        'auxbasis': provider.auxbasis,
        'exp_to_discard': provider.exp_to_discard,
        'auxiliary_nao': int(auxcell.nao_nr()),
        'auxiliary_cell_precision': float(auxcell.precision),
        'auxiliary_metric_reciprocal_precision': float(
            auxcell.precision * 1.e-6),
        'aux_metric_eigenvalue_threshold': float(
            provider.linear_dep_threshold),
        'auxiliary_metric_integral_tolerance_owner': (
            'GPU4PySCF auxcell.precision * 1e-6'),
        'auxiliary_metric_integral_tolerance_configurable_per_provider': False,
        'device_residency': getattr(
            provider, 'cderi_device_residency', 'off'),
        'device_residency_fraction': float(getattr(
            provider, 'cderi_device_residency_fraction', 1.)),
        'transfer_profile': bool(getattr(
            provider, 'cderi_transfer_profile', False)),
    }


def resolved_response_settings(provider):
    r'''Return the exact partition recorded by the active source CDERI build.

    Analytic response must consume the representation which produced the
    source Hamiltonian.  Re-running GPU4PySCF's automatic omega resolver here
    is incorrect for an explicit candidate and can also drift if the upstream
    heuristic changes.  Missing build metadata therefore triggers the normal
    lazy source build.

    The central CDERI itself is not part of this response identity.  A
    gradient-only low-memory lifecycle deliberately releases ``_cderi`` after
    the last CPHF action, while retaining the built auxiliary cell and the
    resolved source/metric partition.  Rebuilding CDERI merely because the
    factor was released duplicates the most expensive source stage and is
    mathematically unnecessary for the primitive GDF nuclear response.
    '''
    _check_gdf_provider(provider)
    values = (
        getattr(provider, 'resolved_omega', None),
        getattr(provider, 'resolved_source_mesh', None),
        getattr(provider, 'resolved_metric_mesh', None),
    )
    if (getattr(provider, 'auxcell', None) is None
            or any(value is None for value in values)):
        provider.response_partition_lazy_build_count = int(getattr(
            provider, 'response_partition_lazy_build_count', 0)) + 1
        provider.build(j_only=False)
        values = (
            getattr(provider, 'resolved_omega', None),
            getattr(provider, 'resolved_source_mesh', None),
            getattr(provider, 'resolved_metric_mesh', None),
        )
    elif getattr(provider, '_cderi', None) is None:
        provider.response_partition_metadata_reuse_count = int(getattr(
            provider, 'response_partition_metadata_reuse_count', 0)) + 1
    if any(value is None for value in values):
        raise RuntimeError(
            'built GDF provider has no resolved source/response partition')
    omega = _positive_finite(values[0], 'resolved_omega')
    source_mesh = _positive_mesh(values[1], 'resolved_source_mesh')
    metric_mesh = _positive_mesh(values[2], 'resolved_metric_mesh')
    expected_source, expected_metric = _representation_meshes(
        provider.cell, provider.auxcell, omega)
    if not np.array_equal(source_mesh, expected_source):
        raise RuntimeError(
            'recorded GDF source mesh is inconsistent with resolved_omega: '
            f'{source_mesh.tolist()} != {expected_source.tolist()}')
    if not np.array_equal(metric_mesh, expected_metric):
        raise RuntimeError(
            'recorded GDF auxiliary-metric mesh is inconsistent with '
            f'resolved_omega: {metric_mesh.tolist()} != '
            f'{expected_metric.tolist()}')
    return omega, source_mesh, metric_mesh


def _representation_meshes(cell, auxcell, omega):
    physical_omega = abs(float(getattr(cell, 'omega', 0.)))
    if physical_omega < float(omega):
        source_cutoff = estimate_source_ke_cutoff(cell, omega)
        source_mesh = cell.symmetrize_mesh(
            cell.cutoff_to_mesh(source_cutoff))
        metric_cutoff = rsdf_builder.estimate_ke_cutoff_for_omega(
            auxcell, omega, auxcell.precision * 1.e-6)
        metric_mesh = auxcell.cutoff_to_mesh(metric_cutoff)
    else:
        source_mesh = metric_mesh = np.ones(3, dtype=int)
    return (
        np.asarray(source_mesh, dtype=int).copy(),
        np.asarray(metric_mesh, dtype=int).copy(),
    )


def _check_gdf_provider(provider):
    if getattr(provider, 'backend_name', None) != 'gdf':
        raise TypeError('GDF configuration requires a dcforge GDF provider')


def _positive_finite(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0.:
        raise ValueError(f'{name} must be positive and finite')
    return value


def _positive_mesh(value, name):
    value = np.asarray(value, dtype=int)
    if value.shape != (3,) or np.any(value <= 0):
        raise ValueError(f'{name} must contain three positive integers')
    return value.copy()


def _residency_mode(value):
    value = str(value).strip().lower()
    if value not in ('off', 'auto', 'required'):
        raise ValueError(
            "device_residency must be 'off', 'auto', or 'required'")
    return value


__all__ = [
    'configure_gdf_candidate',
    'gdf_candidate_configuration',
    'resolved_response_settings',
]
