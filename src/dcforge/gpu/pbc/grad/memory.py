'''Calculation-wide GPU GDF gradient/stress memory policies.'''

from dcforge.gpu.pbc.df.gdf import GDF

_POLICIES = ('custom', 'fast', 'balanced', 'low_memory')
_PROPERTIES = ('gradient', 'stress')
_GIB = 1024**3


def resolve_memory_policy(mf_grad, required_property=None):
    '''Resolve one visible policy into the primitive GDF lifecycle controls.

    The policy changes storage and execution only; all choices evaluate the
    same GDF operator and derivative ledger.  ``custom`` preserves individual
    PySCF-style attributes exactly.  Other policies are authoritative and
    report every primitive value they select.

    Args:
        mf_grad : dcforge.gpu.pbc.grad.rhf_dft.Gradients
            Active derivative driver.

    Kwargs:
        required_property : {'gradient', 'stress'} or None
            Property about to execute.  A non-custom policy fails closed if
            the declared property sequence omitted it, preventing an early
            CDERI release from being followed by a hidden rebuild.

    Returns:
        dict
            JSON-serializable requested/resolved policy ledger.
    '''
    policy = str(mf_grad.memory_policy).strip().lower()
    if policy not in _POLICIES:
        raise ValueError(
            f'memory_policy must be one of {_POLICIES}, got {policy!r}')
    sequence = tuple(str(name).strip().lower()
                     for name in mf_grad.property_sequence)
    if not sequence or any(name not in _PROPERTIES for name in sequence):
        raise ValueError(
            'property_sequence must contain gradient and/or stress')
    if len(set(sequence)) != len(sequence):
        raise ValueError('property_sequence contains a duplicate property')
    if (required_property is not None and policy != 'custom'
            and required_property not in sequence):
        raise RuntimeError(
            f'memory_policy={policy!r} was planned for {sequence}, but '
            f'{required_property!r} was requested; declare the complete '
            'gradient/stress lifecycle before starting the calculation')

    provider = mf_grad.base.with_df
    if isinstance(provider, GDF) and policy != 'custom':
        if policy == 'fast':
            mf_grad.df_cphf_response_cache = 'all'
            mf_grad.df_cphf_response_memory_fraction = .90
            mf_grad.df_cphf_response_minimum_reserve_bytes = 8 * _GIB
            provider.adjoint_storage = 'host_cache'
            provider.adjoint_host_cache = 'mapped'
            provider.adjoint_host_memory_fraction = .90
            provider.adjoint_host_minimum_reserve_bytes = 64 * _GIB
            provider.release_cderi_after_response = False
        elif policy == 'balanced':
            mf_grad.df_cphf_response_cache = 'auto'
            mf_grad.df_cphf_response_memory_fraction = .80
            mf_grad.df_cphf_response_minimum_reserve_bytes = 8 * _GIB
            provider.adjoint_storage = 'host_cache'
            provider.adjoint_host_cache = 'mapped'
            provider.adjoint_host_memory_fraction = .80
            provider.adjoint_host_minimum_reserve_bytes = 64 * _GIB
            provider.release_cderi_after_response = False
        else:
            mf_grad.df_cphf_response_cache = 'recompute'
            mf_grad.df_cphf_response_memory_fraction = .70
            mf_grad.df_cphf_response_minimum_reserve_bytes = 12 * _GIB
            provider.adjoint_storage = 'double_pass'
            provider.adjoint_host_cache = 'mapped'
            provider.adjoint_host_memory_fraction = .70
            provider.adjoint_host_minimum_reserve_bytes = 64 * _GIB
            # Gradient-only work may release central factors after the Z
            # action.  A declared stress consumer keeps them alive so stress
            # never triggers an unreported provider rebuild.
            provider.release_cderi_after_response = sequence == ('gradient',)

    report = {
        'requested_policy': policy,
        'resolved_policy': policy,
        'property_sequence': list(sequence),
        'required_property': required_property,
        'provider': type(provider).__name__,
        'algebra': 'unchanged',
        'primitives': {
            'df_cphf_response_mode': str(
                mf_grad.df_cphf_response_mode),
            'df_cphf_response_cache': str(
                mf_grad.df_cphf_response_cache),
            'df_cphf_response_memory_fraction': float(
                mf_grad.df_cphf_response_memory_fraction),
            'df_cphf_response_minimum_reserve_bytes': int(
                mf_grad.df_cphf_response_minimum_reserve_bytes),
            'adjoint_storage': getattr(provider, 'adjoint_storage', None),
            'adjoint_host_cache': getattr(
                provider, 'adjoint_host_cache', None),
            'adjoint_host_memory_fraction': getattr(
                provider, 'adjoint_host_memory_fraction', None),
            'adjoint_host_minimum_reserve_bytes': getattr(
                provider, 'adjoint_host_minimum_reserve_bytes', None),
            'release_cderi_after_response': bool(getattr(
                provider, 'release_cderi_after_response', False)),
        },
    }
    mf_grad.resolved_memory_policy = report
    return report


__all__ = ['resolve_memory_policy']
