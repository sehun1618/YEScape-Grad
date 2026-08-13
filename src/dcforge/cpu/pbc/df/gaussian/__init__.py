'''Common periodic Gaussian density-fitting response primitives.'''

from dcforge.cpu.pbc.df.gaussian.adjoint import (
    contract_response_adjoints,
    make_cderi_observable_adjoints,
    push_cderi_adjoints,
)
from dcforge.cpu.pbc.df.gaussian.cderi import (
    cholesky_cderi,
    factor_metric,
    grad_cderi,
    grad_cderi_all,
    grad_cderi_with_factor,
    solve_cderi,
    solve_cderi_with_factor,
)
from dcforge.cpu.pbc.df.gaussian.fake_nuclear import (
    fake_nuclear_refuniq,
    make_fake_nuclear_cell,
)
from dcforge.cpu.pbc.df.gaussian.jk import (
    grad_ewald_exchange,
    grad_ewald_exchange_kpts,
    grad_j_energy,
    grad_j_energy_kpts,
    grad_jk,
    grad_jk_kpts,
)
from dcforge.cpu.pbc.df.gaussian.metric import (
    g0_metric,
    grad_long_range_metric,
    long_range_metric,
)
from dcforge.cpu.pbc.df.gaussian.pseudo import (
    contract_vpploc_part2_strain,
    contract_vppnl_strain,
)
from dcforge.cpu.pbc.df.gaussian.source import (
    grad_three_center_source,
    three_center_source,
)
from dcforge.cpu.pbc.df.gaussian.source_long_range import (
    grad_long_range_source,
    long_range_source,
)
from dcforge.cpu.pbc.df.gaussian.source_short_range import (
    grad_short_range_source,
    short_range_source,
)
from dcforge.cpu.pbc.df.gaussian.storage import load_cderi
from dcforge.cpu.pbc.df.gaussian.strain import (
    ao_pair_fourier_strain,
    auxiliary_fourier_strain,
    contract_one_electron_strain,
    kinetic_strain,
    matrix_strain_to_voigt,
    one_electron_strain,
    overlap_strain,
    weighted_coulomb_strain,
)
from dcforge.cpu.pbc.df.gaussian.vbar import (
    grad_vbar_source,
    vbar_source,
)

__all__ = [
    'ao_pair_fourier_strain',
    'auxiliary_fourier_strain',
    'cholesky_cderi',
    'contract_one_electron_strain',
    'contract_response_adjoints',
    'contract_vpploc_part2_strain',
    'contract_vppnl_strain',
    'factor_metric',
    'fake_nuclear_refuniq',
    'g0_metric',
    'grad_cderi',
    'grad_cderi_all',
    'grad_cderi_with_factor',
    'grad_ewald_exchange',
    'grad_ewald_exchange_kpts',
    'grad_j_energy',
    'grad_j_energy_kpts',
    'grad_jk',
    'grad_jk_kpts',
    'grad_long_range_metric',
    'grad_long_range_source',
    'grad_short_range_source',
    'grad_three_center_source',
    'grad_vbar_source',
    'kinetic_strain',
    'load_cderi',
    'long_range_metric',
    'long_range_source',
    'make_cderi_observable_adjoints',
    'make_fake_nuclear_cell',
    'matrix_strain_to_voigt',
    'one_electron_strain',
    'overlap_strain',
    'push_cderi_adjoints',
    'short_range_source',
    'solve_cderi',
    'solve_cderi_with_factor',
    'three_center_source',
    'vbar_source',
    'weighted_coulomb_strain',
]
