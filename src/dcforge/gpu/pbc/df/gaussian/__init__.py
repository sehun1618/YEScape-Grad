'''Shared GPU real-Gamma Gaussian-provider analytic primitives.'''

from dcforge.gpu.pbc.df.gaussian.ewald import (
    cross_derivative as ewald_cross_derivative,
)
from dcforge.gpu.pbc.df.gaussian.ewald import (
    signed_derivative as ewald_signed_derivative,
)
from dcforge.gpu.pbc.df.gaussian.hcore import contract_hcore_pair
from dcforge.gpu.pbc.df.gaussian.local_part1 import (
    LocalPart1Derivative,
    get_pp_loc_part1,
    resolve_pp_loc_part1,
)
from dcforge.gpu.pbc.df.gaussian.nonlocal_pseudo import (
    contract_nonlocal_pseudo_batch,
)
from dcforge.gpu.pbc.df.gaussian.pp_strain import (
    contract_gdf_pseudopotential_strain,
    contract_local_part1_strain,
    contract_local_part2_strain,
)
from dcforge.gpu.pbc.df.gaussian.pseudo import (
    PP_LOC_PART1_KEYS,
    get_pp,
    get_pp_loc_part1_plan,
    init_pp_loc_part1,
    reset_pp_loc_part1,
)
from dcforge.gpu.pbc.df.gaussian.reciprocal import (
    ReciprocalBlockExecution,
    ReciprocalBlockPlan,
    execute_gamma_reciprocal_blocks,
    gamma_conjugate_reciprocal_indices,
    plan_gamma_reciprocal_blocks,
)
from dcforge.gpu.pbc.df.gaussian.strain import (
    contract_kinetic_strain,
    contract_overlap_strain,
)

__all__ = [
    'PP_LOC_PART1_KEYS',
    'LocalPart1Derivative',
    'ReciprocalBlockExecution',
    'ReciprocalBlockPlan',
    'contract_gdf_pseudopotential_strain',
    'contract_hcore_pair',
    'contract_kinetic_strain',
    'contract_local_part1_strain',
    'contract_local_part2_strain',
    'contract_nonlocal_pseudo_batch',
    'contract_overlap_strain',
    'ewald_cross_derivative',
    'ewald_signed_derivative',
    'execute_gamma_reciprocal_blocks',
    'gamma_conjugate_reciprocal_indices',
    'get_pp',
    'get_pp_loc_part1',
    'get_pp_loc_part1_plan',
    'init_pp_loc_part1',
    'plan_gamma_reciprocal_blocks',
    'reset_pp_loc_part1',
    'resolve_pp_loc_part1',
]
