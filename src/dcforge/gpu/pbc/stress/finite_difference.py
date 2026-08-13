'''GPU namespace for backend-independent finite-strain energy oracles.'''

from dcforge.cpu.pbc.stress.finite_difference import (
    DEFAULT_STRAIN_STEPS,
    finite_difference_strain,
    frozen_density_finite_difference,
    relaxed_energy_finite_difference,
)

__all__ = [
    'DEFAULT_STRAIN_STEPS',
    'finite_difference_strain',
    'frozen_density_finite_difference',
    'relaxed_energy_finite_difference',
]
