'''GPU namespace for backend-independent homogeneous-strain algebra.'''

# These helpers manipulate only Cell metadata and small NumPy tensors.  A
# single implementation prevents CPU/GPU shear, sign and unit conventions
# from drifting while numerical GPU stress primitives remain device-native.
from dcforge.cpu.pbc.stress.strain import (
    HARTREE_PER_BOHR3_TO_GPA,
    VOIGT_LABELS,
    assert_same_strain_representation,
    deform_cell,
    fractional_coordinates,
    pressure_from_stress,
    raw_strain_basis,
    snapshot_strain_state,
    strain_tensor_to_voigt,
    strain_voigt_to_tensor,
    stress_from_work,
    stress_tensor_to_voigt,
    stress_to_gpa,
    stress_voigt_to_tensor,
    symmetrize_tensor,
    virial_from_stress,
    voigt_strain_basis,
)

__all__ = [
    'HARTREE_PER_BOHR3_TO_GPA',
    'VOIGT_LABELS',
    'assert_same_strain_representation',
    'deform_cell',
    'fractional_coordinates',
    'pressure_from_stress',
    'raw_strain_basis',
    'snapshot_strain_state',
    'strain_tensor_to_voigt',
    'strain_voigt_to_tensor',
    'stress_from_work',
    'stress_tensor_to_voigt',
    'stress_to_gpa',
    'stress_voigt_to_tensor',
    'symmetrize_tensor',
    'virial_from_stress',
    'voigt_strain_basis',
]
