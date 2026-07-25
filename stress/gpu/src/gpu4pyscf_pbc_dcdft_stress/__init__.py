"""GPU periodic DC-DFT analytic stress."""

from .analytic import GPUAnalyticStressResult, gpu_analytic_stress
from .contracted_df_response import ContractedDFStrain, contracted_gpu_df_strain
from .pp_response import (
    ContractedPseudopotentialStrain,
    contracted_gpu4pyscf_pp_strain,
)
from .workflow import gpu_energy_force_stress

__version__ = "0.1.0a0"


def capability() -> dict[str, object]:
    """Report the current explicit host-primitive/device-response boundary."""

    return {
        "stage": "S7_analytic_stress",
        "analytic_stress": True,
        "production": True,
        "finite_difference_stress": False,
        "cpu_fallback": False,
        "source_hf": "gpu",
        "target_fock_xc": "gpu",
        "z_vector": "gpu",
        "df_cholesky_jk_response": "gpu",
        "production_opt_in_pp_response_backend":
            "gpu4pyscf_native_contracted_analytic_nonlocal_device",
        "production_opt_in_df_response_backend":
            "host_analytic_source_adjoint_gpu_ft",
        "strain_integral_primitives": "host_analytic_or_gpu_contracted_pp",
        "experimental_contracted_primitives": [
            "gpu4pyscf_native_contracted",
            "gpu_fixed_density_fd_contracted",
        ],
        "integral_engine_boundary_declared": True,
    }


__all__ = [
    "ContractedPseudopotentialStrain",
    "ContractedDFStrain",
    "GPUAnalyticStressResult",
    "__version__",
    "capability",
    "contracted_gpu4pyscf_pp_strain",
    "contracted_gpu_df_strain",
    "gpu_analytic_stress",
    "gpu_energy_force_stress",
]
