"""Machine-readable GPU capability and route-ownership report.

A ``True`` value is a dispatch promise: the corresponding numerical work is
owned by GPU4PySCF or this package and is not silently delegated to the CPU
DC-DFT implementation.  FFTDF, GDF, and RSDF energy/gradient routes satisfy that
promise for restricted Gamma-point and sampled-k-point DC-DFT calculations.
Native GDF owns central CDERI, two-electron response, and local-part-1
one-electron response on the GPU.  Local-part-2 and the reference GDF/RSDF
primitive builders remain explicit host setup boundaries.
"""

import gpu4pyscf
import pyscf


def capability() -> dict[str, object]:
    return {
        "device": "gpu",
        "fftdf_energy": True,
        "fftdf_analytic_gradient": True,
        "gdf_energy": True,
        "gdf_analytic_gradient": True,
        "rsdf_energy": True,
        "rsdf_analytic_gradient": True,
        "scan_energy": True,
        "scan_analytic_gradient": True,
        "r2scan_energy": True,
        "r2scan_analytic_gradient": True,
        "validated_target_functionals": ["PBE", "SCAN", "R2SCAN"],
        "validated_df_backends": ["FFTDF", "GDF", "RSDF"],
        "optional_dispersion": {
            "d2c_d4": {
                "energy": True,
                "analytic_gradient": True,
                "electronic_structure_device": "gpu",
                "dispersion_library_device": "cpu",
                "density_response": False,
            }
        },
        "target_integrators": {
            "uniform": {
                "energy": True,
                "analytic_gradient": True,
                "target_j_owner": "selected_df_backend",
                "target_xc_owner": "gpu_numint",
            },
            "multigrid_xc": {
                "energy": True,
                "analytic_gradient": True,
                "target_j_owner": "selected_df_backend",
                "target_xc_owner": "gpu_multigrid_v2",
            },
            "multigrid_j_xc": {
                "energy": True,
                "analytic_gradient": True,
                "target_j_owner": "gpu_multigrid_v2",
                "target_xc_owner": "gpu_multigrid_v2",
            },
        },
        "dependency_versions": {
            "pyscf": pyscf.__version__,
            "gpu4pyscf": gpu4pyscf.__version__,
        },
        "gamma": True,
        "sampled_kpoints": True,
        "status": "stage_7_target_multigrid_opt_in_validated",
        "silent_cpu_fallback": False,
        "host_boundaries": [
            "PySCF Cell metadata and reciprocal-grid setup",
            "native GDF mapped-host CDERI storage and bookkeeping",
            "native GDF local-part-2 pseudopotential H1 primitive generation",
            "sampled-k GDF metric derivative upstream host pbc_intor calls",
            "reference GDF central and derivative primitive generation",
            "RSDF central/derivative source and metric integral setup",
            "small GTH radial/projector table setup",
            "Ewald nuclear-repulsion gradient",
            "final scalar/array reporting",
            "optional periodic D2C-D4 energy/gradient library",
        ],
        "route_ownership": {
            "hf_scf": "GPU4PySCF RHF/KRHF with project-local GPU FFTDF/GDF/RSDF J/K",
            "target_pbe": "GPU4PySCF RKS/KRKS + GPU KNumInt + GPU FFTDF/GDF/RSDF J/K",
            "target_dfa": "GPU4PySCF RKS/KRKS + GPU KNumInt + GPU FFTDF/GDF/RSDF J/K",
            "target_multigrid_xc": (
                "GPU4PySCF multigrid_v2 XC + selected FFTDF/GDF/RSDF J"
            ),
            "target_multigrid_j_xc": (
                "GPU4PySCF multigrid_v2 combined target J+XC"
            ),
            "fftdf_full_jk_derivative": "gpu4pyscf_pbc_dcdft.df.fft",
            "gdf_full_jk_derivative": "gpu4pyscf_pbc_dcdft.df.gdf",
            "gdf_native_cderi_and_jk_response": (
                "GPU4PySCF CUDA + gpu4pyscf_pbc_dcdft.df.gdf"
            ),
            "gdf_native_local_part1_derivative": (
                "gpu4pyscf_pbc_dcdft.df.gdf.local_part1_derivative"
            ),
            "gdf_local_part2_derivative": (
                "pyscf_pbc_dcdft.df.gdf.local_part2 host primitive"
            ),
            "gdf_reference_full_response": "gpu4pyscf_pbc_dcdft.df.gdf",
            "rsdf_metric_cderi_response": "gpu4pyscf_pbc_dcdft.df.rsdf",
            "rsdf_full_jk_derivative": "gpu4pyscf_pbc_dcdft.df.rsdf",
            "nonlocal_gth_derivative": "gpu4pyscf_pbc_dcdft.pseudo",
            "hf_ax_zvector": "gpu4pyscf_pbc_dcdft.grad",
            "final_gradient": "gpu4pyscf_pbc_dcdft.grad.fftdf/gdf/rsdf",
            "d2c_d4": "pyscf_pbc_dcdft.dispersion host library boundary",
        },
    }
