"""Write one GPU RSDF DC-PBE frame with exact optimized stress owners."""

from pathlib import Path

import numpy as np
from pyscf.pbc import gto

from gpu4pyscf_pbc_dcdft import dft
from gpu4pyscf_pbc_dcdft_stress import gpu_energy_force_stress
from pyscf_pbc_dcdft_stress import write_mlip_extxyz, write_mlip_json


cell = gto.Cell(
    atom="H 0 0 0; H 0 0 1.4", a=np.eye(3)*5.0, unit="Bohr",
    basis="gth-szv", pseudo="gth-hf", mesh=np.asarray((75, 75, 75)), verbose=0,
).build()
method = dft.DCRKS(cell, xc="PBE").rs_density_fit(auxbasis="gth-szv")
method.conv_tol = 1e-10
method.kernel()

result = gpu_energy_force_stress(
    method,
    pp_response_backend="gpu4pyscf_native_contracted_analytic_nonlocal_device",
    df_response_backend="host_analytic_source_adjoint_gpu_ft",
)
output = Path("stress_output_optimized")
write_mlip_json(result.record, output/"gpu_frame.json")
write_mlip_extxyz(result.record, output/"gpu_frame.xyz")
print(result.record.stress_tensile_ha_per_bohr3)
