"""Write one CPU GDF DC-PBE energy/force/stress MLIP frame."""

from pathlib import Path

import numpy as np
from pyscf.pbc import gto
from pyscf_pbc_dcdft import dft
from pyscf_pbc_dcdft_stress import (
    cpu_energy_force_stress,
    write_mlip_extxyz,
    write_mlip_json,
)


cell = gto.Cell(
    atom="H 0 0 0; H 0 0 1.4", a=np.eye(3)*5.0, unit="Bohr",
    basis="gth-szv", pseudo="gth-hf", mesh=np.asarray((9, 9, 9)), verbose=0,
).build()
method = dft.DCRKS(cell, xc="PBE").density_fit(auxbasis="gth-szv")
method.conv_tol = 1e-10
method.kernel()

result = cpu_energy_force_stress(method)
output = Path("stress_output")
write_mlip_json(result.record, output/"cpu_frame.json")
write_mlip_extxyz(result.record, output/"cpu_frame.xyz")
print(result.record.stress_tensile_ha_per_bohr3)

