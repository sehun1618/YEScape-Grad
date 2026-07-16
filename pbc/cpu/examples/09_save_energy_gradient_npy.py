"""Save analytic gradient as NPY and calculation metadata as JSON."""

import json
from pathlib import Path

import numpy as np
from pyscf.pbc import gto

from pyscf_pbc_dcdft import dft


output_directory = Path("dc_pbe_result")
output_directory.mkdir(exist_ok=True)

cell = gto.Cell(
    atom="H 0 0 0; H 0 0 1.4",
    a=np.eye(3) * 6.0,
    unit="Bohr",
    basis="gth-szv",
    pseudo="gth-hf",
    ke_cutoff=20.0,
    verbose=0,
).build()
mf = dft.DCRKS(cell, xc="PBE")
energy = mf.kernel()
gradient_object = mf.nuc_grad_method()
gradient = gradient_object.kernel()

np.save(output_directory / "gradient_ha_per_bohr.npy", gradient)
metadata = {
    "energy_ha": float(energy),
    "gradient_file": "gradient_ha_per_bohr.npy",
    "gradient_units": "Ha/Bohr",
    "calculation": mf.metadata(),
    "gradient": gradient_object.result.metadata(),
}
(output_directory / "metadata.json").write_text(
    json.dumps(metadata, indent=2) + "\n",
    encoding="utf-8",
)
print(output_directory.resolve())
