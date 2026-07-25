"""Validate all six Gamma/GDF DC-PBE stress components by central FD."""

from pathlib import Path
import sys

import numpy as np
from pyscf.pbc import gto


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "stress" / "cpu" / "src"))
sys.path.insert(0, str(ROOT / "pbc" / "cpu" / "src"))

from pyscf_pbc_dcdft_stress import (  # noqa: E402
    FDStress,
    FDStressConfig,
    GammaGDFEnergyEvaluator,
)


cell = gto.Cell(
    atom="H 0 0 0; H 0 0 1.4", a=np.eye(3) * 5.0,
    unit="Bohr", basis="gth-szv", pseudo="gth-hf",
    mesh=np.asarray((9, 9, 9)), verbose=0,
).build()
evaluator = GammaGDFEnergyEvaluator(
    xc="PBE", auxbasis="gth-szv", conv_tol=1e-10, verbose=0
)
result = FDStress(evaluator, FDStressConfig(step=5e-4)).kernel(cell)

print("Central-FD stress (Eh/Bohr^3):")
print(result.stress_tensor_ha_per_bohr3)
print("Central-FD stress (GPa):")
print(result.stress_tensor_gpa)
print(f"Pressure: {result.pressure_ha_per_bohr3:.12e} Eh/Bohr^3")
