# CPU PBC DC-DFT Stress

This package is the isolated CPU reference for periodic DC-DFT stress. It does
not add methods or conditionals to the existing energy/gradient packages.

The S1 route is a central finite-difference oracle for restricted Gamma-point
GDF calculations. It holds fractional atomic coordinates and the integer
integration mesh fixed for every strain pair.

S2 is available as a component-resolved development API. It keeps the
central source-HF AO density fixed and provides:

- a GDF component-FD oracle for kinetic, pseudopotential, Coulomb, XC, Ewald,
  and their total;
- analytic kinetic, uniform-grid XC, and package-owned 3D Ewald strain;
- an explicit capability list that still marks GDF pseudopotential, GDF
  Coulomb, overlap/Lagrangian, and orbital response as missing.

PySCF's current PP stress primitive uses an FFT representation. It is exposed
only as a diagnostic because it is not the derivative of the GDF target hcore.

The complete restricted DC-PBE/DC-r2SCAN GDF/RSDF API is analytic for Gamma
and sampled k-points:

```python
from pyscf_pbc_dcdft_stress import gamma_gdf_analytic_stress

stress = gamma_gdf_analytic_stress(converged_dc_method)
print(stress.stress_tensor_ha_per_bohr3)
```

This production capability label describes equation completeness, not current
CPU performance. The first image-resolved reference kernels remain slower than
full FD on the tiny H2/H2O validation systems.

Energy, force and stress can be derived from one converged state with one
Z-vector and written as an MLIP record:

```python
from pyscf_pbc_dcdft_stress import (
    cpu_energy_force_stress,
    write_mlip_extxyz,
    write_mlip_json,
)

result = cpu_energy_force_stress(converged_dc_method)
write_mlip_json(result.record, "frame.json")
write_mlip_extxyz(result.record, "frame.xyz")
```

The record stores force as `-gradient`, tensile-positive stress and
`virial = -volume * stress` with explicit units.

```python
from pyscf_pbc_dcdft_stress import FDStress, FDStressConfig
from pyscf_pbc_dcdft_stress.adapters import GammaGDFEnergyEvaluator

evaluator = GammaGDFEnergyEvaluator(
    xc="PBE", auxbasis="gth-szv", conv_tol=1e-11
)
result = FDStress(
    evaluator, FDStressConfig(step=5e-4)
).kernel(cell)
print(result.stress_tensor_ha_per_bohr3)
```

FD stress is validation code, not the intended production route.

The partial S2 result is likewise not a production total:

```python
from pyscf_pbc_dcdft_stress import frozen_density_analytic_components

partial = frozen_density_analytic_components(converged_dc_method)
print(partial.stress_ha_per_bohr3["xc"])
print(partial.missing_components)
```
