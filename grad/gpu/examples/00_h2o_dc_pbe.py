"""Compute a molecular GPU DC-PBE gradient from a density-fitted RHF state."""

from gpu4pyscf.scf import hf as gpu_hf
from pyscf import gto

from gpu4pyscf_dcdft_grad import RDCGradient


mol = gto.M(
    atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
    basis="def2-svp",
)
hf = gpu_hf.RHF(mol).density_fit().run()
gradient = RDCGradient(hf, xc="PBE").kernel()
print(gradient)
