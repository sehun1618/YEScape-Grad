"""Compute a molecular GPU DC-PBE gradient from a density-fitted RHF state."""

from pathlib import Path
import sys

from gpu4pyscf.scf import hf as gpu_hf
from pyscf import gto

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gpu4pyscf_dcdft_grad import RDCGradient


mol = gto.M(
    atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
    basis="def2-svp",
)
hf = gpu_hf.RHF(mol).density_fit().run()
gradient = RDCGradient(
    hf, xc="PBE", response_backend="mo_direct", share_df=True,
    factorized_z_f1=True,
).kernel()
print(gradient)
