"""Compute an open-shell molecular GPU DC-PBE gradient with a UHF density."""

from pathlib import Path
import sys

from gpu4pyscf.scf import uhf as gpu_uhf
from pyscf import gto

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gpu4pyscf_dcdft_grad import UDCGradient


mol = gto.M(
    atom="O 0 0 0; H 0 0 0.97",
    basis="def2-svp",
    spin=1,
)
hf = gpu_uhf.UHF(mol).density_fit().run()
gradient = UDCGradient(
    hf, xc="PBE", response_backend="mo_direct", share_df=True,
    direct_z_f1=True,
).kernel()
print(gradient)
