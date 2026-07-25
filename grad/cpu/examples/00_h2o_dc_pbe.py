"""Compute a molecular DC-PBE energy gradient with an RHF density."""

from pathlib import Path
import sys

from pyscf import gto, scf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pyscf_dcdft_grad import dc_grad


mol = gto.M(
    atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
    basis="sto-3g",
)
hf = scf.RHF(mol).run()
gradient = dc_grad(hf, xc="PBE").kernel()
print(gradient)
