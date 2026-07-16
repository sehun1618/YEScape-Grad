"""Compute a molecular DC-PBE energy gradient with an RHF density."""

from pyscf import gto, scf

from pyscf_dcdft_grad import dc_grad


mol = gto.M(
    atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587",
    basis="sto-3g",
)
hf = scf.RHF(mol).run()
gradient = dc_grad(hf, xc="PBE").kernel()
print(gradient)
