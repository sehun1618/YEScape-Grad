'''PBE-r2SCAN: evaluate r2SCAN on the stationary PBE density.'''

from pyscf import gto
from dcforge.gpu.energy import RNSC

mol = gto.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
            basis='def2-svp', verbose=4)
# source_xc also accepts LDA,VWN, TPSS, r2SCAN, B3LYP and CAM-B3LYP.
mf = RNSC(mol, xc='r2SCAN', source_xc='PBE').density_fit()
mf.conv_tol = 1e-8
mf.conv_tol_grad = 1e-6
mf.mf_source.grids.level = 4
mf.grids.level = 4
energy = mf.kernel()
gradient = mf.nuc_grad_method().kernel()
