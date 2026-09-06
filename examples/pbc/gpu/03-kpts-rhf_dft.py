'''GPU sampled-k GDF plus XC-only MultiGrid PBE@KRHF gradient.'''

import numpy as np
from pyscf.pbc import gto

from dcforge.gpu.pbc.energy import KRHFDFT

cell = gto.Cell()
cell.atom = 'H 0 0 0; H 0 0 1.4'
cell.a = np.eye(3) * 6.0
cell.unit = 'Bohr'
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pbe'
cell.precision = 1e-7
cell.ke_cutoff = 30
cell.verbose = 0
cell.build()


# The ndarray is the explicit complete Brillouin-zone mesh.  This first GPU
# slice supports equal weights only.  PySCF/GPU4PySCF apply the 1/nk energy
# normalization inside KRHF/KRKS; do not multiply the result a second time.
kpts = cell.make_kpts([2, 1, 1], wrap_around=True)
mf = KRHFDFT(
    cell,
    kpts=kpts,
    kpts_wrap_around=True,
    xc='PBE',
).density_fit(auxbasis='gth-szv').multigrid_numint(with_j=False).run()
mf_grad = mf.nuc_grad_method()
gradient = mf_grad.kernel()
# Replace ``density_fit(...)`` by ``fft_density_fit()`` to use FFTDF.  Both
# providers share the full-BZ Z-vector/Pulay algebra.  GDF contracts its
# nuclear response to per-atom scalars through GPU4PySCF and records that
# private-kernel boundary in ``gdf_response_metadata``.
# ``multigrid_numint(with_j=False)`` moves only target XC to MultiGrid.
# Source J/K, target J, and CPHF remain on the selected FFTDF/GDF provider.

print('k-points [1/Bohr]:\n', mf.kpts)
print('k-point weights:', mf.kpts_weights)
print('DF provider:', type(mf.with_df).__name__)
print('source KRHF energy [Ha]:', mf.e_hf)
print('target PBE@KRHF energy [Ha]:', mf.e_dft)
print('global sampled-k HF gap [Ha]:', mf.source_gap)
print('PBE@KRHF gradient [Ha/Bohr]:\n', gradient)
print('Z-vector method:', mf_grad.zvec_method_used)
print('provider response:', mf_grad.provider_response_backend)
print('target XC backend:', mf.multigrid_backend)
print('scientific host boundaries:', mf.host_fallback_components)
