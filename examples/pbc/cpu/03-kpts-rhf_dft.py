'''CPU full-BZ sampled-k GDF PBE@KRHF energy and gradient.'''

import numpy as np
from pyscf.pbc import gto

from dcforge.cpu.pbc.energy import KRHFDFT

cell = gto.Cell()
cell.atom = 'H 0 0 0; H 0 0 1.4'
cell.a = np.eye(3) * 6.0
cell.unit = 'Bohr'
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pbe'
cell.precision = 1e-7
cell.mesh = [15, 15, 15]
cell.verbose = 0
cell.build()


# An explicit ndarray represents the complete Brillouin-zone mesh.  This
# first slice supports equal weights only; PySCF applies the 1/nk factor in
# its KRHF/KRKS energy expressions.
kpts = cell.make_kpts([2, 1, 1], wrap_around=True)
mf = KRHFDFT(
    cell,
    kpts=kpts,
    kpts_wrap_around=True,
    xc='PBE',
).density_fit(auxbasis='gth-szv').run()
# Omit density_fit(), or call fft_density_fit(), for the FFTDF route.  In
# either case source KRHF, target Hartree, and CPHF share the same provider.
mf_grad = mf.nuc_grad_method()
gradient = mf_grad.kernel()

print('k-points [1/Bohr]:\n', mf.kpts)
print('k-point weights:', mf.kpts_weights)
print('DF provider:', type(mf.with_df).__name__)
print('source KRHF energy [Ha]:', mf.e_hf)
print('target PBE@KRHF energy [Ha]:', mf.e_dft)
print('global sampled-k HF gap [Ha]:', mf.source_gap)
print('PBE@KRHF gradient [Ha/Bohr]:\n', gradient)
print('Z-vector method:', mf_grad.zvec_method_used)
print('Z-vector physical residual:', mf_grad.z_residual_max_abs)
print('GDF central CDERI parity:', mf.with_df.cderi_value_parity_max_abs)
