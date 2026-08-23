'''Reference, moderate and aggressive GDF precision profiles.'''

import numpy as np
from pyscf.pbc import gto

from dcforge.gpu.pbc.energy import RHFDFT

cell = gto.Cell()
cell.atom = '''
    O  0.10 -0.20  0.30
    H  1.55  0.15  0.05
    H -0.35  1.25 -0.25
'''
cell.a = np.eye(3) * 7.
cell.unit = 'Bohr'
cell.basis = 'gth-dzvp'
cell.pseudo = 'gth-hf'
cell.ke_cutoff = 100
cell.precision = 1e-8
cell.verbose = 4
cell.build()

mf = RHFDFT(cell, xc='R2SCAN').density_fit(auxbasis=None)
mf.with_df.cderi_storage = 'mapped_host'

# Case 1: reference (default).
# profile = 'fp64'

# Case 2: moderate acceleration.  Native primitive integrals, metric
# factorization and CDERI transformation remain FP64.  Only the completed
# CDERI is stored in FP32; source SCF, Z-vector and derivative-source products
# use guarded mixed arithmetic.  Validate a representative production subset
# against Case 1 before generating MLIP labels.
profile = 'mixed'

# Case 3: aggressive experimental acceleration.  The large CDERI metric
# transformation and reciprocal contraction additionally use FP32/complex64.
# SCF/CPHF convergence does not measure the resulting Hamiltonian bias.  This
# mode requires an explicit energy/force error study against Case 1 and must
# never be treated as an automatically accuracy-controlled replacement.
# profile = 'fp32_build'

mf.set_gdf_precision_profile(
    profile,
    block_size=256,
    energy_normalization=1,
)
mf.conv_tol = 1e-8
mf.conv_tol_grad = 1e-5
mf.max_cycle = 80

# Source HF, target Hartree and response use GDF.  Only target XC uses MG.
mf = mf.multigrid_numint(with_j=False)
mf.run()
# Both accelerated workflow profiles select the default Z3 mixed response.
# It uses a separate physical residual gate and retains a finite low-precision
# solution as an exact-fallback seed.  To compare only the Z-vector arithmetic
# on the same already-built Hamiltonian, set ``precision_policy='fp64'`` on
# ``mf_grad`` below.  To recover a true native-FP64 Hamiltonian, instead set
# the complete ``profile = 'fp64'`` above and rebuild source SCF/CDERI.
mf_grad = mf.nuc_grad_method().set(
    cphf_conv_tol=1e-9,
    cphf_residual_tol=1e-8,
).run()

print(f'profile requested = {mf.gdf_precision_profile}')
print(f'profile effective = {mf.gdf_precision_profile_effective}')
print(f'E_HF = {mf.e_hf:.15f} Ha')
print(f'E_R2SCAN@HF = {mf.e_dft:.15f} Ha')
print(f'max |gradient| = {np.max(np.abs(mf_grad.de)):.6e} Ha/Bohr')
print('precision report =', mf.gdf_precision_report)
print('gradient precision =', mf_grad.precision_policy_effective,
      mf_grad.gdf_derivative_precision)
