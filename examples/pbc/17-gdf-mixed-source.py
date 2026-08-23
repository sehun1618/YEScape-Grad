'''Guarded mixed-precision source RHF with automatic CDERI residency.'''

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

# ``auto`` keeps the central FP64 CDERI in mapped host memory only when it and
# the declared FP32 sidecar fit while preserving the configured reserve.
mf = RHFDFT(cell, xc='R2SCAN').density_fit(auxbasis=None)
mf.with_df.cderi_storage = 'auto'
mf.with_df.cderi_auto_memory_fraction = .8
mf.with_df.cderi_auto_minimum_reserve_bytes = 16 * 1024**3

# The first SCF J/K call already uses the FP32 factor.  The operator advances
# monotonically through an adaptive-K tier and finishes with native FP64 GDF.
# ``energy_normalization`` is explicit because the package must not infer that
# every three atoms form one water molecule.  Use the number of waters for a
# water-only production Cell.
mf.source_mixed_precision(
    block_size=256,
    copy_block_size=256,
    energy_normalization=1,
    adaptive_norm_fraction=.9,
    fallback=True,
)
mf.conv_tol = 1e-8
mf.conv_tol_grad = 1e-5
mf.max_cycle = 80

# Only target XC uses multigrid.  Source RHF, target J and response remain GDF.
mf = mf.multigrid_numint(with_j=False)
mf.run()

# Analytic response precision is independent of source-SCF precision.  A
# real-Gamma GDF gradient now defaults to guarded Z3 mixed response, but this
# example selects the canonical FP64 Z arithmetic explicitly to isolate the
# source-SCF experiment.
mf_grad = mf.nuc_grad_method().set(
    precision_policy='fp64',
    cphf_conv_tol=1e-9,
    cphf_residual_tol=1e-8,
).run()

print(f'E_R2SCAN@HF = {mf.e_dft:.15f} Ha')
print(f'max |gradient| = {np.max(np.abs(mf_grad.de)):.6e} Ha/Bohr')
print('source precision requested/effective =',
      mf.source_precision_policy, mf.source_precision_policy_effective)
print('CDERI requested/resolved =',
      mf.with_df.cderi_storage, mf.with_df.cderi_storage_resolved)
print('CDERI auto plan =', mf.with_df.cderi_auto_plan)
print('source mixed-precision report =', mf.source_precision_report)
