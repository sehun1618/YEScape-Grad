'''Inspect the production GPU real-Gamma GDF acceleration policy.'''

import numpy as np
from pyscf.pbc import gto

from dcforge.gpu.pbc.energy import RHFDFT

cell = gto.Cell()
cell.atom = 'O 0 0 0; H 0 1.43 1.10; H 0 -1.43 1.10'
cell.a = np.eye(3) * 7.0
cell.unit = 'Bohr'
cell.basis = 'gth-szv'
cell.pseudo = 'gth-hf'
cell.ke_cutoff = 100
cell.precision = 1e-8
cell.build()


mf = RHFDFT(cell, xc='R2SCAN').density_fit(auxbasis=None)
mf = mf.multigrid_numint(with_j=False)

# The source Hamiltonian stays FP64 by default.  The real-Gamma gradient uses
# the guarded Z3 mixed response, followed by an independent physical-residual
# check.  For a complete FP64 reference, retain the workflow's ``fp64`` profile
# and set both gradient controls before calling ``run``:
#
# mf_grad.precision_policy = 'fp64'
# mf_grad.gdf_derivative_precision = 'fp64'
mf.run()
mf_grad = mf.nuc_grad_method()
gradient = mf_grad.run().de

print('HF-r2SCAN energy [Ha]:', mf.e_dft)
print('Gradient [Ha/Bohr]:\n', gradient)
print('GDF precision profile:', mf.gdf_precision_profile_effective)
print('Z response precision:', mf_grad.precision_policy_effective)
print('GDF response mode:', mf_grad.gdf_response_mode)
print('GDF adjoint backend:', mf_grad.gdf_adjoint_backend_used)
print('Physical Z residual:', mf_grad.z_residual_max_abs)
print('Terminal Z response reused:', mf_grad.terminal_response_reuse_used)
print('CDERI build memory plan:', mf.with_df.native_build_memory_plan)

# Approximate Hamiltonians are explicit opt-ins and emit warnings.  ``mixed``
# quantizes only completed FP64-built CDERI blocks; ``fp32_build`` also lowers
# the large CDERI transformation.  FP64 arithmetic over either stored factor
# cannot recover discarded bits.
# mf.set_gdf_precision_profile('mixed')
# mf.set_gdf_precision_profile('fp32_build')
# Set these BEFORE mf.run() when using a mixed source profile:
# mf.source_precision_coarse_max_cycle = 20  # First-tier residence budget.
# mf.source_precision_fine_max_cycle = 20    # Second-tier residence budget.
# Residual gates may promote earlier; None disables an individual budget.
# The final FP64-arithmetic tier still has to meet the SCF/audit tolerances.
# They require independent energy/force validation on representative data.
