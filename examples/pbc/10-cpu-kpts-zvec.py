'''Inspect the CPU sampled-k dense real-linear KRHF Z-vector oracle.'''

import numpy as np
from pyscf.pbc import gto

from dcforge.cpu.pbc.energy import KRHFDFT
from dcforge.cpu.pbc.grad.krhf_zvec import (
    gen_hf_response,
    get_complex_linearity_error,
    make_fvo,
    make_hessian_operator,
    make_mo_partitions,
    solve_z_vector,
    solve_z_vector_matrix_free,
)

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

kpts = cell.make_kpts([2, 1, 1], wrap_around=True)
mf = KRHFDFT(
    cell, kpts=kpts, xc='PBE', kpts_wrap_around=True).set(
    conv_tol=1e-10,
    max_cycle=80,
).run()


#
# Case 1: build F^PBE_vo and solve A_R^T z_R = f_R
#
# make_mo_partitions keeps one complex Co/Cv/energy-gap block per k-point.
# gen_hf_response delegates the coupled full-k J-K/2 action to PySCF KRHF.
# solve_z_vector explicitly materializes the small real-expanded Hessian and
# is an accuracy oracle, not the later large-system production solver.
#
Co, Cv, e_ai = make_mo_partitions(
    mf.mo_energy, mf.mo_coeff, mf.mo_occ)
fvo = make_fvo(mf, Co, Cv)
fvind = gen_hf_response(mf.mf_hf)
zvec, z_info = solve_z_vector(
    mf.mo_energy, mf.mo_occ, Co, Cv, fvo, fvind,
    residual_tol=1e-10,
    dense_oracle_size=512,
    equation='adjoint',
    kpts_weights=mf.kpts_weights,
)

print('\n[CPU sampled-k dense adjoint]')
print(f'E_PBE[P_KRHF] = {mf.e_dft:.15f} Ha')
print(f'VO shapes = {[block.shape for block in fvo]}')
print(f'Energy-gap shapes = {[block.shape for block in e_ai]}')
print(f'max |F^PBE_vo| = '
      f'{max(np.max(np.abs(block)) for block in fvo):.9e} Ha')
print(f'Real-linear dimension = {z_info["dimension"]}')
print(f'Z physical residual = {z_info["residual_max_abs"]:.9e}')
print(f'Hessian condition number = {z_info["condition_number"]:.9e}')
print(f'Hessian asymmetry = '
      f'{z_info["matrix_asymmetry_max_abs"]:.9e}')
print(f'Direct/adjoint solution difference = '
      f'{z_info["direct_adjoint_max_abs"]:.9e}')
print('Z blocks')
for k, block in enumerate(zvec):
    print(f'k={k}:\n{block}')


#
# Case 2: use the K4 matrix-free production solver
#
# This path never materializes A_R.  GMRES sees a real LinearOperator,
# D_ai^-1 is the preconditioner, and a fresh unpreconditioned action decides
# whether the returned Z-vector is physically acceptable.
#
zvec_mf, z_info_mf = solve_z_vector_matrix_free(
    mf.mo_energy, mf.mo_occ, Co, Cv, fvo, fvind,
    conv_tol=1e-9, residual_tol=1e-8, max_cycle=200,
    kpts_weights=mf.kpts_weights)
z_difference = max(
    np.max(np.abs(actual - reference))
    for actual, reference in zip(zvec_mf, zvec, strict=True))

print('\n[CPU sampled-k matrix-free adjoint]')
print(f'Solver = {z_info_mf["solver"]}')
print(f'Iterations = {z_info_mf["iterations"]}')
print(f'Operator calls = {z_info_mf["operator_calls"]}')
print(f'Physical residual = {z_info_mf["residual_max_abs"]:.9e}')
print(f'Self-adjoint probe = '
      f'{z_info_mf["self_adjoint_probe_error"]:.9e}')
print(f'Dense/matrix-free max difference = {z_difference:.9e}')
print(f'Dense Hessian bytes = {z_info_mf["dense_matrix_bytes"]}')
print(f'Krylov workspace estimate = '
      f'{z_info_mf["workspace_estimate_bytes"]} bytes')


#
# Case 3: demonstrate why a complex-linear solver is insufficient
#
# Because D_X = Cv X Co^H + h.c. contains X and X*, A(iX) generally differs
# from i A(X).  Real scalar linearity remains exact and is the correct vector
# space used by the dense transpose solve.
#
vind = make_hessian_operator(
    mf.mo_energy, mf.mo_occ, Co, Cv, fvind)
rng = np.random.default_rng(41)
x = tuple(
    rng.normal(size=block.shape) + 1j * rng.normal(size=block.shape)
    for block in fvo
)
complex_defect = get_complex_linearity_error(vind, x)

print('\n[Real-linear diagnostic]')
print(f'max |A(iX)-iA(X)| = {complex_defect:.9e}')
