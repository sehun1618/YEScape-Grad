'''CPU full-BZ sampled-k FFTDF/GDF PBE@KRHF energy and gradient.'''

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


#
# Case 1: explicit full-BZ 2x1x1 FFTDF PBE@KRHF energy
#
# cell.make_kpts returns Cartesian reciprocal vectors in 1/Bohr.  DC-Forge's
# first sampled-k release requires this explicit full-BZ ndarray and assigns
# equal weights 1/nkpts.  kpts_wrap_around records how this array was generated
# but does not regenerate or modify the supplied coordinates.
#
kpts = cell.make_kpts([2, 1, 1], wrap_around=True)
mf = KRHFDFT(
    cell, kpts=kpts, xc='PBE', kpts_wrap_around=True).set(
    conv_tol=1e-10,
    conv_tol_grad=1e-8,
    max_cycle=80,
).run()
mf_grad = mf.nuc_grad_method().set(
    zvec_method='auto', cphf_conv_tol=1e-9,
    cphf_residual_tol=1e-8, cphf_max_cycle=200)
gradient = mf_grad.kernel()

print('\n[CPU full-BZ FFTDF]')
print('k-points [1/Bohr]')
print(mf.kpts)
print(f'k-point weights = {mf.kpts_weights}')
print(f'Source KRHF energy = {mf.e_hf:.15f} Ha')
print(f'Target PBE@KRHF energy = {mf.e_dft:.15f} Ha')
print(f'Global sampled-k HF gap = {mf.source_gap:.9f} Ha')
print(f'Density shape = {mf.dm_hf.shape}, dtype = {mf.dm_hf.dtype}')
print(f'Target Fock shape = {mf.target_fock.shape}')
print('Gradient dE/dR [Ha/Bohr]')
print(gradient)
print('Force -dE/dR [Ha/Bohr]')
print(-gradient)
print(f'Z-vector route = {mf_grad.zvec_method_used}')
print(f'Z physical residual = {mf_grad.z_residual_max_abs:.9e}')
print(f'Term recomposition = {mf_grad.term_recomposition_max_abs:.9e}')


#
# Case 2: the same full-BZ calculation with Gaussian density fitting
#
# density_fit() changes both source KRHF and target Hartree to one shared GDF
# provider.  PySCF's matching auxiliary basis is used when auxbasis is omitted;
# spelling it explicitly, as below, is useful for reproducible comparisons.
# The analytic gradient differentiates every complex (ki,kj) CDERI pair,
# including separate three-center-source, auxiliary-metric and Ewald-exchange
# overlap terms.
#
mf_gdf = KRHFDFT(
    cell, kpts=kpts, xc='PBE', kpts_wrap_around=True,
).density_fit(auxbasis='gth-szv').set(
    conv_tol=1e-10,
    conv_tol_grad=1e-8,
    max_cycle=80,
).run()
mf_gdf_grad = mf_gdf.nuc_grad_method().set(
    zvec_method='auto', cphf_conv_tol=1e-9,
    cphf_residual_tol=1e-8, cphf_max_cycle=200)
gradient_gdf = mf_gdf_grad.kernel()

print('\n[CPU full-BZ GDF]')
print(f'Auxiliary basis = {mf_gdf.with_df.auxbasis}')
print(f'Source KRHF energy = {mf_gdf.e_hf:.15f} Ha')
print(f'Target PBE@KRHF energy = {mf_gdf.e_dft:.15f} Ha')
print('Gradient dE/dR [Ha/Bohr]')
print(gradient_gdf)
print(f'CDERI central parity = '
      f'{mf_gdf.with_df.cderi_value_parity_max_abs:.9e}')
print(f'Z physical residual = {mf_gdf_grad.z_residual_max_abs:.9e}')
print(f'Term recomposition = '
      f'{mf_gdf_grad.term_recomposition_max_abs:.9e}')


#
# Case 3: range-separated Gaussian density fitting
#
# RSDF has two independent numerical splits.  omega/mesh_compact define the
# three-center source, while omega_j2c/mesh_j2c define the auxiliary metric.
# They are resolved by PySCF for the active Cell and must not be replaced by
# one common setting in the energy or derivative.
#
mf_rsdf = KRHFDFT(
    cell, kpts=kpts, xc='PBE', kpts_wrap_around=True,
).rs_density_fit(auxbasis='gth-szv').set(
    conv_tol=1e-10,
    conv_tol_grad=1e-8,
    max_cycle=80,
).run()
mf_rsdf_grad = mf_rsdf.nuc_grad_method().set(
    zvec_method='auto', cphf_conv_tol=1e-9,
    cphf_residual_tol=1e-8, cphf_max_cycle=200)
gradient_rsdf = mf_rsdf_grad.kernel()

print('\n[CPU full-BZ RSDF]')
print(f'Resolved settings = {mf_rsdf.with_df.response_settings()}')
print(f'Source KRHF energy = {mf_rsdf.e_hf:.15f} Ha')
print(f'Target PBE@KRHF energy = {mf_rsdf.e_dft:.15f} Ha')
print('Gradient dE/dR [Ha/Bohr]')
print(gradient_rsdf)
print(f'CDERI central parity = '
      f'{mf_rsdf.with_df.cderi_value_parity_max_abs:.9e}')
print(f'Z physical residual = {mf_rsdf_grad.z_residual_max_abs:.9e}')


#
# Case 4: explicit one-Gamma KRHFDFT energy
#
# This remains the k-resolved API and is useful as the exact sampled-k route
# -> existing RHFDFT Gamma limiting check.  The public gradient call is the
# same for one or many explicit full-BZ k-points.
#
mf_gamma = KRHFDFT(
    cell,
    kpts=np.zeros((1, 3)),
    xc='PBE',
).run()
gradient_gamma = mf_gamma.nuc_grad_method().kernel()

print('\n[CPU explicit one-Gamma KRHFDFT]')
print(f'Target PBE@KRHF energy = {mf_gamma.e_dft:.15f} Ha')
print('Gradient dE/dR [Ha/Bohr]')
print(gradient_gamma)
