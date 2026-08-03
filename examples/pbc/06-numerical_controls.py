'''Distinguish reciprocal-representation and CPHF convergence.'''

import numpy as np
from pyscf.pbc import gto

from dcforge.cpu.pbc.energy import RHFDFT


def make_cell(ke_cutoff):
    '''Build one Cell after assigning its kinetic-energy cutoff.'''
    cell = gto.Cell()
    cell.atom = '''
        O  0.10 -0.20  0.30
        H  1.55  0.15  0.05
        H -0.35  1.25 -0.25
    '''
    cell.a = np.eye(3) * 7.
    cell.unit = 'Bohr'
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-hf'
    cell.ke_cutoff = ke_cutoff
    # Hold precision fixed so that Case 1 changes one numerical input.
    cell.precision = 1e-8
    cell.verbose = 0
    return cell.build()


#
# Case 1: converge the finite reciprocal representation
#
# cell.ke_cutoff changes cell.mesh and can change both energy and gradient.
# A new Cell and workflow are constructed for every cutoff.
#
records = []
for ke_cutoff in (100., 200., 300.):
    mf = RHFDFT(make_cell(ke_cutoff), xc='PBE').set(
        conv_tol=1e-10,
        conv_tol_grad=1e-8,
        max_cycle=80,
    ).run()
    mf_grad = mf.nuc_grad_method().set(
        cphf_conv_tol=1e-10,
        cphf_residual_tol=1e-10,
        cphf_max_cycle=200,
    ).run()
    records.append({
        'ke_cutoff': ke_cutoff,
        'mf': mf,
        'mesh': mf.cell.mesh.copy(),
        'energy': mf.e_dft,
        'gradient': mf_grad.de.copy(),
    })

reference = records[-1]
print('Representation convergence: cell.precision = 1e-8 is fixed')
print('cutoff(Ha)  mesh          E_PBE@HF(Ha)      delta E(uHa)  max delta g')
for record in records:
    delta_energy = (record['energy'] - reference['energy']) * 1e6
    delta_gradient = np.max(np.abs(
        record['gradient'] - reference['gradient']))
    print(f'{record["ke_cutoff"]:10.1f}  '
          f'{record["mesh"].tolist()!s:12s}  '
          f'{record["energy"]:16.10f}  {delta_energy:12.4f}  '
          f'{delta_gradient:.3e}')


#
# Case 2: converge CPHF on one fixed Cell, mesh, and source-HF state
#
# cphf_conv_tol is the requested primary-solver tolerance.  The independently
# reconstructed physical equation must also pass cphf_residual_tol.
#
mf = reference['mf']
cphf_records = []
for conv_tol in (1e-6, 1e-9, 1e-12):
    mf_grad = mf.nuc_grad_method().set(
        cphf_conv_tol=conv_tol,
        cphf_residual_tol=1e-10,
        cphf_max_cycle=200,
    ).run()
    cphf_records.append((
        conv_tol,
        mf_grad.de.copy(),
        mf_grad.z_solver,
        mf_grad.z_residual_max_abs,
    ))

reference_gradient = cphf_records[-1][1]
print('\nResponse convergence: representation and residual gate are fixed')
print('requested tol  solver                  physical residual  max delta g')
for conv_tol, gradient, solver, residual in cphf_records:
    delta_gradient = np.max(np.abs(gradient - reference_gradient))
    print(f'{conv_tol:13.1e}  {solver:22s}  {residual:17.3e}  '
          f'{delta_gradient:.3e}')
