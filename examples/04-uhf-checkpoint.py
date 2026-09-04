'''Save a PySCF UHF state, restore it, and evaluate an HF-DFT gradient.'''

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from pyscf import gto, scf
from pyscf.scf import chkfile

from dcforge.cpu.energy import UHFDFT

ROOT = Path(__file__).resolve().parents[1]


def run_source(mol, directory, with_df):
    '''Run source UHF and save the canonical state used by the gradient.'''
    mf_hf = scf.UHF(mol)
    if with_df:
        mf_hf = mf_hf.density_fit(auxbasis=None, only_dfj=False)
    mf_hf.conv_tol = 1e-12
    mf_hf.conv_tol_grad = 1e-10
    mf_hf.max_cycle = 100

    energy = mf_hf.kernel()
    if not mf_hf.converged:
        raise RuntimeError('source UHF calculation did not converge')

    checkpoint = directory / 'hf.chk'
    density = directory / 'Dsc.npy'
    chkfile.dump_scf(
        mol, str(checkpoint), energy,
        mf_hf.mo_energy, mf_hf.mo_coeff, mf_hf.mo_occ,
        overwrite_mol=True,
    )
    np.save(density, mf_hf.make_rdm1())
    return checkpoint, density


def restore_workflow(checkpoint, density, with_df):
    '''Restore and validate a source state before allowing ``run_hf=False``.'''
    mol, state = chkfile.load_scf(str(checkpoint))

    if with_df:
        # A standard PySCF SCF checkpoint does not record the DF provider.
        # Reconstruct the same representation before attaching the MO state.
        mf = UHFDFT(mol, xc='PBE').density_fit(
            auxbasis=None, only_dfj=False)
        mf_hf = mf.mf_hf
    else:
        mf_hf = scf.UHF(mol)
        mf = UHFDFT(mol, xc='PBE', mf_hf=mf_hf)

    mf_hf.e_tot = float(state['e_tot'])
    mf_hf.mo_energy = np.asarray(state['mo_energy'])
    mf_hf.mo_coeff = np.asarray(state['mo_coeff'])
    mf_hf.mo_occ = np.asarray(state['mo_occ'])

    dm = mf_hf.make_rdm1(mf_hf.mo_coeff, mf_hf.mo_occ)
    s = mf_hf.get_ovlp(mol)
    hcore = mf_hf.get_hcore(mol)
    vhf = mf_hf.get_veff(mol, dm)
    fock = mf_hf.get_fock(hcore, s, vhf, dm)

    density_error = float(np.max(np.abs(dm - np.load(density))))
    orth_error = max(
        float(np.max(np.abs(c.conj().T @ s @ c - np.eye(c.shape[1]))))
        for c in mf_hf.mo_coeff
    )
    electron_error = float(np.max(np.abs(
        np.einsum('sij,ji->s', dm, s).real - np.asarray(mol.nelec)
    )))
    energy_error = abs(float(mf_hf.energy_tot(dm, hcore, vhf)) - mf_hf.e_tot)
    fock_residual = max(
        float(np.max(np.abs(f @ c - s @ (c * e))))
        for f, c, e in zip(fock, mf_hf.mo_coeff, mf_hf.mo_energy)
    )

    if density_error > 1e-12:
        raise RuntimeError(f'restored density mismatch: {density_error:.3e}')
    if orth_error > 1e-9:
        raise RuntimeError(f'restored MO orthogonality error: {orth_error:.3e}')
    if electron_error > 1e-8:
        raise RuntimeError(f'restored electron-count error: {electron_error:.3e}')
    if energy_error > 1e-9:
        raise RuntimeError(f'restored source-energy error: {energy_error:.3e}')
    if fock_residual > 1e-7:
        raise RuntimeError(f'restored canonical Fock residual: {fock_residual:.3e}')

    # PySCF's SCF checkpoint contains canonical arrays but no convergence flag.
    # Mark the state reusable only after the independent checks above pass.
    mf_hf.converged = True
    return mf, {
        'density_error': density_error,
        'orth_error': orth_error,
        'electron_error': electron_error,
        'energy_error': energy_error,
        'fock_residual': fock_residual,
    }


mol = gto.M(
    atom=str(ROOT / 'structures' / 'oh.xyz'),
    unit='Angstrom',
    basis='aug-cc-pvtz',
    spin=1,
    verbose=0,
)
np.set_printoptions(precision=12, suppress=True)

for label, with_df in (('conventional', False), ('DF', True)):
    with TemporaryDirectory(prefix='dcforge-uhf-checkpoint-') as tmp:
        checkpoint, density = run_source(mol, Path(tmp), with_df)
        mf, checks = restore_workflow(checkpoint, density, with_df)

        mf.grids.level = 4
        energy = mf.kernel(run_hf=False)
        mf_grad = mf.nuc_grad_method().set(
            grid_response=True,
            cphf_conv_tol=1e-9,
            cphf_max_cycle=200,
            fock_deriv_method='direct',
        )
        if with_df:
            mf_grad.auxbasis_response = True
        gradient = mf_grad.kernel()

        print(f'\n[{label} checkpoint restore]')
        print(f'Source UHF energy = {mf.e_uhf:.15f} Ha')
        print(f'Target PBE@UHF energy = {energy:.15f} Ha')
        print('Gradient dE/dR [Ha/Bohr]')
        print(gradient)
        print(f'Density error = {checks["density_error"]:.3e}')
        print(f'MO orthogonality error = {checks["orth_error"]:.3e}')
        print(f'Electron-count error = {checks["electron_error"]:.3e}')
        print(f'Source-energy error = {checks["energy_error"]:.3e} Ha')
        print(f'Canonical Fock residual = {checks["fock_residual"]:.3e}')
        print(f'Z residual max abs = {mf_grad.z_residual_max_abs:.3e}')
