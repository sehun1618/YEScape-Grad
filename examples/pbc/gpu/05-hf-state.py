'''Save or restore a GPU Gamma-point GDF source for a later gradient.

Run the two commands in separate processes::

    python 05-hf-state.py source /tmp/gpu-pbc-hf.chk
    python 05-hf-state.py gradient /tmp/gpu-pbc-hf.chk

MO arrays are stored in a PySCF-compatible host checkpoint.  CDERI and other
GPU process-local state are rebuilt during restore; source RHF SCF is not.
'''

import argparse

import numpy as np
from pyscf.pbc import gto

from dcforge.gpu.pbc.energy import RHFDFT

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=('source', 'gradient'))
parser.add_argument('checkpoint')
args = parser.parse_args()

if args.mode == 'source':
    cell = gto.Cell()
    cell.atom = 'O 0 0 0; H 0 1.43 1.10; H 0 -1.43 1.10'
    cell.a = np.eye(3) * 7.0
    cell.unit = 'Bohr'
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-hf'
    cell.ke_cutoff = 40
    cell.precision = 1e-7
    cell.verbose = 0
    cell.build()

    mf = RHFDFT(cell, xc='PBE').density_fit(auxbasis='gth-szv')
    mf.conv_tol = 1e-9
    mf.conv_tol_grad = 1e-7
    mf.kernel_hf()
    mf.save_hf_state(args.checkpoint)
    print('source RHF energy [Ha]:', mf.e_hf)
    print('checkpoint:', args.checkpoint)
else:
    mf = RHFDFT.from_hf_state(args.checkpoint, xc='PBE')
    energy = mf.kernel(run_hf=False)
    mf_grad = mf.nuc_grad_method()
    gradient = np.asarray(mf_grad.kernel())
    print('target PBE@RHF energy [Ha]:', energy)
    print('gradient [Ha/Bohr]:\n', gradient)
    print('restore validation:', mf.source_validation.to_dict())
