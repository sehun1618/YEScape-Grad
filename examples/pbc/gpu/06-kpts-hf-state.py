'''Save or restore a GPU sampled-k FFTDF source for a later gradient.

Run ``source`` and ``gradient`` as separate processes::

    python 06-kpts-hf-state.py source /tmp/gpu-kpts-hf.chk
    python 06-kpts-hf-state.py gradient /tmp/gpu-kpts-hf.chk
'''

import argparse

import numpy as np
from pyscf.pbc import gto

from dcforge.gpu.pbc.energy import KRHFDFT

parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=('source', 'gradient'))
parser.add_argument('checkpoint')
args = parser.parse_args()

if args.mode == 'source':
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
        cell, kpts=kpts, kpts_wrap_around=True, xc='PBE')
    mf.conv_tol = 1e-9
    mf.conv_tol_grad = 1e-7
    mf.kernel_hf()
    mf.save_hf_state(args.checkpoint)
    print('source KRHF energy [Ha]:', mf.e_hf)
    print('checkpoint:', args.checkpoint)
else:
    mf = KRHFDFT.from_hf_state(args.checkpoint, xc='PBE')
    energy = mf.kernel(run_hf=False)
    mf_grad = mf.nuc_grad_method()
    gradient = np.asarray(mf_grad.kernel())
    print('target PBE@KRHF energy [Ha]:', energy)
    print('gradient [Ha/Bohr]:\n', gradient)
    print('restore validation:', mf.source_validation.to_dict())
