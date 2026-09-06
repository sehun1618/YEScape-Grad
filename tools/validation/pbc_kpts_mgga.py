#!/usr/bin/env python
'''Validate sampled-k pure tau-MGGA gradients and finite-mesh eggbox error.'''

import argparse
import json
import platform
from pathlib import Path
import sys

import numpy as np
import pyscf
from pyscf import lib
from pyscf.pbc import gto

# Direct execution otherwise resolves whichever same-named checkout happens
# to be installed in the active environment.  Validation must exercise the
# repository containing this script.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

import dcforge
from dcforge.cpu.pbc.energy import KRHFDFT


WATER_COORDS = np.asarray([
    [0.10, -0.20, 0.30],
    [1.55, 0.15, 0.05],
    [-0.35, 1.25, -0.25],
])


def make_cell(coords, mesh):
    '''Return the asymmetric Water1 validation Cell.'''
    cell = gto.Cell()
    cell.atom = [
        (symbol, coordinate)
        for symbol, coordinate in zip(('O', 'H', 'H'), coords, strict=True)
    ]
    cell.a = np.eye(3) * 7.0
    cell.unit = 'Bohr'
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pbe'
    cell.precision = 1e-10
    cell.mesh = [mesh] * 3
    cell.verbose = 0
    cell.build()
    return cell


def run_targets(coords, mesh, provider, functionals, transform='identity'):
    '''Run one source KRHF and evaluate all target functionals on its density.'''
    cell = make_cell(coords, mesh)
    kpts = cell.make_kpts([2, 1, 1], wrap_around=True)
    if transform == 'reverse':
        kpts = kpts[::-1]
    elif transform == 'time_reversal':
        kpts = -kpts
    elif transform != 'identity':
        raise ValueError(f'unknown k-point transform {transform!r}')

    targets = {}
    first = KRHFDFT(
        cell, kpts=kpts, xc=functionals[0], kpts_wrap_around=True)
    if provider == 'gdf':
        first.density_fit(auxbasis='gth-szv')
    elif provider != 'fft':
        raise ValueError("provider must be 'fft' or 'gdf'")
    first.conv_tol = 1e-12
    first.conv_tol_grad = 1e-10
    first.max_cycle = 100
    first.kernel()
    targets[functionals[0]] = first

    for xc in functionals[1:]:
        target = KRHFDFT(
            cell, kpts=kpts, xc=xc, mf_hf=first.mf_hf,
            kpts_wrap_around=True)
        target.kernel(run_hf=False)
        targets[xc] = target
    return targets


def validate_gradients(functionals, providers, mesh, step):
    r'''Compare all :math:`3N` analytic coordinates with relaxed-source FD.'''
    results = {}
    for provider in providers:
        print(f'[gradient] central {provider}, mesh={mesh}', flush=True)
        central = run_targets(
            WATER_COORDS, mesh, provider, functionals)
        analytic = {}
        diagnostics = {}
        for xc, mf in central.items():
            print(f'[gradient] analytic {provider}/{xc}', flush=True)
            mf_grad = mf.nuc_grad_method()
            analytic[xc] = np.asarray(mf_grad.kernel())
            diagnostics[xc] = {
                'z_residual_max_abs': mf_grad.z_residual_max_abs,
                'term_recomposition_max_abs':
                    mf_grad.term_recomposition_max_abs,
                'imaginary_residual_max_abs':
                    mf_grad.term_imaginary_residual_max_abs,
                'translation_residual':
                    np.asarray(mf_grad.translation_residual).tolist(),
            }

        numerical = {
            xc: np.zeros_like(analytic[xc]) for xc in functionals}
        for atom in range(len(WATER_COORDS)):
            for axis in range(3):
                print(
                    f'[gradient] FD {provider}, atom={atom}, axis={axis}',
                    flush=True)
                plus = WATER_COORDS.copy()
                minus = WATER_COORDS.copy()
                plus[atom, axis] += step
                minus[atom, axis] -= step
                targets_plus = run_targets(
                    plus, mesh, provider, functionals)
                targets_minus = run_targets(
                    minus, mesh, provider, functionals)
                for xc in functionals:
                    numerical[xc][atom, axis] = (
                        targets_plus[xc].e_dft - targets_minus[xc].e_dft
                    ) / (2 * step)

        for xc in functionals:
            error = analytic[xc] - numerical[xc]
            results[f'{provider}/{xc}'] = {
                'energy_hartree': central[xc].e_dft,
                'analytic_gradient_hartree_per_bohr': analytic[xc].tolist(),
                'fd_gradient_hartree_per_bohr': numerical[xc].tolist(),
                'gradient_error_rms_hartree_per_bohr':
                    float(np.sqrt(np.mean(error**2))),
                'gradient_error_max_hartree_per_bohr':
                    float(np.max(np.abs(error))),
                **diagnostics[xc],
            }
    return results


def validate_symmetry(functionals, providers, mesh):
    '''Check full-BZ ordering and time-reversal invariance.'''
    results = {}
    for provider in providers:
        print(f'[symmetry] reference {provider}, mesh={mesh}', flush=True)
        reference = run_targets(
            WATER_COORDS, mesh, provider, functionals)
        candidates = {}
        for transform in ('reverse', 'time_reversal'):
            print(f'[symmetry] {provider}/{transform}', flush=True)
            candidates[transform] = run_targets(
                WATER_COORDS, mesh, provider, functionals, transform)
        reference_gradient = {
            xc: np.asarray(mf.nuc_grad_method().kernel())
            for xc, mf in reference.items()
        }
        for transform, targets in candidates.items():
            for xc, mf in targets.items():
                gradient = np.asarray(mf.nuc_grad_method().kernel())
                results[f'{provider}/{xc}/{transform}'] = {
                    'energy_max_abs_hartree': abs(
                        mf.e_dft - reference[xc].e_dft),
                    'gradient_max_abs_hartree_per_bohr': float(np.max(
                        np.abs(gradient - reference_gradient[xc]))),
                }
    return results


def validate_eggbox(functionals, providers, meshes, step):
    '''Compare the analytic translation sum with a rigid-translation FD.'''
    results = {}
    for mesh in meshes:
        for provider in providers:
            print(f'[eggbox] {provider}, mesh={mesh}', flush=True)
            central = run_targets(
                WATER_COORDS, mesh, provider, functionals)
            displaced = {}
            for axis in range(3):
                displacement = np.zeros_like(WATER_COORDS)
                displacement[:, axis] = step
                displaced[axis] = (
                    run_targets(
                        WATER_COORDS + displacement, mesh, provider,
                        functionals),
                    run_targets(
                        WATER_COORDS - displacement, mesh, provider,
                        functionals),
                )
            for xc, mf in central.items():
                mf_grad = mf.nuc_grad_method()
                gradient = np.asarray(mf_grad.kernel())
                analytic = gradient.sum(axis=0)
                numerical = np.zeros(3)
                for axis in range(3):
                    axis_plus, axis_minus = displaced[axis]
                    numerical[axis] = (
                        axis_plus[xc].e_dft - axis_minus[xc].e_dft
                    ) / (2 * step)
                results[f'{provider}/{xc}/mesh_{mesh}'] = {
                    'energy_hartree': mf.e_dft,
                    'gradient_hartree_per_bohr': gradient.tolist(),
                    'analytic_translation_sum_hartree_per_bohr':
                        analytic.tolist(),
                    'rigid_translation_fd_hartree_per_bohr':
                        numerical.tolist(),
                    'analytic_fd_max_abs_hartree_per_bohr': float(np.max(
                        np.abs(analytic - numerical))),
                    'eggbox_max_abs_hartree_per_bohr':
                        float(np.max(np.abs(analytic))),
                }
    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--phase', nargs='+', choices=('gradient', 'symmetry', 'eggbox'),
        default=('gradient', 'symmetry', 'eggbox'))
    parser.add_argument(
        '--xc', nargs='+', default=('TPSS', 'SCAN', 'M06-L'))
    parser.add_argument(
        '--provider', nargs='+', choices=('fft', 'gdf'),
        default=('fft', 'gdf'))
    parser.add_argument('--mesh', type=int, default=21)
    parser.add_argument(
        '--eggbox-mesh', nargs='+', type=int, default=(15, 21, 31, 41))
    parser.add_argument('--fd-step', type=float, default=1e-4)
    parser.add_argument(
        '--threads', type=int,
        help=('PySCF/OpenMP thread count. By default, preserve the thread '
              'count selected by the active environment.'))
    parser.add_argument('--output', type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mesh <= 0 or any(mesh <= 0 for mesh in args.eggbox_mesh):
        raise ValueError('mesh dimensions must be positive')
    if args.fd_step <= 0:
        raise ValueError('fd-step must be positive')
    if args.threads is not None:
        if args.threads <= 0:
            raise ValueError('threads must be positive')
        lib.num_threads(args.threads)
    result = {
        'environment': {
            'python': platform.python_version(),
            'pyscf': pyscf.__version__,
            'dcforge': str(Path(dcforge.__file__).resolve()),
            'pyscf_threads': lib.num_threads(),
        },
        'conditions': {
            'structure': 'asymmetric Water1',
            'basis': 'gth-szv',
            'pseudo': 'gth-pbe',
            'kmesh': [2, 1, 1],
            'functionals': args.xc,
            'providers': args.provider,
            'mesh': args.mesh,
            'eggbox_meshes': args.eggbox_mesh,
            'fd_step_bohr': args.fd_step,
            'scf_conv_tol': 1e-12,
        },
    }
    if 'gradient' in args.phase:
        result['gradient'] = validate_gradients(
            args.xc, args.provider, args.mesh, args.fd_step)
    if 'symmetry' in args.phase:
        result['symmetry'] = validate_symmetry(
            args.xc, args.provider, args.mesh)
    if 'eggbox' in args.phase:
        result['eggbox'] = validate_eggbox(
            args.xc, args.provider, args.eggbox_mesh, args.fd_step)

    text = json.dumps(result, indent=2)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + '.tmp')
        temporary.write_text(text + '\n')
        temporary.replace(args.output)
        print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
