'''D2C-D4 energy and molecular nuclear-gradient correction.

The rational-damping parameters are from Table S1 of Lee et al.,
J. Chem. Theory Comput. 2024, DOI 10.1021/acs.jctc.4c00689.  D2C-D4 is a
geometry-only correction.  It therefore adds an explicit derivative to the
HF-DFT gradient but does not change the source-HF CPHF/Z-vector equation.
'''

from dataclasses import dataclass
from time import perf_counter

import numpy as np


# Ref.
# M. Lee et al., "Correcting Dispersion Corrections with Density-Corrected
# DFT", J. Chem. Theory Comput. 20, 7155-7167 (2024);
# DOI: 10.1021/acs.jctc.4c00689.
# The parameters below are the D2C-DFT column of Supporting Information,
# Table S1, rather than the ordinary self-consistent DFT-D4 column.
# Published molecular D2C-D4 parameters.  All entries use s6=s9=1 and
# alp=16; only s8, a1 and a2 were fitted for each target functional.
_PARAMETERS = {
    'pbe': ('PBE', 1.78595387, 0.88511469, 2.32863362),
    'pbe0': ('PBE0', 1.53360351, 0.78689150, 3.25641582),
    'revpbe': ('revPBE', 1.80966761, 0.58558155, 2.71965468),
    'revpbe0': ('revPBE0', 1.74590783, 0.40385673, 4.26818294),
    'blyp': ('BLYP', 1.65244425, 0.56438878, 2.65715701),
    'b3lyp': ('B3LYP', 1.35513689, 0.41757850, 3.84594813),
    'tpss': ('TPSS', 1.54044984, 0.69473318, 2.51512802),
    'tpss0': ('TPSS0', 1.50843498, 0.60162555, 3.65500533),
    'scan': ('SCAN', 1.72616184, 0.06450398, 8.62911596),
    'scan0': ('SCAN0', 3.69655894, 0.16214976, 8.90158495),
    'r2scan': ('r2SCAN', 0.02734375, 0.74707031, 3.34667969),
    'r2scan0': ('r2SCAN0', 3.97877459, 0.75987648, 5.45977445),
    'm06l': ('M06L', 0.75781250, 0.81445313, 6.16992188),
    'm06': ('M06', 1.30522230, 0.83568617, 4.37780185),
}


def normalize_dispersion(value):
    '''Normalize the optional DC-Forge dispersion mode.'''
    if value is None:
        return None
    if value == 'd2c':
        return 'd2c'
    raise ValueError("dispersion must be None or 'd2c'")


def _functional_key(xc):
    return ''.join(char for char in str(xc).lower() if char.isalnum())


def get_d2c_parameters(xc):
    '''Return the published D2C-D4 parameters for one target functional.

    Args:
        xc : str
            Target functional label, for example ``'PBE'`` or ``'r2SCAN'``.

    Returns:
        dict
            Rational-damping parameters accepted by
            :class:`dftd4.interface.DampingParam`.

    Raises:
        ValueError
            If no D2C parameterization was published for ``xc``.  An unknown
            functional is never replaced by ordinary D4 parameters.
    '''
    key = _functional_key(xc)
    try:
        label, s8, a1, a2 = _PARAMETERS[key]
    except KeyError as exc:
        supported = ', '.join(value[0] for value in _PARAMETERS.values())
        raise ValueError(
            f'no published D2C-D4 parameters for xc={xc!r}; '
            f'supported functionals: {supported}'
        ) from exc
    return {
        'functional': label,
        's6': 1.0,
        's8': s8,
        's9': 1.0,
        'a1': a1,
        'a2': a2,
        'alp': 16.0,
    }


@dataclass(frozen=True)
class D2CCorrection:
    '''One molecular D2C-D4 result in atomic units.'''

    energy: float
    gradient: np.ndarray | None
    parameters: dict
    wall_time_s: float


def _import_dftd4():
    try:
        from dftd4.interface import DampingParam, DispersionModel
    except ImportError as exc:
        raise ImportError(
            'D2C-D4 requires the optional dftd4 package; install dftd4>=4.2'
        ) from exc
    return DampingParam, DispersionModel


def evaluate_d2c(mol, xc, *, gradient=False):
    '''Evaluate molecular D2C-D4 for the current geometry.

    Args:
        mol : pyscf.gto.Mole
            Molecular geometry.  Coordinates are read in Bohr from PySCF.
        xc : str
            Target HF-DFT functional used to select the published D2C set.

    Kwargs:
        gradient : bool
            Also evaluate ``dE_D2C/dR`` when true.

    Returns:
        D2CCorrection
            D2C energy in Hartree and, when requested, its analytic gradient
            in Hartree/Bohr.

    Notes:
        D4 uses atomic numbers, coordinates and total molecular charge.  It
        does not use the HF density, orbitals or Z-vector.
    '''
    if hasattr(mol, 'lattice_vectors'):
        raise TypeError('this evaluator supports molecular PySCF Mole objects only')

    start = perf_counter()
    DampingParam, DispersionModel = _import_dftd4()
    parameters = get_d2c_parameters(xc)
    model = DispersionModel(
        numbers=np.asarray(mol.atom_charges(), dtype=np.int32),
        positions=np.asarray(mol.atom_coords(), dtype=float),
        charge=float(mol.charge),
        model='d4',
    )
    damping = DampingParam(**{
        key: parameters[key]
        for key in ('s6', 's8', 's9', 'a1', 'a2', 'alp')
    })
    result = model.get_dispersion(param=damping, grad=bool(gradient))
    energy = float(np.asarray(result['energy']).item())
    de = result.get('gradient')

    if not np.isfinite(energy):
        raise RuntimeError('D2C-D4 returned a non-finite energy')
    if de is not None:
        de = np.asarray(de, dtype=float)
        if de.shape != (mol.natm, 3):
            raise RuntimeError(
                f'D2C-D4 gradient has shape {de.shape}, '
                f'expected {(mol.natm, 3)}'
            )
        if not np.all(np.isfinite(de)):
            raise RuntimeError('D2C-D4 returned a non-finite gradient')

    return D2CCorrection(
        energy=energy,
        gradient=de,
        parameters=parameters,
        wall_time_s=perf_counter() - start,
    )
