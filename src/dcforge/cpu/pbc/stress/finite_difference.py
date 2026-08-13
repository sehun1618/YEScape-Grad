'''Finite-strain energy oracles for periodic DC-DFT stress validation.'''

import numpy as np

from dcforge.cpu.pbc.stress.strain import (
    VOIGT_LABELS,
    assert_same_strain_representation,
    deform_cell,
    pressure_from_stress,
    raw_strain_basis,
    snapshot_strain_state,
    stress_from_work,
    stress_tensor_to_voigt,
    virial_from_stress,
    voigt_strain_basis,
)

DEFAULT_STRAIN_STEPS = (5.e-4, 2.5e-4, 1.25e-4)


def finite_difference_strain(cell, energy_function, steps=None,
                             directions='voigt', mode='relaxed_energy'):
    r'''Evaluate central finite differences of a caller-owned energy.

    This function is an independent numerical oracle, not the production
    analytic stress implementation.  ``energy_function`` is called once for
    the central Cell and once for every ``+h/-h`` Cell.  It may return either
    a real scalar energy or a dictionary with at least

    ``{'energy': energy, 'state': snapshot}``.

    When a state snapshot is supplied, its ``representation`` must match the
    central snapshot exactly.  This turns mesh jumps, changed GDF resolution,
    auxiliary-basis fallback and multigrid hierarchy changes into hard
    failures instead of silently accepting them as stress.

    Args:
        cell : pyscf.pbc.gto.Cell
            Built central Cell.
        energy_function : callable
            ``energy_function(strained_cell)`` returning a scalar or result
            dictionary.  A relaxed callback reconverges source RHF at every
            Cell; a frozen callback owns and documents its density transport.

    Kwargs:
        steps : sequence of float or None
            Positive dimensionless strain steps.  Defaults to
            ``(5e-4, 2.5e-4, 1.25e-4)``.
        directions : {'voigt', 'raw9'} or ndarray
            Six symmetric engineering-Voigt directions, nine ordered raw
            directions, or an explicit ``(n, 3, 3)`` array.
        mode : {'relaxed_energy', 'frozen_density'}
            Provenance label; it does not alter callback behavior.

    Returns:
        dict
            JSON-safe energies, work derivatives, stresses and representation
            metadata for every requested step.
    '''
    if mode not in ('relaxed_energy', 'frozen_density'):
        raise ValueError('finite-strain mode must be relaxed_energy or frozen_density')
    if steps is None:
        steps = DEFAULT_STRAIN_STEPS
    steps = tuple(float(step) for step in steps)
    if not steps or any(not np.isfinite(step) or step <= 0 for step in steps):
        raise ValueError('finite-strain steps must be positive and finite')
    direction_values, labels = _resolve_directions(directions)

    central = _evaluate_energy(energy_function, cell)
    central_state = central.get('state')
    if central_state is None:
        central_state = snapshot_strain_state(cell)
        central['state'] = central_state
    central_volume = float(cell.vol)
    if not np.isfinite(central_volume) or central_volume <= 0:
        raise ValueError('central Cell volume must be positive and finite')
    central_mesh = np.asarray(cell.mesh, dtype=int).copy()

    records = []
    for label, direction in zip(labels, direction_values, strict=True):
        step_records = []
        for step in steps:
            plus_cell = deform_cell(
                cell, direction, scale=step, mesh=central_mesh)
            minus_cell = deform_cell(
                cell, direction, scale=-step, mesh=central_mesh)
            plus = _evaluate_energy(energy_function, plus_cell)
            minus = _evaluate_energy(energy_function, minus_cell)
            _validate_state(central_state, plus, plus_cell)
            _validate_state(central_state, minus, minus_cell)
            work = (plus['energy'] - minus['energy']) / (2 * step)
            step_records.append({
                'step': step,
                'plus': plus,
                'minus': minus,
                'work_hartree': work,
                'stress_ha_per_bohr3': work / central_volume,
            })
        records.append({
            'label': label,
            'direction': np.asarray(direction).tolist(),
            'steps': step_records,
        })

    assembled = []
    for index, step in enumerate(steps):
        components = np.asarray([
            record['steps'][index]['work_hartree'] for record in records])
        work_raw = np.einsum(
            'm,mxy->xy', components, _dual_direction_basis(direction_values))
        stress = stress_from_work(work_raw, central_volume)
        antisymmetric_work = (work_raw - work_raw.T) * .5
        assembled.append({
            'step': step,
            'work_components_hartree': components.tolist(),
            'work_raw_hartree': work_raw.tolist(),
            'stress_tensor_ha_per_bohr3': stress.tolist(),
            'stress_voigt_ha_per_bohr3': (
                stress_tensor_to_voigt(stress).tolist()),
            'pressure_ha_per_bohr3': float(pressure_from_stress(stress)),
            'virial_hartree': virial_from_stress(
                stress, central_volume).tolist(),
            'antisymmetric_work_max_abs_hartree': float(
                np.max(np.abs(antisymmetric_work))),
            'antisymmetric_stress_max_abs_ha_per_bohr3': float(
                np.max(np.abs(antisymmetric_work)) / central_volume),
        })

    return {
        'schema': 'dcforge.pbc.stress.finite_difference.v1',
        'production_analytic': False,
        'mode': mode,
        'direction_set': (
            directions if isinstance(directions, str) else 'explicit'),
        'voigt_order': list(VOIGT_LABELS),
        'central_volume_bohr3': central_volume,
        'central': central,
        'directions': records,
        'assembled': assembled,
    }


def relaxed_energy_finite_difference(cell, energy_function, steps=None,
                                     directions='voigt'):
    r'''Return a relaxed-source energy finite-strain stress oracle.

    ``energy_function`` must reconverge source RHF in every strained Cell and
    then evaluate the target DFA only on that converged source density.
    '''
    return finite_difference_strain(
        cell, energy_function, steps=steps, directions=directions,
        mode='relaxed_energy')


def frozen_density_finite_difference(cell, energy_function, steps=None,
                                     directions='voigt'):
    r'''Return a frozen-density primitive-debugging finite-strain oracle.

    The callback is responsible for defining and reporting how the central AO
    density is transported to a strained AO basis.  This result must not be
    presented as the relaxed DC-DFT stress because it omits source-orbital
    response.
    '''
    return finite_difference_strain(
        cell, energy_function, steps=steps, directions=directions,
        mode='frozen_density')


def _resolve_directions(directions):
    if isinstance(directions, str):
        if directions == 'voigt':
            return voigt_strain_basis(), VOIGT_LABELS
        if directions == 'raw9':
            labels = tuple(
                f'{row}{column}' for row in 'xyz' for column in 'xyz')
            return raw_strain_basis(), labels
        raise ValueError("directions must be 'voigt', 'raw9', or an array")
    values = np.asarray(directions, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (3, 3):
        raise ValueError('explicit strain directions must have shape (n, 3, 3)')
    labels = tuple(f'direction_{index}' for index in range(len(values)))
    return values, labels


def _dual_direction_basis(directions):
    r'''Return the Frobenius-dual basis used to reconstruct a raw tensor.'''
    directions = np.asarray(directions, dtype=float)
    matrix = directions.reshape(len(directions), 9)
    rank = np.linalg.matrix_rank(matrix)
    if rank == 9:
        return np.linalg.pinv(matrix).T.reshape(len(directions), 3, 3)
    if len(directions) == 6 and np.allclose(
            directions, voigt_strain_basis(), atol=0, rtol=0):
        # The six directional derivatives determine only the physical
        # symmetric work tensor.  Stress shear components equal derivatives
        # with respect to engineering shear, so both raw off-diagonals carry
        # the same component.
        dual = np.zeros_like(directions)
        dual[:3] = directions[:3]
        dual[3, 1, 2] = dual[3, 2, 1] = 1
        dual[4, 0, 2] = dual[4, 2, 0] = 1
        dual[5, 0, 1] = dual[5, 1, 0] = 1
        return dual
    raise ValueError(
        'explicit directions must span all nine tensor components; use the '
        'canonical six Voigt directions for a symmetric stress')


def _evaluate_energy(energy_function, cell):
    result = energy_function(cell)
    if isinstance(result, dict):
        result = dict(result)
        if 'energy' not in result:
            raise ValueError('energy callback dictionary must contain energy')
    else:
        result = {'energy': result}
    energy = np.asarray(result['energy'])
    if energy.size != 1 or np.max(np.abs(energy.imag)) > 1.e-10:
        raise ValueError('finite-strain energy must be one real scalar')
    energy = float(energy.real)
    if not np.isfinite(energy):
        raise ValueError('finite-strain energy must be finite')
    result['energy'] = energy
    return result


def _validate_state(central_state, result, cell):
    state = result.get('state')
    if state is None:
        state = snapshot_strain_state(cell)
        result['state'] = state
    assert_same_strain_representation(central_state, state)
    if not np.array_equal(
            np.asarray(state['representation']['mesh']),
            np.asarray(central_state['representation']['mesh'])):
        raise ValueError('strained energy callback changed the integer mesh')
