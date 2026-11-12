'''CPU energy evaluation operations.'''

from yescape_grad.cpu.energy import rnsc, unsc
from yescape_grad.cpu.energy.rnsc import RNSC, energy_tot
from yescape_grad.cpu.energy.unsc import UNSC

__all__ = [
    'RNSC', 'UNSC', 'energy_tot', 'rnsc', 'unsc',
]

# Source-neutral public names; legacy HF-only names remain compatible.
RHFDFT = RNSC
UHFDFT = UNSC
__all__ += ['RHFDFT', 'UHFDFT']
