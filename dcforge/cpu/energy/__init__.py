'''CPU energy evaluation operations.'''

from dcforge.cpu.energy import rnsc, unsc
from dcforge.cpu.energy.rnsc import RNSC, energy_tot
from dcforge.cpu.energy.unsc import UNSC

__all__ = [
    'RNSC', 'UNSC', 'energy_tot', 'rnsc', 'unsc',
]

# Source-neutral public names; legacy HF-only names remain compatible.
RHFDFT = RNSC
UHFDFT = UNSC
__all__ += ['RHFDFT', 'UHFDFT']
