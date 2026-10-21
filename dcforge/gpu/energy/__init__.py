'''GPU fixed-density energy operations.'''

from dcforge.gpu.energy import rnsc, unsc
from dcforge.gpu.energy.rnsc import RNSC
from dcforge.gpu.energy.unsc import UNSC

__all__ = ['RNSC', 'UNSC', 'rnsc', 'unsc']

# Source-neutral public names; legacy HF-only names remain compatible.
RHFDFT = RNSC
UHFDFT = UNSC
__all__ += ['RHFDFT', 'UHFDFT']
