'''GPU fixed-density energy operations.'''

from yescape_grad.gpu.energy import rnsc, unsc
from yescape_grad.gpu.energy.rnsc import RNSC
from yescape_grad.gpu.energy.unsc import UNSC

__all__ = ['RNSC', 'UNSC', 'rnsc', 'unsc']

# Source-neutral public names; legacy HF-only names remain compatible.
RHFDFT = RNSC
UHFDFT = UNSC
__all__ += ['RHFDFT', 'UHFDFT']
