'''YEScape-Grad public package.

The top-level package exposes the CPU namespace without importing GPU4PySCF.
GPU workflows require an explicit ``yescape_grad.gpu`` import.
'''

from yescape_grad import cpu

__all__ = ['cpu']
__version__ = '1.0.0'
