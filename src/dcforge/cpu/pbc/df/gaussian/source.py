r'''Complete Gamma-point Gaussian-DF three-center source and response.'''

from dcforge.cpu.pbc.df.gaussian.source_long_range import (
    gen_grad_long_range_source,
    long_range_source,
)
from dcforge.cpu.pbc.df.gaussian.source_short_range import (
    gen_grad_short_range_source,
    short_range_source,
)
from dcforge.cpu.pbc.df.gaussian.vbar import (
    gen_grad_vbar_source,
    vbar_source,
)


def three_center_source(
        cell, auxcell, omega, mesh, precision=None, max_memory=None):
    r'''Return :math:`B=B^{SR}+B^{LR}+B^{vbar}`.

    ``max_memory`` is a PySCF-style budget in MB shared by the short-range
    integral traversal and long-range reciprocal-grid blocking.
    '''
    return (
        short_range_source(
            cell, auxcell, omega, precision, max_memory)
        + long_range_source(
            cell, auxcell, omega, mesh, max_memory)
        + vbar_source(cell, auxcell, omega)
    )


def grad_three_center_source(cell, auxcell, omega, mesh, precision=None,
                             atmlst=None, max_memory=None):
    r'''Return :math:`B^\lambda` for the selected nuclear centers.

    Kwargs:
        precision : float or None
            Real-space three-center screening threshold.
        atmlst : sequence of int or None
            Atom indices owned by this response block.  ``None`` returns the
            full :math:`3N` oracle.
        max_memory : float or None
            Shared short-range and long-range memory budget in MB.
    '''
    source1 = gen_grad_three_center_source(
        cell, auxcell, omega, mesh, precision, max_memory)
    return source1(atmlst)


def gen_grad_three_center_source(cell, auxcell, omega, mesh,
                                 precision=None, max_memory=None):
    r'''Generate atom-block assembly of the complete :math:`B^\lambda`.

    Each term generator owns its coordinate-common integral or Fourier
    intermediates.  The returned callable preserves the explicit
    SR/LR/vbar sum while limiting the atom-resolved output allocation.

    ``max_memory`` is passed unchanged to both SR and LR generators.  The
    resulting callable reports whether the LR auxiliary response caches the
    full AO-pair Fourier transform or streams it over reciprocal-grid blocks.
    '''
    short_range1 = gen_grad_short_range_source(
        cell, auxcell, omega, precision, max_memory)
    long_range1 = gen_grad_long_range_source(
        cell, auxcell, omega, mesh, max_memory)
    vbar1 = gen_grad_vbar_source(cell, auxcell, omega)

    def source1(atmlst=None):
        return (
            short_range1(atmlst)
            + long_range1(atmlst)
            + vbar1(atmlst)
        )

    source1.fourier_aux_storage = long_range1.fourier_aux_storage
    source1.fourier_ao_pair_bytes = long_range1.fourier_ao_pair_bytes
    return source1
