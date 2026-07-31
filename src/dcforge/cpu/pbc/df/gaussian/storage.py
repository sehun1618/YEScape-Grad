'''Provider-stored Gamma-point CDERI access shared by GDF and RSDF.'''

import numpy as np


def load_cderi(with_df):
    '''Return provider-stored Gamma CDERI as ``(naux,nao,nao)``.'''
    blocks = []
    for real, imag, sign in with_df.sr_loop(compact=False):
        if sign != 1:
            raise NotImplementedError(
                'Gaussian-DF response requires positive-definite 3D '
                'metric blocks')
        blocks.append(real + 1j * imag)
    if not blocks:
        raise RuntimeError('Gaussian-DF provider did not yield a CDERI block')
    nao = with_df.cell.nao_nr()
    return _real_if_close(
        np.concatenate(blocks).reshape(-1, nao, nao))


def _real_if_close(value, tol=1e-10):
    value = np.asarray(value)
    if value.size and np.max(np.abs(value.imag)) <= tol:
        return value.real
    return value
