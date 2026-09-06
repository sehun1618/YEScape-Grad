'''Provider-stored Gamma-point CDERI access shared by GDF and RSDF.'''

import numpy as np
from pyscf import lib
from pyscf.pbc.df.df import _load3c


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


def canonical_kpt_pairs(nkpts):
    '''Return lower-triangular ``(ki,kj)`` pairs in PySCF storage order.'''
    nkpts = int(nkpts)
    if nkpts < 1:
        raise ValueError('nkpts must be positive')
    return tuple(
        (ki, kj) for ki in range(nkpts) for kj in range(ki + 1))


def load_cderi_pairs(with_df, pairs=None):
    '''Return provider-stored sampled-k CDERI as dense canonical blocks.

    Args:
        with_df : pyscf.pbc.df.GDF
            Built sampled-k Gaussian provider.

    Kwargs:
        pairs : sequence of pair or None
            Canonical pair subset satisfying ``ki >= kj``.  ``None`` loads
            every lower-triangular pair.

    Returns:
        dict
            ``{(ki,kj): C_P(ki,kj)}`` with dense complex shape
            ``(naux,nao,nao)``.  No HDF5 view remains open.

    Notes:
        PySCF packs diagonal symmetric AO pairs but stores off-diagonal pairs
        as full complex matrices.  This function changes only storage layout;
        central values remain exactly those used by the source KRHF SCF.
    '''
    if getattr(with_df, '_cderi', None) is None or with_df.auxcell is None:
        with_df.build(j_only=False)
    kpts = np.asarray(with_df.kpts, dtype=float).reshape(-1, 3)
    if pairs is None:
        pairs = canonical_kpt_pairs(len(kpts))
    pairs = tuple((int(ki), int(kj)) for ki, kj in pairs)
    nao = with_df.cell.nao_nr()
    naux = with_df.auxcell.nao_nr()
    dataname = getattr(with_df, '_dataname', 'j3c')
    blocks = {}
    records = []
    for ki, kj in pairs:
        if not (0 <= kj <= ki < len(kpts)):
            raise ValueError(
                'canonical k-point pair must satisfy 0 <= kj <= ki < nkpts')
        with _load3c(
                with_df._cderi, dataname, kpts[[ki, kj]]) as cderi:
            raw = np.asarray(cderi[:]).copy()
        if raw.shape[0] != naux:
            raise ValueError('stored CDERI auxiliary dimension is inconsistent')
        if raw.shape[-1] == nao * nao:
            dense = raw.reshape(naux, nao, nao)
            storage = 'full_s1'
        elif raw.shape[-1] == nao * (nao + 1) // 2:
            dense = np.asarray([lib.unpack_tril(row) for row in raw])
            storage = 'packed_s2'
        else:
            raise ValueError(f'unsupported stored CDERI shape {raw.shape}')
        blocks[(ki, kj)] = dense
        records.append({
            'pair': (ki, kj), 'raw_shape': raw.shape,
            'dense_shape': dense.shape, 'storage': storage,
            'dtype': str(dense.dtype),
        })
    with_df.cderi_pair_records = tuple(records)
    return blocks


def get_cderi_pair(blocks, ki, kj):
    r'''Return ``C(ki,kj)`` using Hermitian pair reversal.

    ``C_P(ki,kj)[mu,nu] = C_P(kj,ki)[nu,mu]*``.
    '''
    pair = (int(ki), int(kj))
    if pair in blocks:
        return np.asarray(blocks[pair])
    reverse = (pair[1], pair[0])
    if reverse in blocks:
        return np.asarray(blocks[reverse]).conj().transpose(0, 2, 1)
    raise KeyError(f'CDERI block for pair {pair} is unavailable')


def get_cderi_deriv_pair(blocks, ki, kj, atom, axis):
    '''Return one ``C^lambda(ki,kj)`` component under pair reversal.'''
    pair = (int(ki), int(kj))
    atom = int(atom)
    axis = int(axis)
    if pair in blocks:
        return np.asarray(blocks[pair])[atom, axis]
    reverse = (pair[1], pair[0])
    if reverse in blocks:
        return np.asarray(
            blocks[reverse])[atom, axis].conj().transpose(0, 2, 1)
    raise KeyError(f'CDERI derivative block for pair {pair} is unavailable')


def _real_if_close(value, tol=1e-10):
    value = np.asarray(value)
    if value.size and np.max(np.abs(value.imag)) <= tol:
        return value.real
    return value
