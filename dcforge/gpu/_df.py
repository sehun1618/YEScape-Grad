'''Auxiliary-basis helpers for molecular GPU density fitting.'''

from pyscf.df import addons as pyscf_df_addons


def default_auxbasis(mol, auxbasis=None, with_df=None):
    '''Return PySCF's matching HF auxiliary basis when none is supplied.

    Args:
        mol : pyscf.gto.Mole
            Molecule whose orbital basis determines the auxiliary basis.

    Kwargs:
        auxbasis : str, basis dict, or None
            User-supplied auxiliary basis. A non-None value is returned
            unchanged.
        with_df : gpu4pyscf.df.df.DF or None
            Existing provider supplied by the caller. Its auxiliary basis must
            remain authoritative, so no default is selected when it is given.

    Returns:
        str, basis dict, or None
            PySCF's predefined HF/JK-fit match for a string orbital basis, or
            None when PySCF should generate the final element-wise/even-tempered
            fallback while building the DF provider.

    Notes:
        DC-Forge shares one exchange-capable provider between source HF and
        target DFT. The default is therefore selected with ``xc='HF'`` even
        when the target functional is a local or semilocal DFA. This mirrors
        ``pyscf.df.df_jk.density_fit`` for an RHF or UHF source.
    '''
    if auxbasis is not None or with_df is not None:
        return auxbasis
    if isinstance(mol.basis, str):
        return pyscf_df_addons.predefined_auxbasis(
            mol, mol.basis, xc='HF')
    return None
