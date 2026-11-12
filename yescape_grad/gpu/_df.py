# YEScape-Grad
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

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
        YEScape-Grad shares one exchange-capable provider between source SCF and
        target DFT. The default deliberately remains ``xc='HF'`` (JK-fit),
        including pure-DFA sources/targets, to preserve the HF-source policy.
        This is a basis-selection policy, not a request to compute exchange
        for a functional whose exact-exchange coefficient is zero.
    '''
    if auxbasis is not None or with_df is not None:
        return auxbasis
    if isinstance(mol.basis, str):
        return pyscf_df_addons.predefined_auxbasis(
            mol, mol.basis, xc='HF')
    return None
