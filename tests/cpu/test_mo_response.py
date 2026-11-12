"""Independent AO oracles and relaxed-source FD for CPU DF-HF MO response."""
import numpy as np
import pytest
from pyscf import gto, dft
from yescape_grad.cpu.energy import RNSC, UNSC
from yescape_grad.cpu.df.grad.response import MOResponse
from tests.common import configure_workflow, displaced_mol


def workflow(kind, atom, spin=0, basis='def2-svp', source=None):
    mol = gto.M(atom=atom, spin=spin, basis=basis, verbose=0)
    cls = UNSC if kind == 'u' else RNSC
    mf = cls(mol, xc='PBE', mf_source=source).density_fit()
    configure_workflow(mf, grid_level=1)
    mf.kernel()
    assert mf.mf_source.converged
    return mf


@pytest.mark.cpu
@pytest.mark.parametrize('kind,atom,spin', [
    ('r','O 0 0 0; H 0 -.76 .59; H 0 .76 .59',0),
    ('u','O 0 0 0; O 0 0 1.23',2),
    ('u','H 0 0 0',1),  # beta occupied space is empty
    ('u','H 0 0 0',-1),  # alpha occupied space is empty
])
def test_mo_actions_gradient_and_fd(kind, atom, spin):
    mf = workflow(kind,atom,spin)
    unrestricted = kind == 'u'
    cs = mf.mo_coeff if unrestricted else (mf.mo_coeff,)
    occs = mf.mo_occ if unrestricted else (mf.mo_occ,)
    co = tuple(c[:,o>0] for c,o in zip(cs,occs))
    cv = tuple(c[:,o==0] for c,o in zip(cs,occs))
    response = MOResponse(mf.with_df,co if unrestricted else co[0],
                          cv if unrestricted else cv[0],unrestricted)
    rng = np.random.default_rng(12)
    xs = tuple(rng.normal(size=(v.shape[1],o.shape[1]))*.01 for o,v in zip(co,cv))
    # Exercise spin-local and cross-spin Coulomb responses and batched solver inputs.
    for active in range(len(co)):
        trial = tuple(x if s==active else np.zeros_like(x) for s,x in enumerate(xs))
        dms = []
        for o,v,x in zip(co,cv,trial):
            dm=v@x@o.T;dms.append(dm+dm.T)
        ao = mf.mf_source.gen_response(hermi=1)(np.asarray(dms) if unrestricted else dms[0])
        ao = ao if unrestricted else (ao,)
        for block,left in [('vo',cv),('oo',co)]:
            actual=response.potential(trial,block)
            for value,l,o,v in zip(actual,left,co,ao):
                np.testing.assert_allclose(value,l.T@v@o,atol=2e-10,rtol=0)
        packed=np.concatenate([x.ravel() for x in trial])
        expected=np.concatenate([(v.T@a@o).ravel() for v,a,o in zip(cv,ao,co)])
        expected*=1 if unrestricted else 2
        np.testing.assert_allclose(response.vind(np.array([packed,2*packed])),
                                   np.array([expected,2*expected]),atol=4e-10,rtol=0)
    gm=mf.nuc_grad_method(); assert gm.response_backend=='mo_direct'
    actual=gm.kernel(); assert gm.response_backend_used=='mo_direct'
    ga=mf.nuc_grad_method().set(response_backend='ao');reference=ga.kernel()
    print(f'{kind} spin={spin}: max gradient delta={np.max(abs(actual-reference)):.3e}, AO physical Z residual={gm.z_residual_max_abs:.3e}')
    np.testing.assert_allclose(actual,reference,atol=2e-9,rtol=0)
    np.testing.assert_allclose(gm.de_ovlp_response,ga.de_ovlp_response,atol=2e-9,rtol=0)
    for zm,za in zip(gm.zvec,ga.zvec):
        np.testing.assert_allclose(zm,za,atol=2e-8,rtol=0)
    assert abs(gm.z_residual_max_abs-ga.z_residual_max_abs)<2e-8
    # One nonzero bond-coordinate FD; both signs reconverge source orbitals.
    if mf.mol.natm>1:
        energies=[]
        for shift in [1e-4,-1e-4]:
            mol=displaced_mol(mf.mol,mf.mol.natm-1,2,shift)
            displaced=(UNSC if unrestricted else RNSC)(mol,xc='PBE').density_fit()
            configure_workflow(displaced,grid_level=1)
            energies.append(displaced.kernel())
        fd=(energies[0]-energies[1])/2e-4
        print(f'{kind} spin={spin}: FD error={abs(fd-actual[-1,2]):.3e} Ha/Bohr')
        assert abs(fd-actual[-1,2])<2e-6, (fd,actual[-1,2])


@pytest.mark.cpu
@pytest.mark.parametrize('kind,spin', [('r',0),('u',2)])
def test_defaults_guards_memory_and_rebuild(kind,spin):
    atom='O 0 0 0; O 0 0 1.23' if spin else 'H 0 0 0; H 0 0 .8'
    mf=workflow(kind,atom,spin,basis='sto-3g')
    g=mf.nuc_grad_method();first=g.kernel()
    assert g.response_mo_cache_bytes>0
    # Same driver after a source recalculation must not retain old MO factors.
    mol=displaced_mol(mf.mol,1,2,.1)
    scan=g.as_scanner();_,changed=scan(mol)
    ao=scan.base.nuc_grad_method().set(response_backend='ao').kernel()
    np.testing.assert_allclose(changed,ao,atol=2e-9,rtol=0)
    assert np.max(abs(first-changed))>1e-5
    g=scan.base.nuc_grad_method().set(max_memory=0)
    g.kernel();assert g.response_backend_used=='ao' and g.response_mo_cache_bytes==0
    with pytest.raises(ValueError,match='response_backend'):
        scan.base.nuc_grad_method().set(response_backend='invalid').kernel()
    src=(dft.UKS if kind=='u' else dft.RKS)(mol).set(xc='PBE')
    cls=UNSC if kind=='u' else RNSC
    dfa=cls(mol,xc='PBE',mf_source=src).density_fit()
    assert dfa.nuc_grad_method().response_backend=='ao'
    with pytest.raises(ValueError,match='DFA source'):
        dfa.nuc_grad_method().set(response_backend='mo_direct').make_response(None,None)
    non_df=cls(mol,xc='PBE').nuc_grad_method()
    assert non_df.response_backend=='ao'
    with pytest.raises(ValueError,match='conventional CPU'):
        non_df.set(response_backend='mo_direct').make_response(None,None)


@pytest.mark.cpu
@pytest.mark.parametrize('kind', ['r','u'])
def test_empty_vo(kind):
    mf=workflow(kind,'He 0 0 0',basis='sto-3g')
    g=mf.nuc_grad_method();np.testing.assert_allclose(g.kernel(),0,atol=1e-10)
    assert g.z_solver=='zero_dimensional'
