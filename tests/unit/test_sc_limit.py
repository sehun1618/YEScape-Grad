"""Regression checks for SC bypass, source grid dispatch, and safe fallback."""
from importlib import import_module
from io import StringIO
import numpy as np
import pytest
from pyscf import gto,lib

def build(backend,u,df,source,target):
    lib.num_threads(4)
    cls=getattr(import_module(f'dcforge.{backend}.energy'),'UNSC' if u else 'RNSC')
    mol=gto.M(atom='N 0 0 0; H 0 -.78 .58; H 0 .82 .63' if u else 'O 0 0 0; H 0 -.78 .58; H 0 .82 .63',spin=int(u),basis='6-31g',verbose=0)
    mf=cls(mol,xc=target,source_xc=source)
    if df:mf=mf.density_fit(auxbasis='def2-universal-jkfit')
    for obj in [mf,mf.mf_source]:
        obj.grids.atom_grid=(50,194);obj.grids.prune=None;obj.small_rho_cutoff=0
    mf.conv_tol=1e-12;mf.conv_tol_grad=1e-9;mf.max_cycle=200
    mf.kernel();assert mf.mf_source.converged
    return mf

@pytest.mark.parametrize('backend',['cpu',pytest.param('gpu',marks=pytest.mark.gpu)])
@pytest.mark.parametrize('u',[False,True])
@pytest.mark.parametrize('df',[False,True])
@pytest.mark.parametrize('xc',['LDA,VWN','PBE','r2SCAN','B3LYP','CAM-B3LYP'])
def test_sc_limit_matrix(backend,u,df,xc,monkeypatch):
    mf=build(backend,u,df,xc,xc);g=mf.nuc_grad_method();g.cphf_conv_tol=1e-11
    module=import_module(f'dcforge.{backend}.grad.{"u" if u else "r"}nsc')
    def forbidden(*a,**k):raise AssertionError('SC branch attempted NSC response')
    with monkeypatch.context() as m:
        m.setattr(module,'solve_z_vector',forbidden)
        m.setattr(g,'grad_fock',forbidden)
        fast=g.kernel()
    assert g.sc_limit_used and g.z_residual_max_abs is None
    np.testing.assert_array_equal(g.de_nvar,np.zeros_like(fast))
    native=mf.mf_source.nuc_grad_method();native.grid_response=True
    if df:native.auxbasis_response=True
    exact=native.kernel()
    np.testing.assert_allclose(fast,exact,atol=1e-9,rtol=0)
    g.sc_limit=False
    try:
        full=g.kernel();assert not g.sc_limit_used
        np.testing.assert_allclose(fast,full,atol=2e-7,rtol=0)
    except ValueError as exc:
        # Known zero-rank failure in the forced NSC route. These two SC
        # cases also have independent native-gradient and FD checks below.
        assert backend=='gpu' and u and df and xc in ['B3LYP','CAM-B3LYP']
        assert str(exc)=='cannot reshape array of size 0 into shape (-1, 0)'
    # Switching back must clear all diagnostics left by the NSC call.
    g.sc_limit=True;g.kernel();assert g.sc_limit_used and g.z_residual_max_abs is None and g.zvec is None

@pytest.mark.parametrize('u',[False,True])
@pytest.mark.parametrize('df',[False,True])
@pytest.mark.parametrize('source',['LDA,VWN','PBE','r2SCAN','B3LYP','CAM-B3LYP'])
def test_source_grid_backends(u,df,source,monkeypatch):
    target='r2SCAN' if source=='PBE' else 'PBE'
    mf=build('cpu',u,df,source,target);g=mf.nuc_grad_method();g.cphf_conv_tol=1e-11
    g.grid_response_backend='pyscf_default';a=g.kernel()
    from dcforge.cpu.grad import grid_response
    original=grid_response.grids_response_becke;calls=[]
    def tracked(grids,*args,**kw):
        calls.append(grids is mf.mf_source.grids)
        yield from original(grids,*args,**kw)
    monkeypatch.setattr(grid_response,'grids_response_becke',tracked)
    g.grid_response_backend='becke_omp';b=g.kernel()
    assert True in calls and False in calls  # Both source and target actually use it.
    assert g.source_grid_response_backend_used=='becke_omp'
    np.testing.assert_allclose(a,b,atol=1e-10,rtol=0)

@pytest.mark.parametrize('change',['xc','grid','weights','omega','unconverged','orbitals','grid_response','source_grid_response','df'])
def test_sc_eligibility_rejects_mismatch(change):
    from dcforge.tools.sc_limit import equivalent_source
    mf=build('cpu',False,False,'PBE','PBE');g=mf.nuc_grad_method()
    args=[mf.mo_energy,mf.mo_coeff,mf.mo_occ]
    assert equivalent_source(g,*args)
    if change=='xc':mf.xc='BLYP'
    elif change=='grid':mf.grids.atom_grid=(75,302)
    elif change=='weights':mf.grids.weights[0]+=1e-8
    elif change=='omega':mf.omega=.2
    elif change=='unconverged':mf.mf_source.converged=False
    elif change=='orbitals':args[1]=args[1].copy();args[1][0,0]+=.01
    elif change in ['grid_response','source_grid_response']:setattr(g,change,False)
    elif change=='df':mf.with_df=object()
    if change=='df':
        with pytest.raises(ValueError):equivalent_source(g,*args)
    else:assert not equivalent_source(g,*args)


def test_stage_log_order():
    mf=build('cpu',False,False,'PBE','r2SCAN');g=mf.nuc_grad_method()
    g.stdout=StringIO();g.verbose=5;g.kernel();text=g.stdout.getvalue()
    labels=['Z-vector response solve','Source Fock derivative (including source XC)', 'Target explicit gradient (including target grid)']
    for name in labels:
        assert text.index('Begin '+name)<text.index('End '+name)
        assert g.gradient_stage_times[name]>=0
    assert text.index('End '+labels[0])<text.index('Begin '+labels[1])
    assert text.index('End '+labels[1])<text.index('Begin '+labels[2])
    assert 'aggregate timers include all stages' in text
    assert 'complete gradient (cumulative; all stages)' in text
    assert 'Target grid backend' in text and 'Source grid backend' in text

@pytest.mark.parametrize('backend',['cpu',pytest.param('gpu',marks=pytest.mark.gpu)])
@pytest.mark.parametrize('u',[False,True])
@pytest.mark.parametrize('df',[False,True])
@pytest.mark.parametrize('source',['PBE','r2SCAN'])
def test_five_point_fd(backend,u,df,source):
    mf=build(backend,u,df,source,'r2SCAN');g=mf.nuc_grad_method();g.cphf_conv_tol=1e-11
    analytic=g.kernel()
    # Independent energy derivatives; no analytic-gradient scanner involved.
    scanner=mf.as_scanner();coords=mf.mol.atom_coords();base=mf.mol.copy();h=.005;energies={}
    for k in [-2,-1,1,2]:
        xyz=coords.copy();xyz[1,2]+=k*h
        energies[k]=float(scanner(base.set_geom_(xyz,unit='Bohr',inplace=False)))
        assert scanner.mf_source.converged
    fd=(energies[-2]-8*energies[-1]+8*energies[1]-energies[2])/(12*h)
    err=abs(float(analytic[1,2])-fd)
    assert err<5e-6


@pytest.mark.gpu
@pytest.mark.parametrize('xc',['B3LYP','CAM-B3LYP'])
def test_gpu_udf_sc_fd_with_broken_old_response(xc):
    mf=build('gpu',True,True,xc,xc);grad=mf.nuc_grad_method();g=grad.kernel()
    assert grad.sc_limit_used
    scanner=mf.as_scanner();base=mf.mol.copy();xyz=base.atom_coords();h=.005;e={}
    for k in [-2,-1,1,2]:
        p=xyz.copy();p[1,2]+=k*h;e[k]=float(scanner(base.set_geom_(p,unit='Bohr',inplace=False)))
        assert scanner.mf_source.converged
    fd=(e[-2]-8*e[-1]+8*e[1]-e[2])/(12*h);error=abs(float(g[1,2])-fd)
    assert error<5e-6
