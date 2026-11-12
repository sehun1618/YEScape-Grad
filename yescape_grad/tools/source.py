# YEScape-Grad
# Developed by Sehun Kim <sehun1618@yonsei.ac.kr>
# Theoretical and Computational Chemistry Laboratory (TCCL)
# Yonsei University

'''Shared source-SCF API for non-self-consistent molecular workflows.

Legacy ``mf_hf``/``run_hf`` names remain compatible. New code should use
``mf_source`` and ``kernel(run_source=...)``. Source and target grids are
independent; configure ``mf.mf_source.grids`` before source SCF.
'''


class SourceAPI:
    '''Source-neutral names without duplicating the stored SCF state.'''

    @property
    def mf_hf(self):
        '''Compatibility view; the source object is stored only as mf_source.'''
        return self.mf_source

    @mf_hf.setter
    def mf_hf(self, value):
        self.mf_source = value

    @property
    def source_xc(self):
        return getattr(self.mf_source, 'xc', 'HF')

    @property
    def e_hf(self):
        return self.e_source

    @e_hf.setter
    def e_hf(self, value):
        self.e_source = value

    e_uhf = e_hf

    @property
    def dm_hf(self):
        return self.dm_source

    @dm_hf.setter
    def dm_hf(self, value):
        self.dm_source = value

    def kernel_hf(self, dm0=None, **kwargs):
        return self.kernel_source(dm0=dm0, **kwargs)

    kernel_uhf = kernel_hf

    def run_hf(self, dm0=None, **kwargs):
        return self.run_source(dm0=dm0, **kwargs)

    run_uhf = run_hf

    def make_hf_rdm1(self):
        return self.make_source_rdm1()

    make_uhf_rdm1 = make_hf_rdm1

    def _prepare_source_grid(self):
        '''Rebuild changed source quadrature, but reuse an unchanged grid.

        Native SCF can retain built points after grids.level is assigned.
        Rerunning the source must not use old points with new grid settings.
        Geometry reset is already handled by the native scanner lifecycle.
        '''
        from yescape_grad.cpu.energy.rnsc import _source_dfa_signature
        if hasattr(self.mf_source, 'xc') and self._source_signature is not None:
            old = self._source_signature[-1][3:]
            new = _source_dfa_signature(self.mf_source)[3:]
            if old != new:
                self.mf_source.grids.reset()


def make_source(mol, hf_type, ks_type, mf_hf, mf_source, source_xc):
    '''Build or validate a conventional source; DF is applied by the workflow.

No SCF options or grids of a supplied object are silently replaced. Exact
native types exclude ROHF, solvent, relativistic and other unvalidated
wrappers, as in the original HF-only interface.
'''
    if hasattr(mol, 'a'):
        raise NotImplementedError('This workflow supports molecules, not PBC cells')
    if source_xc is not None and not isinstance(source_xc, str):
        raise TypeError('source_xc must be a functional string or None')
    if mf_hf is not None and mf_source is not None:
        raise ValueError('Specify mf_source or legacy mf_hf, not both')
    source = mf_source if mf_source is not None else mf_hf
    if source is not None and source_xc is not None:
        raise ValueError('Configure XC on mf_source or pass source_xc, not both')
    supplied = source is not None
    if source is None:
        if source_xc is None or source_xc.upper() == 'HF':
            source = hf_type(mol)
        else:
            source = ks_type(mol, xc=source_xc)
            # The analytic moving-grid derivative uses a fixed atom-grid rule,
            # not a density-dependent point-removal mask.
            source.small_rho_cutoff = 0
    if type(source) not in (hf_type, ks_type):
        raise TypeError('mf_source must be a conventional native HF or KS object '
                        'with matching spin/backend; apply density_fit afterward')
    if source.mol is not mol:
        raise ValueError('mf_source and workflow must reference the same Mole')
    return source, supplied
