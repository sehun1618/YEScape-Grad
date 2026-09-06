r'''Real-Gamma Gaussian-DF source-RHF response contractions.

The public PySCF response contract maps an AO transition density to
``J[D] - K[D]/2``.  In the RHF orbital Hessian used by a Z-vector solve,
however, every trial density already has the known cross-factor form

.. math::

   D[X] = 2\{(C_v X) C_o^T + C_o (C_v X)^T\}.

The production action contracts only ``(J[D]-K[D]/2) C_occ`` before the final
virtual projection.  Complete AO Coulomb and exchange matrices are therefore
not materialized for every Krylov vector.  The previous factor-aware AO-matrix
implementation remains available as :func:`gen_vind_ao_reference`; it is an
explicit numerical reference, not the production default.
'''

import time

import cupy as cp
import numpy as np
from gpu4pyscf.lib.cupy_helper import asarray, contract, get_avail_mem
from gpu4pyscf.pbc import tools
from gpu4pyscf.pbc.df.fft_jk import _ewald_exxdiv_for_G0
from gpu4pyscf.pbc.gto.int1e import int1e_ovlp
from pyscf import lib
from pyscf.pbc.lib.kpts_helper import is_zero

_CACHE_MODES = ('recompute', 'occupied', 'cderi', 'all', 'auto')


def gen_vind(mf_hf, Co, Cv, cache_mode='recompute',
             memory_fraction=0.8, minimum_reserve_bytes=4 * 1024**3,
             profile_detail=False, capture_terminal_response=False):
    r'''Return the direct-VO Gaussian-DF RHF Hessian action.

    Args:
        mf_hf : gpu4pyscf.pbc.scf.hf.RHF
            Converged real-Gamma source RHF.  Its ``with_df`` object must
            expose GPU4PySCF's compressed Gamma CDERI and block iterator.
        Co, Cv : cupy.ndarray
            Occupied and virtual source-RHF coefficient matrices.

    Returns:
        callable
            Function mapping one or more ``X[a,i]`` trial amplitudes to
            ``C_v^T (J[D[X]] - K[D[X]]/2) C_o`` with the input shape.

    Notes:
        For one symmetric real-Gamma CDERI factor, define

        ``A_P = L_P C_o``, ``B_P = L_P (C_v X)`` and
        ``rho_P = Tr(C_o^T B_P)``.  The action accumulates

        ``(J-K/2) C_o = sum_P s_P [4 rho_P A_P``
        ``                            - B_P(A_P^T C_o)``
        ``                            - A_P(B_P^T C_o)]``.

        Its largest accumulated potential is ``(nao,nocc)`` rather than two
        ``(nao,nao)`` AO matrices.  :func:`gen_vind_ao_reference` retains those
        matrices for regression tests and numerical diagnosis.
    '''
    return _DirectVOResponse(
        mf_hf, Co, Cv, cache_mode=cache_mode,
        memory_fraction=memory_fraction,
        minimum_reserve_bytes=minimum_reserve_bytes,
        profile_detail=profile_detail,
        capture_terminal_response=capture_terminal_response)


class _DirectVOResponse:
    r'''One bounded-lifetime direct-VO response execution context.

    ``A_P = L_P C_o`` and ``H_P = A_P^T C_o`` do not depend on the Krylov
    trial vector.  The ``occupied`` cache retains those two tensors, while
    the ``cderi`` cache retains only the compressed device CDERI.  ``all``
    combines both.  Dense ``(naux,nao,nao)`` CDERI is never persistent.

    The object is intentionally owned by one Z-vector solve.  Call
    :meth:`close` in ``finally`` so no orbital- or geometry-dependent cache
    can leak into a later scanner call or the nuclear-response stage.
    '''

    backend = 'gaussian_df_direct_vo'

    def __init__(self, mf_hf, Co, Cv, cache_mode='recompute',
                 memory_fraction=0.8,
                 minimum_reserve_bytes=4 * 1024**3,
                 profile_detail=False, capture_terminal_response=False):
        self.provider = mf_hf.with_df
        self.source_object_id = id(mf_hf)
        self.exxdiv = mf_hf.exxdiv
        self.Co = _real_gamma_matrix(Co, 'Co')
        self.Cv = _real_gamma_matrix(Cv, 'Cv')
        if self.Co.shape[0] != self.Cv.shape[0]:
            raise ValueError('occupied and virtual AO dimensions differ')
        _check_provider(self.provider, self.Co.shape[0])
        self.nvir = int(self.Cv.shape[1])
        self.nocc = int(self.Co.shape[1])
        self.nao = int(self.Co.shape[0])
        self.naux = int(self.provider.get_naoaux())
        self.ewald = _make_ewald_intermediate(
            self.provider, self.Co, mf_hf.exxdiv)
        self.profile_detail = bool(profile_detail)
        # The terminal response is an experiment-only, bounded observable.
        # Retaining one ``(nao,nocc)`` response column block is much smaller
        # than another CDERI action and lets the caller recover the Pulay OO
        # projection only after proving that the last action was evaluated at
        # the accepted Z vector.  It is disabled by default, so production
        # lifetime and allocation behavior remain unchanged.
        self.capture_terminal_response = bool(capture_terminal_response)
        self._terminal_amplitude = None
        self._terminal_response_occ = None
        self._terminal_action_ordinal = None
        self.requested_cache_mode = str(cache_mode).lower()
        if self.requested_cache_mode not in _CACHE_MODES:
            raise ValueError(
                'cache_mode must be recompute, occupied, cderi, all, or auto')
        if not 0.0 < float(memory_fraction) <= 1.0:
            raise ValueError('memory_fraction must be in (0,1]')
        self.memory_fraction = float(memory_fraction)
        self.minimum_reserve_bytes = max(0, int(minimum_reserve_bytes))
        self._entries = []
        self._prepared = False
        self._closed = False
        self.profile = self._make_profile()

    def _make_profile(self):
        '''Build the stable, JSON-serializable action/cache ledger.'''
        available = int(get_avail_mem(exclude_memory_pool=True))
        itemsize = np.dtype(np.float64).itemsize
        occupied = self.naux * (
            self.nao * self.nocc + self.nocc * self.nocc) * itemsize
        raw = self.provider._cderi[0]
        cderi = int(raw.size) * np.dtype(raw.dtype).itemsize
        extra = 2 * self.nao * self.nocc + self.nocc * self.nocc
        block_size = int(self.provider.get_blksize(extra=extra))
        working = block_size * (
            self.nao * self.nao + 2 * self.nao * self.nocc
            + self.nocc * self.nocc) * itemsize
        working += 2 * self.nao * self.nocc * itemsize
        reserve = max(
            self.minimum_reserve_bytes,
            int((1.0 - self.memory_fraction) * available))
        budget = max(0, available - reserve - working)
        estimates = {
            'recompute': 0,
            'occupied': int(occupied),
            'cderi': int(cderi),
            'all': int(occupied + cderi),
        }
        requested = self.requested_cache_mode
        if requested == 'auto':
            resolved = next((name for name in ('all', 'occupied', 'cderi')
                             if estimates[name] <= budget), 'recompute')
        elif estimates[requested] <= budget:
            resolved = requested
        else:
            resolved = 'recompute'
        fallback = None
        if resolved != requested and requested != 'auto':
            fallback = (
                f'predicted cache {estimates[requested]} bytes exceeds '
                f'bounded budget {budget} bytes')
        elif requested == 'auto' and resolved != 'all':
            fallback = (
                f'auto selected {resolved}; larger cache modes exceed '
                f'bounded budget {budget} bytes')
        pool = cp.get_default_memory_pool()
        return {
            'requested_cache_mode': requested,
            'resolved_cache_mode': resolved,
            'fallback_reason': fallback,
            'available_device_bytes': available,
            'minimum_reserve_bytes': self.minimum_reserve_bytes,
            'resolved_reserve_bytes': reserve,
            'predicted_action_working_bytes': int(working),
            'cache_budget_bytes': int(budget),
            'predicted_cache_bytes': estimates,
            'predicted_resolved_cache_bytes': int(estimates[resolved]),
            'cache_identity': {
                'source_object_id': self.source_object_id,
                'provider_object_id': id(self.provider),
                'cderi_object_id': id(raw),
                'cderi_layout_object_id': id(self.provider._cderi_idx),
                'occupied_device_pointer': int(self.Co.data.ptr),
                'occupied_shape': [int(value) for value in self.Co.shape],
                'virtual_shape': [int(value) for value in self.Cv.shape],
                'ao_ordering': 'provider_compressed_pair_mapping',
                'exxdiv': self.exxdiv,
            },
            'capture_terminal_response': self.capture_terminal_response,
            'terminal_response_lookup_count': 0,
            'terminal_response_hit_count': 0,
            'terminal_response_last_match': None,
            'terminal_response_last_reason': None,
            'actual_cache_bytes': 0,
            'block_size': block_size,
            'action_count': 0,
            'cache_hit_count': 0,
            'cache_miss_count': 0,
            'cache_build_seconds': 0.0,
            'cache_release_seconds': 0.0,
            'host_to_device_cderi_bytes': 0,
            'host_to_device_cderi_seconds': 0.0,
            'cderi_unpack_seconds': 0.0,
            'left_contraction_seconds': 0.0,
            'b_contraction_seconds': 0.0,
            'a_contraction_seconds': 0.0,
            'occupied_metric_seconds': 0.0,
            'remaining_jk_contraction_seconds': 0.0,
            'response_diagonal_build_count': 0,
            'response_diagonal_build_seconds': 0.0,
            'response_diagonal_approximation': None,
            'response_diagonal_virtual_block_size': None,
            'response_diagonal_min': None,
            'response_diagonal_max': None,
            'ewald_seconds': 0.0,
            'final_projection_seconds': 0.0,
            'action_seconds': 0.0,
            'unattributed_action_seconds': 0.0,
            'pool_used_bytes_initial': int(pool.used_bytes()),
            'pool_total_bytes_initial': int(pool.total_bytes()),
            'pool_actions': [],
            'pool_used_bytes_after_close': None,
            'pool_total_bytes_after_close': None,
            'closed': False,
        }

    def __call__(self, x):
        if self._closed:
            raise RuntimeError('direct-VO response cache is already closed')
        self._prepare()
        shape = cp.asarray(x).shape
        amplitudes = cp.asarray(
            x, dtype=cp.float64).reshape(-1, self.nvir, self.nocc)
        values = [self._contract_one(amplitude) for amplitude in amplitudes]
        return cp.stack(values).reshape(shape)

    def _prepare(self):
        '''Construct only the cache selected by the bounded memory plan.'''
        if self._prepared:
            return
        self._prepared = True
        mode = self.profile['resolved_cache_mode']
        if mode == 'recompute':
            return
        start = self._clock_start()
        try:
            for cderi, packed, sign in self._iter_provider_blocks():
                entry = {'sign': float(sign)}
                if mode in ('cderi', 'all'):
                    entry['packed'] = packed.copy()
                if mode in ('occupied', 'all'):
                    a_p = contract('Lpq,qi->Lpi', cderi, self.Co)
                    h_p = contract('Lpi,pj->Lij', a_p, self.Co)
                    entry['a_p'] = a_p
                    entry['h_p'] = h_p
                entry['width'] = int(cderi.shape[0])
                self._entries.append(entry)
            self.profile['actual_cache_bytes'] = int(sum(
                value.nbytes for entry in self._entries
                for value in entry.values() if isinstance(value, cp.ndarray)))
        except (cp.cuda.memory.OutOfMemoryError, MemoryError) as error:
            self._entries.clear()
            self.profile['resolved_cache_mode'] = 'recompute'
            self.profile['fallback_reason'] = (
                f'cache construction failed safely: {type(error).__name__}: '
                f'{error}')
            self.profile['actual_cache_bytes'] = 0
        finally:
            self.profile['cache_build_seconds'] += self._clock_stop(start)

    def _contract_one(self, amplitude):
        '''Contract one Krylov vector and accumulate an optional time ledger.'''
        action_start = self._clock_start()
        component_names = (
            'left_contraction_seconds', 'host_to_device_cderi_seconds',
            'cderi_unpack_seconds', 'b_contraction_seconds',
            'a_contraction_seconds', 'occupied_metric_seconds',
            'remaining_jk_contraction_seconds', 'ewald_seconds',
            'final_projection_seconds')
        components_before = sum(
            self.profile[name] for name in component_names)
        pool = cp.get_default_memory_pool()
        action_record = None
        if self.profile_detail:
            action_record = {
                'pool_used_bytes_before': int(pool.used_bytes()),
                'pool_total_bytes_before': int(pool.total_bytes()),
            }
        start = self._clock_start()
        left = contract('pa,ai->pi', self.Cv, amplitude)
        self.profile['left_contraction_seconds'] += self._clock_stop(start)
        response_occ = cp.zeros(
            (self.nao, self.nocc), dtype=cp.float64)
        mode = self.profile['resolved_cache_mode']
        if mode == 'recompute':
            iterator = (
                (cderi, None, None, sign)
                for cderi, _packed, sign in self._iter_provider_blocks())
            self.profile['cache_miss_count'] += 1
        else:
            iterator = self._iter_cached_blocks(mode)
            self.profile['cache_hit_count'] += 1
        for cderi, a_p, h_p, sign in iterator:
            start = self._clock_start()
            b_p = contract('Lpq,qi->Lpi', cderi, left)
            self.profile['b_contraction_seconds'] += self._clock_stop(start)
            if a_p is None:
                start = self._clock_start()
                a_p = contract('Lpq,qi->Lpi', cderi, self.Co)
                self.profile['a_contraction_seconds'] += self._clock_stop(start)
            start = self._clock_start()
            rho = contract('Lpi,pi->L', b_p, self.Co)
            if h_p is None:
                h_p = contract('Lpi,pj->Lij', a_p, self.Co)
            self.profile['occupied_metric_seconds'] += self._clock_stop(start)
            start = self._clock_start()
            contract('L,Lpi->pi', rho, a_p, alpha=4.0 * sign,
                     beta=1.0, out=response_occ)
            contract('Lpi,Lij->pj', b_p, h_p, alpha=-sign,
                     beta=1.0, out=response_occ)
            b_metric = contract('Lpi,pj->Lij', b_p, self.Co)
            contract('Lpi,Lij->pj', a_p, b_metric, alpha=-sign,
                     beta=1.0, out=response_occ)
            self.profile['remaining_jk_contraction_seconds'] += (
                self._clock_stop(start))
        start = self._clock_start()
        if self.ewald is not None:
            overlap, overlap_occ, madelung = self.ewald
            overlap_left = contract('pq,qi->pi', overlap, left)
            occ_metric = contract('pi,pj->ij', self.Co, overlap_occ)
            left_metric = contract('pi,pj->ij', left, overlap_occ)
            contract('pi,ij->pj', overlap_left, occ_metric,
                     alpha=-madelung, beta=1.0, out=response_occ)
            contract('pi,ij->pj', overlap_occ, left_metric,
                     alpha=-madelung, beta=1.0, out=response_occ)
        self.profile['ewald_seconds'] += self._clock_stop(start)
        start = self._clock_start()
        result = contract('ap,pi->ai', self.Cv.T, response_occ)
        self.profile['final_projection_seconds'] += self._clock_stop(start)
        if self.capture_terminal_response:
            # Keep the last *input* as an immutable value, rather than trusting
            # a CuPy pointer that a Krylov solver may overwrite in-place.  The
            # response columns themselves are already a fresh allocation for
            # this action.  A later exact elementwise equality check is the
            # gate that makes this a reuse rather than an assumption about
            # which Krylov call happened to be terminal.
            self._terminal_amplitude = amplitude.copy()
            self._terminal_response_occ = response_occ
            self._terminal_action_ordinal = self.profile['action_count'] + 1
        action_seconds = self._clock_stop(action_start)
        self.profile['action_seconds'] += action_seconds
        components_after = sum(
            self.profile[name] for name in component_names)
        self.profile['unattributed_action_seconds'] += max(
            0.0, action_seconds - (components_after - components_before))
        self.profile['action_count'] += 1
        if action_record is not None:
            self._synchronize()
            action_record.update({
                'pool_used_bytes_after': int(pool.used_bytes()),
                'pool_total_bytes_after': int(pool.total_bytes()),
            })
            self.profile['pool_actions'].append(action_record)
        return result

    def get_response_diagonal(self, approximation='mo_coupling'):
        r'''Return one GDF-MO response diagonal for Krylov preconditioning.

        This is the diagonal of the same linear map evaluated by
        :meth:`__call__`; it does not include the canonical orbital-energy
        gap.  For one real-Gamma factor :math:`L_P`, define

        .. math::

           Q^P_{ai} = C_{va}^T L_P C_{oi},\qquad
           V^P_{aa} = C_{va}^T L_P C_{va},\qquad
           O^P_{ii} = C_{oi}^T L_P C_{oi}.

        Direct differentiation of the restricted ``2J-K`` response gives the
        exact diagonal

        .. math::

           \operatorname{diag}(A_{2e})_{ai}
             = \sum_P s_P\left[3(Q^P_{ai})^2
                    - V^P_{aa}O^P_{ii}\right].

        ``approximation='exact_reference'`` evaluates that expression and the
        Ewald term, but requires the expensive :math:`L_PC_v` transform.  It
        is retained only as a small-system numerical reference.

        The executable acceleration candidate uses
        ``approximation='mo_coupling'`` and retains only the inexpensive
        positive-metric :math:`3(Q^P_{ai})^2` coupling.  It omits the costly
        virtual--virtual and Ewald diagonal pieces.  This is an approximation
        to the *preconditioner*, never to the response operator: the full GDF
        Hessian action and the independently reconstructed FP64 physical
        residual still determine the accepted Z vector.

        Args:
            approximation : {'mo_coupling', 'exact_reference'}
                Select the bounded experiment or the small-system algebra
                reference.

        Returns:
            cupy.ndarray of shape (nvir,nocc)
                Approximate or exact-reference two-electron response diagonal for
                the active provider and orbitals.
        '''
        if self._closed:
            raise RuntimeError('direct-VO response cache is already closed')
        approximation = str(approximation).strip().lower()
        if approximation not in ('mo_coupling', 'exact_reference'):
            raise ValueError(
                'response diagonal approximation must be mo_coupling or '
                'exact_reference')
        self._prepare()
        started = self._clock_start()
        diagonal = cp.zeros((self.nvir, self.nocc), dtype=cp.float64)
        mode = self.profile['resolved_cache_mode']
        if mode == 'recompute':
            iterator = (
                (cderi, None, None, sign)
                for cderi, _packed, sign in self._iter_provider_blocks())
        else:
            iterator = self._iter_cached_blocks(mode)
        minimum_virtual_block = self.nvir
        for cderi, a_p, h_p, sign in iterator:
            if a_p is None:
                a_p = contract('Lpq,qi->Lpi', cderi, self.Co)
            if approximation == 'exact_reference' and h_p is None:
                h_p = contract('Lpi,pj->Lij', a_p, self.Co)
            width = int(cderi.shape[0])
            itemsize = np.dtype(np.float64).itemsize
            per_virtual = width * self.nocc * itemsize
            if approximation == 'exact_reference':
                per_virtual += width * (self.nao + 1) * itemsize
            # Limit optional diagonal temporaries to at most one GiB and at
            # most one fifth of currently free VRAM.  This changes only
            # scheduling, not the preconditioner values or response equation.
            available = int(get_avail_mem(exclude_memory_pool=True))
            budget = min(1024**3, max(1, available // 5))
            virtual_block = max(
                1, min(self.nvir, budget // max(1, per_virtual)))
            minimum_virtual_block = min(
                minimum_virtual_block, virtual_block)
            if approximation == 'exact_reference':
                o_diag = cp.diagonal(h_p, axis1=1, axis2=2)
            for a0, a1 in lib.prange(0, self.nvir, virtual_block):
                cv = self.Cv[:, a0:a1]
                q_vo = contract('ap,Lpi->Lai', cv.T, a_p)
                block = 3.0 * contract('Lai,Lai->ai', q_vo, q_vo)
                if approximation == 'exact_reference':
                    l_cv = contract('Lpq,qa->Lpa', cderi, cv)
                    v_diag = contract('Lpa,pa->La', l_cv, cv)
                    block -= contract('La,Li->ai', v_diag, o_diag)
                diagonal[a0:a1] += float(sign) * block

        if approximation == 'exact_reference' and self.ewald is not None:
            overlap, overlap_occ, madelung = self.ewald
            overlap_vir = contract('pq,qa->pa', overlap, self.Cv)
            vv_diag = contract('pa,pa->a', self.Cv, overlap_vir)
            oo_diag = contract('pi,pi->i', self.Co, overlap_occ)
            vo = contract('ap,pi->ai', self.Cv.T, overlap_occ)
            diagonal -= madelung * (
                contract('a,i->ai', vv_diag, oo_diag) + vo * vo)

        if not bool(cp.all(cp.isfinite(diagonal))):
            raise FloatingPointError(
                'GDF response diagonal contains non-finite values')
        self.profile['response_diagonal_build_count'] += 1
        self.profile['response_diagonal_build_seconds'] += self._clock_stop(
            started)
        self.profile['response_diagonal_approximation'] = approximation
        self.profile['response_diagonal_virtual_block_size'] = int(
            minimum_virtual_block)
        if diagonal.size:
            self.profile['response_diagonal_min'] = float(
                cp.min(diagonal).get())
            self.profile['response_diagonal_max'] = float(
                cp.max(diagonal).get())
        return diagonal

    def get_terminal_occupied_projection(
            self, amplitude, *, source_object_id=None,
            provider_object_id=None, occupied_device_pointer=None):
        r'''Return the Pulay OO response only for the exact last amplitude.

        The direct-VO Hessian action uses the restricted transition density

        .. math::

           D^{\mathrm{Hess}}[X]
             = 2\{(C_vX)C_o^T + C_o(C_vX)^T\},

        whereas the overlap-response helper uses the density without the
        leading factor two.  Linearity of the source-HF response therefore
        gives

        .. math::

           C_o^T V[D_Z] C_o
             = \tfrac12 C_o^T V[D^{\mathrm{Hess}}[Z]] C_o.

        Args:
            amplitude : cupy.ndarray of shape (nvir,nocc)
                Accepted Z vector whose exact value must equal the most recent
                response-action input.

        Kwargs:
            source_object_id, provider_object_id : int or None
                Optional caller-side identities of the source mean-field and
                active DF provider.  A supplied mismatch rejects reuse.
            occupied_device_pointer : int or None
                Optional caller-side device pointer for ``C_o``.  This proves
                that the small OO projection is formed in the same occupied
                basis used by the gradient driver.

        Returns:
            tuple
                ``(projection, metadata)``.  ``projection`` is ``None`` when
                capture was disabled, the context is closed, or exact shape,
                dtype-converted value equality cannot prove identity.  The
                JSON-safe metadata states the reason, so callers can fall back
                to the ordinary AO ``fvind`` path without changing a result.
        '''
        metadata = {
            'requested': True,
            'matched': False,
            'reason': None,
            'response_backend': self.backend,
            'source_object_id': self.source_object_id,
            'provider_object_id': (
                None if self.provider is None else id(self.provider)),
            'hessian_density_scale': 2.0,
            'overlap_density_scale': 1.0,
            'projection_scale': 0.5,
            'action_ordinal': self._terminal_action_ordinal,
            'identity_checks': {
                'source_object': source_object_id is not None,
                'provider_object': provider_object_id is not None,
                'occupied_device_pointer': (
                    occupied_device_pointer is not None),
                'amplitude_exact_value': True,
            },
        }
        self.profile['terminal_response_lookup_count'] += 1
        if self._closed:
            metadata['reason'] = 'response context is already closed'
        elif not self.capture_terminal_response:
            metadata['reason'] = 'terminal response capture was disabled'
        elif (self._terminal_amplitude is None
              or self._terminal_response_occ is None):
            metadata['reason'] = 'no completed response action was captured'
        elif (source_object_id is not None
              and int(source_object_id) != self.source_object_id):
            metadata['reason'] = 'source mean-field object identity changed'
        elif (provider_object_id is not None
              and int(provider_object_id) != id(self.provider)):
            metadata['reason'] = 'DF provider object identity changed'
        elif (occupied_device_pointer is not None
              and int(occupied_device_pointer) != int(self.Co.data.ptr)):
            metadata['reason'] = 'occupied coefficient identity changed'
        else:
            expected = cp.asarray(amplitude, dtype=cp.float64)
            if expected.shape != (self.nvir, self.nocc):
                metadata['reason'] = (
                    f'expected amplitude shape {(self.nvir, self.nocc)}, '
                    f'got {expected.shape}')
            elif not bool(cp.array_equal(
                    expected, self._terminal_amplitude)):
                metadata['reason'] = (
                    'last response-action amplitude does not exactly equal '
                    'the accepted Z vector')
            else:
                projection = 0.5 * contract(
                    'pi,pj->ij', self.Co, self._terminal_response_occ)
                if bool(cp.all(cp.isfinite(projection))):
                    metadata['matched'] = True
                    self.profile['terminal_response_hit_count'] += 1
                    self.profile['terminal_response_last_match'] = True
                    self.profile['terminal_response_last_reason'] = None
                    return projection, metadata
                metadata['reason'] = 'occupied projection is not finite'
        self.profile['terminal_response_last_match'] = False
        self.profile['terminal_response_last_reason'] = metadata['reason']
        return None, metadata

    def _iter_provider_blocks(self):
        '''Copy and unpack provider factors with an auditable byte ledger.'''
        block_size = self.profile['block_size']
        raw = self.provider._cderi[0]
        mapping, _diag = self.provider._cderi_idx
        mapping = asarray(mapping)
        rows, cols = divmod(mapping, self.nao)
        dense = cp.zeros((block_size, self.nao, self.nao), dtype=cp.float64)
        for p0, p1 in lib.prange(0, self.naux, block_size):
            start = self._clock_start()
            # ``asarray`` is GPU4PySCF's explicit NumPy/HDF5-to-device copy,
            # not a mapped-host zero-copy view.  This preserves its workaround
            # for large mapped allocations while accepting device factors too.
            packed = asarray(raw[p0:p1])
            self.profile['host_to_device_cderi_seconds'] += (
                self._clock_stop(start))
            self.profile['host_to_device_cderi_bytes'] += int(packed.nbytes)
            start = self._clock_start()
            cderi = dense[:p1-p0]
            cderi[:, cols, rows] = cderi[:, rows, cols] = packed
            self.profile['cderi_unpack_seconds'] += self._clock_stop(start)
            yield cderi, packed, 1.0

    def _iter_cached_blocks(self, mode):
        '''Reconstruct one dense block from compressed cache when required.'''
        mapping, _diag = self.provider._cderi_idx
        mapping = cp.asarray(mapping)
        rows, cols = divmod(mapping, self.nao)
        for index, entry in enumerate(self._entries):
            a_p = entry.get('a_p')
            h_p = entry.get('h_p')
            if mode == 'occupied':
                # B_P still needs L_P, so reload the compressed host block.
                p0 = index * self.profile['block_size']
                p1 = min(self.naux, p0 + entry['width'])
                raw = self.provider._cderi[0][p0:p1]
                start = self._clock_start()
                packed = asarray(raw)
                self.profile['host_to_device_cderi_seconds'] += (
                    self._clock_stop(start))
                self.profile['host_to_device_cderi_bytes'] += int(packed.nbytes)
            else:
                packed = entry['packed']
            start = self._clock_start()
            cderi = cp.zeros((entry['width'], self.nao, self.nao),
                             dtype=cp.float64)
            cderi[:, cols, rows] = cderi[:, rows, cols] = packed
            self.profile['cderi_unpack_seconds'] += self._clock_stop(start)
            yield cderi, a_p, h_p, entry['sign']

    def close(self):
        '''Release all geometry/orbital-dependent device cache references.'''
        if self._closed:
            return self.profile
        start = self._clock_start()
        self._entries.clear()
        # The response context belongs to one Z solve.  Clear even the
        # non-cache references so a retained callable cannot keep an old
        # provider, orbital matrices, or Ewald intermediate alive after a
        # scanner reset, solver exception, or ``KeyboardInterrupt``.
        self.provider = None
        self.Co = None
        self.Cv = None
        self.ewald = None
        self._terminal_amplitude = None
        self._terminal_response_occ = None
        self._closed = True
        self.profile['cache_release_seconds'] += self._clock_stop(start)
        pool = cp.get_default_memory_pool()
        self.profile['pool_used_bytes_after_close'] = int(pool.used_bytes())
        self.profile['pool_total_bytes_after_close'] = int(pool.total_bytes())
        self.profile['closed'] = True
        return self.profile

    def _clock_start(self):
        if not self.profile_detail:
            return None
        self._synchronize()
        return time.perf_counter()

    def _clock_stop(self, start):
        if start is None:
            return 0.0
        self._synchronize()
        return time.perf_counter() - start

    @staticmethod
    def _synchronize():
        cp.cuda.Stream.null.synchronize()


def gen_vind_ao_reference(mf_hf, Co, Cv):
    r'''Return the former factor-aware full-AO response as a reference.

    Args:
        mf_hf, Co, Cv
            Same objects accepted by :func:`gen_vind`.

    Returns:
        callable
            Numerically independent CDERI action that constructs complete AO
            ``J[D[X]]`` and ``K[D[X]]`` matrices before their VO projection.

    Notes:
        This route is retained deliberately for parity tests and diagnosis.
        It is not selected by the production gradient driver.
    '''
    provider = mf_hf.with_df
    Co = _real_gamma_matrix(Co, 'Co')
    Cv = _real_gamma_matrix(Cv, 'Cv')
    if Co.shape[0] != Cv.shape[0]:
        raise ValueError('occupied and virtual AO dimensions differ')
    _check_provider(provider, Co.shape[0])
    nvir = int(Cv.shape[1])
    nocc = int(Co.shape[1])

    def vind(x):
        shape = cp.asarray(x).shape
        amplitudes = cp.asarray(x, dtype=cp.float64).reshape(-1, nvir, nocc)
        values = [_contract_one_ao_reference(
            provider, Co, Cv, amplitude, mf_hf.exxdiv)
                  for amplitude in amplitudes]
        return cp.stack(values).reshape(shape)

    vind.backend = 'gaussian_df_cross_factor_ao_reference'
    return vind


def _contract_one_direct_vo(provider, Co, Cv, amplitude, ewald):
    '''Contract one trial through occupied columns and then into VO space.'''
    left = contract('pa,ai->pi', Cv, amplitude)
    nao, nocc = left.shape
    response_occ = cp.zeros((nao, nocc), dtype=cp.float64)
    naux = int(provider.get_naoaux())
    # One unpacked CDERI block, A_P, B_P and one reusable occupied-occupied
    # contraction are live together.  Include every per-auxiliary workspace
    # in the inherited provider's device-memory block planner.
    extra = 2 * nao * nocc + nocc * nocc
    blksize = provider.get_blksize(extra=extra)
    aux_iter = lib.prange(0, naux, blksize)
    for cderi, _packed, sign in provider.loop_gamma_point(
            blksize, unpack=True, aux_iter=aux_iter):
        cderi = cp.asarray(cderi)
        b_p = contract('Lpq,qi->Lpi', cderi, left)
        a_p = contract('Lpq,qi->Lpi', cderi, Co)

        rho = contract('Lpi,pi->L', b_p, Co)
        contract('L,Lpi->pi', rho, a_p, alpha=4.0 * sign,
                 beta=1.0, out=response_occ)

        # K[D[X]] = 2 sum_P (B_P A_P^T + A_P B_P^T).  Multiplication by
        # -1/2 in the restricted response cancels that leading factor two.
        occupied_metric = contract('Lpi,pj->Lij', a_p, Co)
        contract('Lpi,Lij->pj', b_p, occupied_metric, alpha=-sign,
                 beta=1.0, out=response_occ)
        del occupied_metric
        occupied_metric = contract('Lpi,pj->Lij', b_p, Co)
        contract('Lpi,Lij->pj', a_p, occupied_metric, alpha=-sign,
                 beta=1.0, out=response_occ)

    if ewald is not None:
        overlap, overlap_occ, madelung = ewald
        overlap_left = contract('pq,qi->pi', overlap, left)
        occ_metric = contract('pi,pj->ij', Co, overlap_occ)
        left_metric = contract('pi,pj->ij', left, overlap_occ)
        contract('pi,ij->pj', overlap_left, occ_metric, alpha=-madelung,
                 beta=1.0, out=response_occ)
        contract('pi,ij->pj', overlap_occ, left_metric, alpha=-madelung,
                 beta=1.0, out=response_occ)

    return contract('ap,pi->ai', Cv.T, response_occ)


def _contract_one_ao_reference(provider, Co, Cv, amplitude, exxdiv):
    '''Contract one trial through complete AO J/K matrices for parity tests.'''
    left = contract('pa,ai->pi', Cv, amplitude)
    nao, nocc = left.shape
    density = 2.0 * (
        contract('pi,qi->pq', left, Co)
        + contract('pi,qi->pq', Co, left))
    vj = cp.zeros((nao, nao), dtype=cp.float64)
    vk = cp.zeros_like(vj)
    naux = int(provider.get_naoaux())
    # One unpacked CDERI block plus C*left and C*Co are live together.  The
    # inherited planner accounts for exactly those per-auxiliary workspaces.
    blksize = provider.get_blksize(extra=2 * nao * nocc)
    aux_iter = lib.prange(0, naux, blksize)
    for cderi, _packed, sign in provider.loop_gamma_point(
            blksize, unpack=True, aux_iter=aux_iter):
        cderi = cp.asarray(cderi)
        c_left = contract('Lpq,qi->Lpi', cderi, left)
        c_occ = contract('Lpq,qi->Lpi', cderi, Co)

        # Tr[C_L D[X]] = 4 Tr[C_L (C_v X) C_o^T] at real Gamma because
        # C_L, D[X] are symmetric.  Keeping this contraction factorized
        # avoids the generic O(naux*nao**3) exchange path.
        rho = contract('Lpi,pi->L', c_left, Co)
        contract('L,Lpq->pq', rho, cderi, alpha=4.0 * sign,
                 beta=1.0, out=vj)
        contract('Lpi,Lqi->pq', c_left, c_occ, alpha=2.0 * sign,
                 beta=1.0, out=vk)
        contract('Lpi,Lqi->pq', c_occ, c_left, alpha=2.0 * sign,
                 beta=1.0, out=vk)

    if str(exxdiv).lower() == 'ewald':
        _ewald_exxdiv_for_G0(
            provider.cell, np.zeros(3), density[None], vk[None])
    elif exxdiv is not None:
        raise NotImplementedError(
            f'Gaussian-DF RHF response does not support exxdiv={exxdiv!r}')
    response = vj - 0.5 * vk
    response_vo = contract('ap,pq->aq', Cv.T, response)
    return contract('aq,qi->ai', response_vo, Co)


def _make_ewald_intermediate(provider, Co, exxdiv):
    '''Return fixed overlap quantities for direct occupied-column projection.'''
    if str(exxdiv).lower() == 'ewald':
        kpt = np.zeros(3)
        overlap = cp.asarray(int1e_ovlp(provider.cell, kpts=kpt))
        overlap_occ = contract('pq,qi->pi', overlap, Co)
        madelung = float(tools.madelung(provider.cell, kpt))
        return overlap, overlap_occ, madelung
    if exxdiv is not None:
        raise NotImplementedError(
            f'direct-VO GDF response does not support exxdiv={exxdiv!r}')
    return None


def _check_provider(provider, nao):
    '''Validate the inherited real-Gamma CDERI iterator contract.'''
    kpts = np.asarray(provider.kpts, dtype=float).reshape(-1, 3)
    if len(kpts) != 1 or not is_zero(kpts[0]):
        raise NotImplementedError('factor-aware RHF response requires Gamma')
    if provider.cell.dimension != 3:
        raise NotImplementedError('factor-aware RHF response requires a 3D cell')
    if provider._cderi is None:
        provider.build(j_only=False)
    if getattr(provider, '_cderip', None) is not None:
        raise NotImplementedError(
            'factor-aware RHF response requires a positive Gamma metric')
    if int(provider.nao) != int(nao):
        raise ValueError('CDERI and orbital AO dimensions differ')


def _real_gamma_matrix(value, name):
    '''Return one real rank-two Gamma array or reject complex storage.'''
    value = cp.asarray(value)
    if value.ndim != 2:
        raise ValueError(f'{name} must be a rank-two array')
    if cp.iscomplexobj(value):
        maximum = float(cp.max(cp.abs(value.imag)).get()) if value.size else 0.0
        if maximum > 1.0e-10:
            raise NotImplementedError(
                f'{name} has non-Gamma imaginary component {maximum:.3e}')
        value = value.real
    return cp.asarray(value, dtype=cp.float64)
