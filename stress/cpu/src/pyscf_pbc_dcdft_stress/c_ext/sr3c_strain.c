/* Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
 * Modifications Copyright 2026 Sehun.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * The shell traversal and packed-output helpers are adapted from PySCF's
 * lib/pbc/fill_ints_sr.c.  This callback keeps PySCF's screening and lattice
 * traversal but contracts the surviving image-resolved ip1/ip2 integrals
 * directly into the nine homogeneous-strain components.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <assert.h>
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "cint.h"

#define INTBUFMAX10 8000
#define MAX2(a, b) ((a) > (b) ? (a) : (b))

int GTOmax_shell_dim(int *ao_loc, int *shls_slice, int ncenter);
int GTOmax_cache_size(int (*intor)(), int *shls_slice, int ncenter,
                      int *atm, int natm, int *bas, int nbas, double *env);
int int3c2e_sph();
int int3c2e_cart();
int int3c2e_ip1_sph();
int int3c2e_ip1_cart();
int int3c2e_ip2_sph();
int int3c2e_ip2_cart();

static double distance_square(const double *ri, const double *rj)
{
        const double dx = ri[0] - rj[0];
        const double dy = ri[1] - rj[1];
        const double dz = ri[2] - rj[2];
        return dx * dx + dy * dy + dz * dz;
}

static void product_center(double *rc, const double *ri, const double *rj,
                           double ei, double ej)
{
        const double inverse = 1.0 / (ei + ej);
        rc[0] = (ri[0] * ei + rj[0] * ej) * inverse;
        rc[1] = (ri[1] * ei + rj[1] * ej) * inverse;
        rc[2] = (ri[2] * ei + rj[2] * ej) * inverse;
}

static void shift_center(double *env_loc, const double *env,
                         const double *Ls, int ptr, int image)
{
        env_loc[ptr] = env[ptr] + Ls[image * 3];
        env_loc[ptr + 1] = env[ptr + 1] + Ls[image * 3 + 1];
        env_loc[ptr + 2] = env[ptr + 2] + Ls[image * 3 + 2];
}

static int partition_aux_shells(int *locations, int *ao_loc, int shell0,
                                int shell1, int maximum_ao)
{
        int shell;
        int count = 0;
        int start = ao_loc[shell0];
        locations[0] = shell0;
        for (shell = shell0 + 1; shell < shell1; shell++) {
                assert(ao_loc[shell + 1] - ao_loc[shell] < maximum_ao);
                if (ao_loc[shell + 1] - start > maximum_ao) {
                        count++;
                        locations[count] = shell;
                        start = ao_loc[shell];
                }
        }
        count++;
        locations[count] = shell1;
        return count;
}

static void sort_packed_igtj(double *out, double *input, int *slice,
                             int *ao_loc, int comp, int ish, int jsh,
                             int ksh0, int ksh1)
{
        const int ish0 = slice[0];
        const int jsh0 = slice[2];
        const int aux0 = slice[4];
        const int aux1 = slice[5];
        const size_t naux = ao_loc[aux1] - ao_loc[aux0];
        const size_t offset0 = ao_loc[ish0] * (ao_loc[ish0] + 1) / 2;
        const size_t npair =
                ao_loc[slice[1]] * (ao_loc[slice[1]] + 1) / 2 - offset0;
        const size_t block_size = npair * naux;
        const int di = ao_loc[ish + 1] - ao_loc[ish];
        const int dj = ao_loc[jsh + 1] - ao_loc[jsh];
        const int dij = di * dj;
        const int jp = ao_loc[jsh] - ao_loc[jsh0];
        int i, j, k, ij, ksh, ic, dk, dijk;
        double *pin, *pout;

        out += (ao_loc[ish] * (ao_loc[ish] + 1) / 2 - offset0 + jp) * naux;
        for (ksh = ksh0; ksh < ksh1; ksh++) {
                dk = ao_loc[ksh + 1] - ao_loc[ksh];
                dijk = dij * dk;
                for (ic = 0; ic < comp; ic++) {
                        pout = out + block_size * ic + ao_loc[ksh] - ao_loc[aux0];
                        pin = input + dijk * ic;
                        for (i = 0; i < di; i++) {
                                for (j = 0; j < dj; j++) {
                                        ij = j * di + i;
                                        for (k = 0; k < dk; k++) {
                                                pout[j * naux + k] = pin[k * dij + ij];
                                        }
                                }
                                pout += (i + ao_loc[ish] + 1) * naux;
                        }
                }
                input += dijk * comp;
        }
}

static void sort_packed_ieqj(double *out, double *input, int *slice,
                             int *ao_loc, int comp, int ish, int jsh,
                             int ksh0, int ksh1)
{
        const int ish0 = slice[0];
        const int jsh0 = slice[2];
        const int aux0 = slice[4];
        const int aux1 = slice[5];
        const size_t naux = ao_loc[aux1] - ao_loc[aux0];
        const size_t offset0 = ao_loc[ish0] * (ao_loc[ish0] + 1) / 2;
        const size_t npair =
                ao_loc[slice[1]] * (ao_loc[slice[1]] + 1) / 2 - offset0;
        const size_t block_size = npair * naux;
        const int di = ao_loc[ish + 1] - ao_loc[ish];
        const int dij = di * di;
        const int jp = ao_loc[jsh] - ao_loc[jsh0];
        int i, j, k, ij, ksh, ic, dk, dijk;
        double *pin, *pout;

        out += (ao_loc[ish] * (ao_loc[ish] + 1) / 2 - offset0 + jp) * naux;
        for (ksh = ksh0; ksh < ksh1; ksh++) {
                dk = ao_loc[ksh + 1] - ao_loc[ksh];
                dijk = dij * dk;
                for (ic = 0; ic < comp; ic++) {
                        pout = out + block_size * ic + ao_loc[ksh] - ao_loc[aux0];
                        pin = input + dijk * ic;
                        for (i = 0; i < di; i++) {
                                for (j = 0; j <= i; j++) {
                                        ij = j * di + i;
                                        for (k = 0; k < dk; k++) {
                                                pout[j * naux + k] = pin[k * dij + ij];
                                        }
                                }
                                pout += (i + ao_loc[ish] + 1) * naux;
                        }
                }
                input += dijk * comp;
        }
}

static void accumulate_strain(int (*ip1)(), int (*ip2)(), double *out,
                              int nimages, int ish, int jsh, double *buffer,
                              double *env_loc, double *Ls, int *slice,
                              int *ao_loc, CINTOpt *optimizer,
                              int *refuniq_map, int *auxuniq_map,
                              int naux_unique, double *unique_exponents,
                              double *distance_cutoffs, double distance_step,
                              double *three_center_cutoffs, int *distance_locations,
                              int *atm, int natm, int *bas, int nbas,
                              double *env, int equal_shells)
{
        const int ish0 = slice[0];
        const int jsh0 = slice[2];
        const int aux0 = slice[4];
        const int aux1 = slice[5];
        int iptr, jptr, kptr;
        int di, dj, dij, maximum_aux_ao, maximum_block_ao;
        int aux_locations[aux1 - aux0 + 1];
        int nblocks, block, block0, block1;
        int q, x, y, ksh, dk, dijk;
        int iimage, jimage, shells[3];
        int Ish, Jsh, IJsh, Ksh, distance_bin;
        double *cutoff_base, *cutoff_aux;
        double *ri, *rj, *rk, rc[3];
        double dij2, pair_cutoff, inverse_step, triple_distance, cutoff;
        double ei, ej;
        double *ip1_buffer, *ip2_buffer, *strain_buffer, *cache, *target;

        ish += ish0;
        jsh += jsh0;
        iptr = atm[PTR_COORD + bas[ATOM_OF + ish * BAS_SLOTS] * ATM_SLOTS];
        jptr = atm[PTR_COORD + bas[ATOM_OF + jsh * BAS_SLOTS] * ATM_SLOTS];
        di = ao_loc[ish + 1] - ao_loc[ish];
        dj = ao_loc[jsh + 1] - ao_loc[jsh];
        dij = di * dj;
        maximum_aux_ao = GTOmax_shell_dim(ao_loc, slice + 4, 1);
        maximum_block_ao = MAX2(INTBUFMAX10 / dij, maximum_aux_ao);
        nblocks = partition_aux_shells(
                aux_locations, ao_loc, aux0, aux1, maximum_block_ao);

        Ish = refuniq_map[ish];
        Jsh = refuniq_map[jsh - nbas];
        IJsh = Ish >= Jsh ? Ish * (Ish + 1) / 2 + Jsh
                          : Jsh * (Jsh + 1) / 2 + Ish;
        cutoff_base = three_center_cutoffs
                + distance_locations[IJsh] * naux_unique;
        inverse_step = 1.0 / distance_step;
        pair_cutoff = distance_cutoffs[IJsh];
        ei = unique_exponents[Ish];
        ej = unique_exponents[Jsh];
        shells[0] = ish;
        shells[1] = jsh;

        ip1_buffer = buffer;
        ip2_buffer = ip1_buffer + 3 * dij * maximum_aux_ao;
        strain_buffer = ip2_buffer + 3 * dij * maximum_aux_ao;
        cache = strain_buffer + 9 * dij * maximum_block_ao;

        for (block = 0; block < nblocks; block++) {
                block0 = aux_locations[block];
                block1 = aux_locations[block + 1];
                maximum_block_ao = ao_loc[block1] - ao_loc[block0];
                memset(strain_buffer, 0,
                       sizeof(double) * 9 * dij * maximum_block_ao);

                for (iimage = 0; iimage < nimages; iimage++) {
                        shift_center(env_loc, env, Ls, iptr, iimage);
                        ri = env_loc + iptr;
                        for (jimage = 0; jimage < nimages; jimage++) {
                                shift_center(env_loc, env, Ls, jptr, jimage);
                                rj = env_loc + jptr;
                                dij2 = distance_square(ri, rj);
                                if (dij2 > pair_cutoff) {
                                        continue;
                                }
                                distance_bin = (int)(sqrt(dij2) * inverse_step);
                                cutoff_aux = cutoff_base + distance_bin * naux_unique;
                                product_center(rc, ri, rj, ei, ej);
                                target = strain_buffer;

                                for (ksh = block0; ksh < block1; ksh++) {
                                        shells[2] = ksh;
                                        dk = ao_loc[ksh + 1] - ao_loc[ksh];
                                        dijk = dij * dk;
                                        Ksh = auxuniq_map[ksh - 2 * nbas];
                                        cutoff = cutoff_aux[Ksh];
                                        kptr = atm[PTR_COORD
                                                + bas[ATOM_OF + ksh * BAS_SLOTS]
                                                * ATM_SLOTS];
                                        rk = env_loc + kptr;
                                        triple_distance = distance_square(rc, rk);
                                        if (triple_distance < cutoff) {
                                                if (!(*ip1)(ip1_buffer, NULL, shells,
                                                          atm, natm, bas, nbas,
                                                          env_loc, optimizer, cache)) {
                                                        memset(ip1_buffer, 0,
                                                               sizeof(double) * 3 * dijk);
                                                }
                                                if (!(*ip2)(ip2_buffer, NULL, shells,
                                                          atm, natm, bas, nbas,
                                                          env_loc, NULL, cache)) {
                                                        memset(ip2_buffer, 0,
                                                               sizeof(double) * 3 * dijk);
                                                }
                                                for (x = 0; x < 3; x++) {
                                                        for (y = 0; y < 3; y++) {
                                                                double *component =
                                                                        target + (x * 3 + y) * dijk;
                                                                const double wi = rj[y] - ri[y];
                                                                const double wk = rj[y] - rk[y];
                                                                for (q = 0; q < dijk; q++) {
                                                                        component[q] +=
                                                                                ip1_buffer[x * dijk + q] * wi
                                                                                + ip2_buffer[x * dijk + q] * wk;
                                                                }
                                                        }
                                                }
                                        }
                                        target += 9 * dijk;
                                }
                        }
                }
                if (equal_shells) {
                        sort_packed_ieqj(out, strain_buffer, slice, ao_loc, 9,
                                           ish, jsh, block0, block1);
                } else {
                        sort_packed_igtj(out, strain_buffer, slice, ao_loc, 9,
                                           ish, jsh, block0, block1);
                }
        }
}

static void fill_strain(int (*ip1)(), int (*ip2)(), double *out,
                        int comp, int nimages, int ish, int jsh,
                        double *buffer, double *env_loc, double *Ls,
                        int *slice, int *ao_loc, CINTOpt *optimizer,
                        int *refuniq_map, int *auxuniq_map, int naux_unique,
                        double *unique_exponents, double *distance_cutoffs,
                        double distance_step, double *three_center_cutoffs,
                        int *distance_locations, int *atm, int natm,
                        int *bas, int nbas, double *env)
{
        const int ip = ish + slice[0];
        const int jp = jsh + slice[2] - nbas;
        assert(comp == 9);
        if (ip >= jp) {
                accumulate_strain(
                        ip1, ip2, out, nimages, ish, jsh, buffer, env_loc, Ls,
                        slice, ao_loc, optimizer, refuniq_map, auxuniq_map,
                        naux_unique, unique_exponents, distance_cutoffs,
                        distance_step, three_center_cutoffs, distance_locations,
                        atm, natm, bas, nbas, env, ip == jp);
        }
}

#define DECLARE_FILL(NAME, IP2)                                             \
void NAME(int (*ip1)(), double *out, int comp, int nimages, int ish,        \
          int jsh, double *buffer, double *env_loc, double *Ls,             \
          int *slice, int *ao_loc, CINTOpt *optimizer,                      \
          int *refuniq_map, int *auxuniq_map, int naux_unique,              \
          double *unique_exponents, double *distance_cutoffs,               \
          double distance_step, double *three_center_cutoffs,               \
          int *distance_locations, int *atm, int natm, int *bas,            \
          int nbas, double *env)                                             \
{                                                                            \
        fill_strain(ip1, IP2, out, comp, nimages, ish, jsh, buffer,          \
                    env_loc, Ls, slice, ao_loc, optimizer, refuniq_map,      \
                    auxuniq_map, naux_unique, unique_exponents,              \
                    distance_cutoffs, distance_step, three_center_cutoffs,   \
                    distance_locations, atm, natm, bas, nbas, env);          \
}

DECLARE_FILL(PBCsr3c_strain_gs2_sph, int3c2e_ip2_sph)
DECLARE_FILL(PBCsr3c_strain_gs2_cart, int3c2e_ip2_cart)

/*
 * General k-pair short-range source and homogeneous-strain response.
 *
 * PySCF's periodic three-center driver accumulates the image sum after the
 * molecular integral call.  The homogeneous-strain derivative needs the
 * image displacement that belongs to each surviving integral, so it cannot
 * be reconstructed from the already summed tensor.  This driver retains the
 * same ME screening data but contracts the Bloch phase and image moment while
 * the two lattice indices are still available.
 *
 * Output is C-contiguous (10, ni, nj, naux): component zero is the central
 * source and components 1+x*3+y are dB/d epsilon_xy.  Distinct shell pairs
 * own disjoint AO blocks, making the OpenMP writes race-free.
 */
static void accumulate_kpair_strain(
        int (*value_intor)(), int (*ip1_intor)(), int (*ip2_intor)(),
        double complex *out, int nimages, const double *Ls,
        const double complex *phase_i, const double complex *phase_j,
        int ish, int jsh, int *slice, int *ao_loc,
        int *refuniq_map, int *auxuniq_map, int naux_unique,
        double *unique_exponents, double *distance_cutoffs,
        double distance_step, double *three_center_cutoffs,
        int *distance_locations, int *atm, int natm, int *bas, int nbas,
        double *env, double *env_loc, double *work)
{
        const int ish0 = slice[0];
        const int jsh0 = slice[2];
        const int aux0 = slice[4];
        const int aux1 = slice[5];
        const int ni = ao_loc[slice[1]] - ao_loc[ish0];
        const int nj = ao_loc[slice[3]] - ao_loc[jsh0];
        const int naux = ao_loc[aux1] - ao_loc[aux0];
        const size_t component_size = (size_t)ni * nj * naux;
        int iptr, jptr, kptr;
        int di, dj, dij, dk, dijk;
        int i, j, k, q, x, y, ksh, iimage, jimage, shells[3];
        int Ish, Jsh, IJsh, Ksh, distance_bin;
        double *cutoff_base, *cutoff_aux;
        double *ri, *rj, *rk, rc[3];
        double dij2, pair_cutoff, inverse_step, triple_distance, cutoff;
        double ei, ej, wi, wk;
        double *value_buffer, *ip1_buffer, *ip2_buffer, *cache;
        double complex phase;
        size_t output_index;

        ish += ish0;
        jsh += jsh0;
        iptr = atm[PTR_COORD + bas[ATOM_OF + ish * BAS_SLOTS] * ATM_SLOTS];
        jptr = atm[PTR_COORD + bas[ATOM_OF + jsh * BAS_SLOTS] * ATM_SLOTS];
        di = ao_loc[ish + 1] - ao_loc[ish];
        dj = ao_loc[jsh + 1] - ao_loc[jsh];
        dij = di * dj;
        value_buffer = work;
        ip1_buffer = value_buffer + dij * GTOmax_shell_dim(ao_loc, slice + 4, 1);
        ip2_buffer = ip1_buffer + 3 * dij * GTOmax_shell_dim(ao_loc, slice + 4, 1);
        cache = ip2_buffer + 3 * dij * GTOmax_shell_dim(ao_loc, slice + 4, 1);

        Ish = refuniq_map[ish];
        Jsh = refuniq_map[jsh - nbas];
        IJsh = Ish >= Jsh ? Ish * (Ish + 1) / 2 + Jsh
                          : Jsh * (Jsh + 1) / 2 + Ish;
        cutoff_base = three_center_cutoffs
                + distance_locations[IJsh] * naux_unique;
        inverse_step = 1.0 / distance_step;
        pair_cutoff = distance_cutoffs[IJsh];
        ei = unique_exponents[Ish];
        ej = unique_exponents[Jsh];
        shells[0] = ish;
        shells[1] = jsh;

        for (iimage = 0; iimage < nimages; iimage++) {
                shift_center(env_loc, env, Ls, iptr, iimage);
                ri = env_loc + iptr;
                for (jimage = 0; jimage < nimages; jimage++) {
                        shift_center(env_loc, env, Ls, jptr, jimage);
                        rj = env_loc + jptr;
                        dij2 = distance_square(ri, rj);
                        if (dij2 > pair_cutoff) {
                                continue;
                        }
                        distance_bin = (int)(sqrt(dij2) * inverse_step);
                        cutoff_aux = cutoff_base + distance_bin * naux_unique;
                        product_center(rc, ri, rj, ei, ej);
                        phase = conj(phase_i[iimage]) * phase_j[jimage];

                        for (ksh = aux0; ksh < aux1; ksh++) {
                                shells[2] = ksh;
                                dk = ao_loc[ksh + 1] - ao_loc[ksh];
                                dijk = dij * dk;
                                Ksh = auxuniq_map[ksh - 2 * nbas];
                                cutoff = cutoff_aux[Ksh];
                                kptr = atm[PTR_COORD
                                        + bas[ATOM_OF + ksh * BAS_SLOTS]
                                        * ATM_SLOTS];
                                rk = env_loc + kptr;
                                triple_distance = distance_square(rc, rk);
                                if (triple_distance >= cutoff) {
                                        continue;
                                }
                                if (!(*value_intor)(value_buffer, NULL, shells,
                                          atm, natm, bas, nbas, env_loc, NULL,
                                          cache)) {
                                        memset(value_buffer, 0,
                                               sizeof(double) * dijk);
                                }
                                if (!(*ip1_intor)(ip1_buffer, NULL, shells,
                                          atm, natm, bas, nbas, env_loc, NULL,
                                          cache)) {
                                        memset(ip1_buffer, 0,
                                               sizeof(double) * 3 * dijk);
                                }
                                if (!(*ip2_intor)(ip2_buffer, NULL, shells,
                                          atm, natm, bas, nbas, env_loc, NULL,
                                          cache)) {
                                        memset(ip2_buffer, 0,
                                               sizeof(double) * 3 * dijk);
                                }

                                for (k = 0; k < dk; k++) {
                                        const int kg = ao_loc[ksh] - ao_loc[aux0] + k;
                                        for (j = 0; j < dj; j++) {
                                                const int jg = ao_loc[jsh] - ao_loc[jsh0] + j;
                                                for (i = 0; i < di; i++) {
                                                        const int ig = ao_loc[ish] - ao_loc[ish0] + i;
                                                        q = k * dij + j * di + i;
                                                        output_index = ((size_t)ig * nj + jg) * naux + kg;
                                                        out[output_index] += phase * value_buffer[q];
                                                        for (x = 0; x < 3; x++) {
                                                                for (y = 0; y < 3; y++) {
                                                                        wi = rj[y] - ri[y];
                                                                        wk = rj[y] - rk[y];
                                                                        out[(1 + x * 3 + y) * component_size
                                                                            + output_index] += phase * (
                                                                                ip1_buffer[x * dijk + q] * wi
                                                                                + ip2_buffer[x * dijk + q] * wk);
                                                                }
                                                        }
                                                }
                                        }
                                }
                        }
                }
        }
}

static void kpair_strain_driver(
        int (*value_intor)(), int (*ip1_intor)(), int (*ip2_intor)(),
        double complex *out, int nimages, double *Ls,
        double complex *phase_i, double complex *phase_j, int *slice,
        int *ao_loc, int8_t *shell_pair_mask, int *refuniq_map,
        int *auxuniq_map, int naux_unique, double *unique_exponents,
        double *distance_cutoffs, double distance_step,
        double *three_center_cutoffs, int *distance_locations, int *atm,
        int natm, int *bas, int nbas, double *env, int nenv)
{
        const int nish = slice[1] - slice[0];
        const int njsh = slice[3] - slice[2];
        int cache_size = GTOmax_cache_size(
                value_intor, slice, 3, atm, natm, bas, nbas, env);
        int derivative_cache = GTOmax_cache_size(
                ip1_intor, slice, 3, atm, natm, bas, nbas, env);
        const int di = GTOmax_shell_dim(ao_loc, slice, 1);
        const int dj = GTOmax_shell_dim(ao_loc, slice + 2, 1);
        const int dk = GTOmax_shell_dim(ao_loc, slice + 4, 1);
        const size_t integral_size = (size_t)di * dj * dk;
        int pair;

        if (derivative_cache > cache_size) {
                cache_size = derivative_cache;
        }
        derivative_cache = GTOmax_cache_size(
                ip2_intor, slice, 3, atm, natm, bas, nbas, env);
        if (derivative_cache > cache_size) {
                cache_size = derivative_cache;
        }

#pragma omp parallel
{
        double *env_loc = malloc(sizeof(double) * nenv);
        double *work = malloc(sizeof(double) * (7 * integral_size + cache_size));
        memcpy(env_loc, env, sizeof(double) * nenv);
#pragma omp for schedule(dynamic)
        for (pair = 0; pair < nish * njsh; pair++) {
                const int ish = pair / njsh;
                const int jsh = pair % njsh;
                if (!shell_pair_mask[pair]) {
                        continue;
                }
                accumulate_kpair_strain(
                        value_intor, ip1_intor, ip2_intor, out, nimages, Ls,
                        phase_i, phase_j, ish, jsh, slice, ao_loc,
                        refuniq_map, auxuniq_map, naux_unique,
                        unique_exponents, distance_cutoffs, distance_step,
                        three_center_cutoffs, distance_locations, atm, natm,
                        bas, nbas, env, env_loc, work);
        }
        free(work);
        free(env_loc);
}
}

#define DECLARE_KPAIR_DRIVER(NAME, VALUE, IP1, IP2)                         \
void NAME(double complex *out, int nimages, double *Ls,                    \
          double complex *phase_i, double complex *phase_j, int *slice,    \
          int *ao_loc, int8_t *shell_pair_mask, int *refuniq_map,          \
          int *auxuniq_map, int naux_unique, double *unique_exponents,     \
          double *distance_cutoffs, double distance_step,                  \
          double *three_center_cutoffs, int *distance_locations,           \
          int *atm, int natm, int *bas, int nbas, double *env, int nenv)   \
{                                                                          \
        kpair_strain_driver(VALUE, IP1, IP2, out, nimages, Ls, phase_i,    \
                            phase_j, slice, ao_loc, shell_pair_mask,        \
                            refuniq_map, auxuniq_map, naux_unique,          \
                            unique_exponents, distance_cutoffs,             \
                            distance_step, three_center_cutoffs,            \
                            distance_locations, atm, natm, bas, nbas, env, \
                            nenv);                                          \
}

DECLARE_KPAIR_DRIVER(PBCsr3c_strain_kpair_sph, int3c2e_sph,
                     int3c2e_ip1_sph, int3c2e_ip2_sph)
DECLARE_KPAIR_DRIVER(PBCsr3c_strain_kpair_cart, int3c2e_cart,
                     int3c2e_ip1_cart, int3c2e_ip2_cart)

static struct PyModuleDef module_definition = {
        PyModuleDef_HEAD_INIT,
        "_sr3c_strain",
        "Image-resolved Gamma short-range three-center strain callback.",
        -1,
        NULL,
};

PyMODINIT_FUNC PyInit__sr3c_strain(void)
{
        return PyModule_Create(&module_definition);
}
