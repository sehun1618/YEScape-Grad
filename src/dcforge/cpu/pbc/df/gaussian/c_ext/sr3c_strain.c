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
 * PySCF owns the screened PBCsr3c_g_drv lattice traversal.  This callback
 * adds only the missing homogeneous-strain contraction for one surviving
 * Gamma shell pair.  It intentionally favors a compact, auditable reference
 * implementation over production workspace reuse.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "cint.h"

int GTOmax_shell_dim(int *ao_loc, int *shls_slice, int ncenter);
int GTOmax_cache_size(int (*intor)(), int *shls_slice, int ncenter,
                      int *atm, int natm, int *bas, int nbas, double *env);
int int3c2e_ip1_sph();
int int3c2e_ip1_cart();
int int3c2e_ip2_sph();
int int3c2e_ip2_cart();

static double squared_distance(const double *a, const double *b)
{
        const double x = a[0] - b[0];
        const double y = a[1] - b[1];
        const double z = a[2] - b[2];
        return x * x + y * y + z * z;
}

static void gaussian_product_center(double *center, const double *a,
                                    const double *b, double ea, double eb)
{
        const double inverse = 1.0 / (ea + eb);
        center[0] = (ea * a[0] + eb * b[0]) * inverse;
        center[1] = (ea * a[1] + eb * b[1]) * inverse;
        center[2] = (ea * a[2] + eb * b[2]) * inverse;
}

static void set_image_coordinate(double *local_env, const double *env,
                                 const double *lattice, int pointer,
                                 int image)
{
        local_env[pointer] = env[pointer] + lattice[3 * image];
        local_env[pointer + 1] = env[pointer + 1] + lattice[3 * image + 1];
        local_env[pointer + 2] = env[pointer + 2] + lattice[3 * image + 2];
}

static void accumulate_shell_pair(
        int (*ip1_intor)(), int (*ip2_intor)(), double *out,
        int nimages, const double *lattice, int ish_relative,
        int jsh_relative, double *local_env, int *slice, int *ao_loc,
        CINTOpt *optimizer, int *reference_unique_map,
        int *auxiliary_unique_map, int naux_unique,
        double *unique_exponents, double *distance_cutoffs,
        double distance_step, double *three_center_cutoffs,
        int *distance_locations, int *atm, int natm, int *bas, int nbas,
        double *env)
{
        const int ish = ish_relative + slice[0];
        const int jsh = jsh_relative + slice[2];
        const int aux0 = slice[4];
        const int aux1 = slice[5];
        const int iptr = atm[PTR_COORD
                + bas[ATOM_OF + ish * BAS_SLOTS] * ATM_SLOTS];
        const int jptr = atm[PTR_COORD
                + bas[ATOM_OF + jsh * BAS_SLOTS] * ATM_SLOTS];
        const int di = ao_loc[ish + 1] - ao_loc[ish];
        const int dj = ao_loc[jsh + 1] - ao_loc[jsh];
        const int dij = di * dj;
        const int naux = ao_loc[aux1] - ao_loc[aux0];
        const int max_dk = GTOmax_shell_dim(ao_loc, slice + 4, 1);
        const size_t max_integrals = (size_t)dij * max_dk;
        int cache_size = GTOmax_cache_size(
                ip1_intor, slice, 3, atm, natm, bas, nbas, env);
        const int ip2_cache = GTOmax_cache_size(
                ip2_intor, slice, 3, atm, natm, bas, nbas, env);
        if (ip2_cache > cache_size) {
                cache_size = ip2_cache;
        }

        double *work = malloc(sizeof(double) * (6 * max_integrals + cache_size));
        double *ip1 = work;
        double *ip2 = ip1 + 3 * max_integrals;
        double *cache = ip2 + 3 * max_integrals;
        double *response = calloc((size_t)9 * dij * naux, sizeof(double));
        assert(work != NULL && response != NULL);

        const int unique_i = reference_unique_map[ish];
        const int unique_j = reference_unique_map[jsh - nbas];
        const int pair_unique = unique_i >= unique_j
                ? unique_i * (unique_i + 1) / 2 + unique_j
                : unique_j * (unique_j + 1) / 2 + unique_i;
        const double *cutoff_origin = three_center_cutoffs
                + distance_locations[pair_unique] * naux_unique;
        const double pair_cutoff = distance_cutoffs[pair_unique];
        const double inverse_step = 1.0 / distance_step;
        const double exponent_i = unique_exponents[unique_i];
        const double exponent_j = unique_exponents[unique_j];
        int shells[3] = {ish, jsh, aux0};
        int image_i, image_j, ksh, axis_x, axis_y, q;

        for (image_i = 0; image_i < nimages; image_i++) {
                set_image_coordinate(
                        local_env, env, lattice, iptr, image_i);
                const double *ri = local_env + iptr;
                for (image_j = 0; image_j < nimages; image_j++) {
                        set_image_coordinate(
                                local_env, env, lattice, jptr, image_j);
                        const double *rj = local_env + jptr;
                        const double pair_distance = squared_distance(ri, rj);
                        if (pair_distance > pair_cutoff) {
                                continue;
                        }
                        const int distance_bin =
                                (int)(sqrt(pair_distance) * inverse_step);
                        const double *aux_cutoffs = cutoff_origin
                                + distance_bin * naux_unique;
                        double product[3];
                        gaussian_product_center(
                                product, ri, rj, exponent_i, exponent_j);

                        for (ksh = aux0; ksh < aux1; ksh++) {
                                const int K = auxiliary_unique_map[
                                        ksh - 2 * nbas];
                                const int kptr = atm[PTR_COORD
                                        + bas[ATOM_OF + ksh * BAS_SLOTS]
                                        * ATM_SLOTS];
                                const double *rk = local_env + kptr;
                                if (squared_distance(product, rk)
                                        >= aux_cutoffs[K]) {
                                        continue;
                                }
                                shells[2] = ksh;
                                const int dk = ao_loc[ksh + 1] - ao_loc[ksh];
                                const int dijk = dij * dk;
                                if (!(*ip1_intor)(
                                        ip1, NULL, shells, atm, natm, bas,
                                        nbas, local_env, optimizer, cache)) {
                                        memset(ip1, 0,
                                               sizeof(double) * 3 * dijk);
                                }
                                if (!(*ip2_intor)(
                                        ip2, NULL, shells, atm, natm, bas,
                                        nbas, local_env, NULL, cache)) {
                                        memset(ip2, 0,
                                               sizeof(double) * 3 * dijk);
                                }
                                const int aux_offset =
                                        ao_loc[ksh] - ao_loc[aux0];
                                for (axis_x = 0; axis_x < 3; axis_x++) {
                                        for (axis_y = 0; axis_y < 3; axis_y++) {
                                                const double weight_i =
                                                        rj[axis_y] - ri[axis_y];
                                                const double weight_aux =
                                                        rj[axis_y] - rk[axis_y];
                                                double *target = response
                                                        + ((axis_x * 3 + axis_y)
                                                           * dij) * naux;
                                                for (q = 0; q < dijk; q++) {
                                                        const int pair_ao = q % dij;
                                                        const int aux_ao = q / dij;
                                                        target[pair_ao * naux
                                                               + aux_offset + aux_ao]
                                                                += ip1[axis_x * dijk + q]
                                                                   * weight_i
                                                                 + ip2[axis_x * dijk + q]
                                                                   * weight_aux;
                                                }
                                        }
                                }
                        }
                }
        }

        const int nao = ao_loc[slice[1]] - ao_loc[slice[0]];
        const size_t npair = (size_t)nao * (nao + 1) / 2;
        const size_t component_size = npair * naux;
        int local_i, local_j, auxiliary, component;
        for (local_i = 0; local_i < di; local_i++) {
                const int global_i =
                        ao_loc[ish] - ao_loc[slice[0]] + local_i;
                for (local_j = 0; local_j < dj; local_j++) {
                        const int global_j =
                                ao_loc[jsh] - ao_loc[slice[2]] + local_j;
                        if (global_i < global_j) {
                                continue;
                        }
                        const int pair_ao = local_j * di + local_i;
                        const size_t packed =
                                (size_t)global_i * (global_i + 1) / 2
                                + global_j;
                        for (component = 0; component < 9; component++) {
                                const double *source = response
                                        + ((size_t)component * dij + pair_ao)
                                          * naux;
                                double *target = out
                                        + (size_t)component * component_size
                                        + packed * naux;
                                for (auxiliary = 0; auxiliary < naux; auxiliary++) {
                                        target[auxiliary] = source[auxiliary];
                                }
                        }
                }
        }
        free(response);
        free(work);
}

static void fill_gamma_strain(
        int (*ip1_intor)(), int (*ip2_intor)(), double *out, int components,
        int nimages, int ish, int jsh, double *unused_buffer,
        double *local_env, double *lattice, int *slice, int *ao_loc,
        CINTOpt *optimizer, int *reference_unique_map,
        int *auxiliary_unique_map, int naux_unique,
        double *unique_exponents, double *distance_cutoffs,
        double distance_step, double *three_center_cutoffs,
        int *distance_locations, int *atm, int natm, int *bas, int nbas,
        double *env)
{
        (void)unused_buffer;
        assert(components == 9);
        const int absolute_i = ish + slice[0];
        const int absolute_j = jsh + slice[2] - nbas;
        if (absolute_i < absolute_j) {
                return;
        }
        accumulate_shell_pair(
                ip1_intor, ip2_intor, out, nimages, lattice, ish, jsh,
                local_env, slice, ao_loc, optimizer, reference_unique_map,
                auxiliary_unique_map, naux_unique, unique_exponents,
                distance_cutoffs, distance_step, three_center_cutoffs,
                distance_locations, atm, natm, bas, nbas, env);
}

/* Contract the image-resolved strain integral with one or more source
 * adjoints while the shell/image indices are still available.
 *
 * The ``out`` workspace has two consecutive regions.  The leading region is
 * ``(nobs,9,npair)`` and receives one scalar for every observable, raw strain
 * component and lower-triangular AO pair.  The trailing read-only region is
 * ``(nobs,npair,naux)`` and stores the symmetrized source adjoints prepared by
 * Python.  Keeping the AO-pair index in the leading result prevents OpenMP
 * callbacks for distinct shell pairs from accumulating into the same scalar.
 * Python sums that small final axis after the driver returns.
 */
static void accumulate_shell_pair_contracted(
        int (*ip1_intor)(), int (*ip2_intor)(), double *out, int nobs,
        int nimages, const double *lattice, int ish_relative,
        int jsh_relative, double *local_env, int *slice, int *ao_loc,
        CINTOpt *optimizer, int *reference_unique_map,
        int *auxiliary_unique_map, int naux_unique,
        double *unique_exponents, double *distance_cutoffs,
        double distance_step, double *three_center_cutoffs,
        int *distance_locations, int *atm, int natm, int *bas, int nbas,
        double *env)
{
        const int ish = ish_relative + slice[0];
        const int jsh = jsh_relative + slice[2];
        const int aux0 = slice[4];
        const int aux1 = slice[5];
        const int iptr = atm[PTR_COORD
                + bas[ATOM_OF + ish * BAS_SLOTS] * ATM_SLOTS];
        const int jptr = atm[PTR_COORD
                + bas[ATOM_OF + jsh * BAS_SLOTS] * ATM_SLOTS];
        const int di = ao_loc[ish + 1] - ao_loc[ish];
        const int dj = ao_loc[jsh + 1] - ao_loc[jsh];
        const int dij = di * dj;
        const int nao = ao_loc[slice[1]] - ao_loc[slice[0]];
        const int naux = ao_loc[aux1] - ao_loc[aux0];
        const size_t npair = (size_t)nao * (nao + 1) / 2;
        const size_t result_size = (size_t)nobs * 9 * npair;
        const double *adjoints = out + result_size;
        const int max_dk = GTOmax_shell_dim(ao_loc, slice + 4, 1);
        const size_t max_integrals = (size_t)dij * max_dk;
        int cache_size = GTOmax_cache_size(
                ip1_intor, slice, 3, atm, natm, bas, nbas, env);
        const int ip2_cache = GTOmax_cache_size(
                ip2_intor, slice, 3, atm, natm, bas, nbas, env);
        if (ip2_cache > cache_size) {
                cache_size = ip2_cache;
        }

        double *work = malloc(sizeof(double) * (6 * max_integrals + cache_size));
        double *ip1 = work;
        double *ip2 = ip1 + 3 * max_integrals;
        double *cache = ip2 + 3 * max_integrals;
        assert(work != NULL);

        const int unique_i = reference_unique_map[ish];
        const int unique_j = reference_unique_map[jsh - nbas];
        const int pair_unique = unique_i >= unique_j
                ? unique_i * (unique_i + 1) / 2 + unique_j
                : unique_j * (unique_j + 1) / 2 + unique_i;
        const double *cutoff_origin = three_center_cutoffs
                + distance_locations[pair_unique] * naux_unique;
        const double pair_cutoff = distance_cutoffs[pair_unique];
        const double inverse_step = 1.0 / distance_step;
        const double exponent_i = unique_exponents[unique_i];
        const double exponent_j = unique_exponents[unique_j];
        int shells[3] = {ish, jsh, aux0};
        int image_i, image_j, ksh, axis_x, axis_y, q, observable;

        for (image_i = 0; image_i < nimages; image_i++) {
                set_image_coordinate(
                        local_env, env, lattice, iptr, image_i);
                const double *ri = local_env + iptr;
                for (image_j = 0; image_j < nimages; image_j++) {
                        set_image_coordinate(
                                local_env, env, lattice, jptr, image_j);
                        const double *rj = local_env + jptr;
                        const double pair_distance = squared_distance(ri, rj);
                        if (pair_distance > pair_cutoff) {
                                continue;
                        }
                        const int distance_bin =
                                (int)(sqrt(pair_distance) * inverse_step);
                        const double *aux_cutoffs = cutoff_origin
                                + distance_bin * naux_unique;
                        double product[3];
                        gaussian_product_center(
                                product, ri, rj, exponent_i, exponent_j);

                        for (ksh = aux0; ksh < aux1; ksh++) {
                                const int K = auxiliary_unique_map[
                                        ksh - 2 * nbas];
                                const int kptr = atm[PTR_COORD
                                        + bas[ATOM_OF + ksh * BAS_SLOTS]
                                        * ATM_SLOTS];
                                const double *rk = local_env + kptr;
                                if (squared_distance(product, rk)
                                        >= aux_cutoffs[K]) {
                                        continue;
                                }
                                shells[2] = ksh;
                                const int dk = ao_loc[ksh + 1] - ao_loc[ksh];
                                const int dijk = dij * dk;
                                if (!(*ip1_intor)(
                                        ip1, NULL, shells, atm, natm, bas,
                                        nbas, local_env, optimizer, cache)) {
                                        memset(ip1, 0,
                                               sizeof(double) * 3 * dijk);
                                }
                                if (!(*ip2_intor)(
                                        ip2, NULL, shells, atm, natm, bas,
                                        nbas, local_env, NULL, cache)) {
                                        memset(ip2, 0,
                                               sizeof(double) * 3 * dijk);
                                }
                                const int aux_offset =
                                        ao_loc[ksh] - ao_loc[aux0];
                                for (q = 0; q < dijk; q++) {
                                        const int pair_ao = q % dij;
                                        const int local_i = pair_ao % di;
                                        const int local_j = pair_ao / di;
                                        const int global_i =
                                                ao_loc[ish] - ao_loc[slice[0]]
                                                + local_i;
                                        const int global_j =
                                                ao_loc[jsh] - ao_loc[slice[2]]
                                                + local_j;
                                        if (global_i < global_j) {
                                                continue;
                                        }
                                        const int aux_ao = q / dij;
                                        const size_t packed =
                                                (size_t)global_i
                                                * (global_i + 1) / 2
                                                + global_j;
                                        const size_t auxiliary =
                                                (size_t)aux_offset + aux_ao;
                                        for (axis_x = 0; axis_x < 3; axis_x++) {
                                                for (axis_y = 0; axis_y < 3;
                                                     axis_y++) {
                                                        const int component =
                                                                axis_x * 3
                                                                + axis_y;
                                                        const double response =
                                                                ip1[axis_x * dijk + q]
                                                                * (rj[axis_y]
                                                                   - ri[axis_y])
                                                              + ip2[axis_x * dijk + q]
                                                                * (rj[axis_y]
                                                                   - rk[axis_y]);
                                                        for (observable = 0;
                                                             observable < nobs;
                                                             observable++) {
                                                                const double adjoint =
                                                                        adjoints[
                                                                            ((size_t)observable
                                                                             * npair
                                                                             + packed)
                                                                            * naux
                                                                            + auxiliary];
                                                                out[((size_t)observable
                                                                     * 9 + component)
                                                                    * npair + packed]
                                                                        += response
                                                                           * adjoint;
                                                        }
                                                }
                                        }
                                }
                        }
                }
        }
        free(work);
}

static void fill_gamma_strain_contracted(
        int (*ip1_intor)(), int (*ip2_intor)(), double *out, int nobs,
        int nimages, int ish, int jsh, double *unused_buffer,
        double *local_env, double *lattice, int *slice, int *ao_loc,
        CINTOpt *optimizer, int *reference_unique_map,
        int *auxiliary_unique_map, int naux_unique,
        double *unique_exponents, double *distance_cutoffs,
        double distance_step, double *three_center_cutoffs,
        int *distance_locations, int *atm, int natm, int *bas, int nbas,
        double *env)
{
        (void)unused_buffer;
        assert(nobs > 0);
        const int absolute_i = ish + slice[0];
        const int absolute_j = jsh + slice[2] - nbas;
        if (absolute_i < absolute_j) {
                return;
        }
        accumulate_shell_pair_contracted(
                ip1_intor, ip2_intor, out, nobs, nimages, lattice, ish, jsh,
                local_env, slice, ao_loc, optimizer, reference_unique_map,
                auxiliary_unique_map, naux_unique, unique_exponents,
                distance_cutoffs, distance_step, three_center_cutoffs,
                distance_locations, atm, natm, bas, nbas, env);
}

#define DECLARE_GAMMA_FILL(NAME, IP2)                                      \
void NAME(int (*ip1)(), double *out, int components, int nimages,          \
          int ish, int jsh, double *buffer, double *local_env,             \
          double *lattice, int *slice, int *ao_loc, CINTOpt *optimizer,    \
          int *reference_unique_map, int *auxiliary_unique_map,            \
          int naux_unique, double *unique_exponents,                       \
          double *distance_cutoffs, double distance_step,                  \
          double *three_center_cutoffs, int *distance_locations,           \
          int *atm, int natm, int *bas, int nbas, double *env)             \
{                                                                          \
        fill_gamma_strain(                                                  \
                ip1, IP2, out, components, nimages, ish, jsh, buffer,      \
                local_env, lattice, slice, ao_loc, optimizer,              \
                reference_unique_map, auxiliary_unique_map, naux_unique,   \
                unique_exponents, distance_cutoffs, distance_step,         \
                three_center_cutoffs, distance_locations, atm, natm, bas,  \
                nbas, env);                                                 \
}

DECLARE_GAMMA_FILL(DCFORGE_sr3c_strain_sph, int3c2e_ip2_sph)
DECLARE_GAMMA_FILL(DCFORGE_sr3c_strain_cart, int3c2e_ip2_cart)

#define DECLARE_GAMMA_CONTRACT(NAME, IP2)                                  \
void NAME(int (*ip1)(), double *out, int nobs, int nimages,                \
          int ish, int jsh, double *buffer, double *local_env,             \
          double *lattice, int *slice, int *ao_loc, CINTOpt *optimizer,    \
          int *reference_unique_map, int *auxiliary_unique_map,            \
          int naux_unique, double *unique_exponents,                       \
          double *distance_cutoffs, double distance_step,                  \
          double *three_center_cutoffs, int *distance_locations,           \
          int *atm, int natm, int *bas, int nbas, double *env)             \
{                                                                          \
        fill_gamma_strain_contracted(                                      \
                ip1, IP2, out, nobs, nimages, ish, jsh, buffer,           \
                local_env, lattice, slice, ao_loc, optimizer,              \
                reference_unique_map, auxiliary_unique_map, naux_unique,   \
                unique_exponents, distance_cutoffs, distance_step,         \
                three_center_cutoffs, distance_locations, atm, natm, bas,  \
                nbas, env);                                                 \
}

DECLARE_GAMMA_CONTRACT(
        DCFORGE_sr3c_strain_contract_sph, int3c2e_ip2_sph)
DECLARE_GAMMA_CONTRACT(
        DCFORGE_sr3c_strain_contract_cart, int3c2e_ip2_cart)

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
