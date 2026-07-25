/* Exact PySCF RSDF short-range two-center metric strain at Gamma. */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <omp.h>
#include "config.h"
#include "cint.h"

#define MAX2(a, b) ((a) > (b) ? (a) : (b))

int GTOmax_shell_dim(int *ao_loc, int *shls_slice, int ncenter);
int GTOmax_cache_size(int (*intor)(), int *shls_slice, int ncenter,
                      int *atm, int natm, int *bas, int nbas, double *env);

static double distance_square(const double *first, const double *second)
{
        const double dx = first[0] - second[0];
        const double dy = first[1] - second[1];
        const double dz = first[2] - second[2];
        return dx * dx + dy * dy + dz * dz;
}

static void shift_center(double *local_env, const double *env,
                         const double *lattice, int pointer, int image)
{
        local_env[pointer + 0] = env[pointer + 0] + lattice[image * 3 + 0];
        local_env[pointer + 1] = env[pointer + 1] + lattice[image * 3 + 1];
        local_env[pointer + 2] = env[pointer + 2] + lattice[image * 3 + 2];
}

void PBCsr2c_strain_drv(int (*value_intor)(), int (*ip1_intor)(), double *out,
                        int nimages, double *lattice, int *slice, int *ao_loc,
                        const int *reference_map, const double *cutoff2,
                        int *atm, int natm, int *bas, int nbas,
                        double *env, int nenv)
{
        const int ish0 = slice[0];
        const int ish1 = slice[1];
        const int jsh0 = slice[2];
        const int jsh1 = slice[3];
        const int ni = ao_loc[ish1] - ao_loc[ish0];
        const int nj = ao_loc[jsh1] - ao_loc[jsh0];
        const size_t matrix_size = (size_t)ni * nj;
        int value_cache = GTOmax_cache_size(
                value_intor, slice, 2, atm, natm, bas, nbas, env);
        int ip_cache = GTOmax_cache_size(
                ip1_intor, slice, 2, atm, natm, bas, nbas, env);
        const int cache_size = MAX2(value_cache, ip_cache);
        const int maximum_ao = GTOmax_shell_dim(ao_loc, slice, 2);

#pragma omp parallel
        {
                double *local_env = malloc(sizeof(double) * nenv);
                double *buffer = malloc(
                        sizeof(double) * (4 * maximum_ao * maximum_ao + cache_size));
                double *value = buffer;
                double *ip1 = value + maximum_ao * maximum_ao;
                double *cache = ip1 + 3 * maximum_ao * maximum_ao;
                int jsh;

                memcpy(local_env, env, sizeof(double) * nenv);
#pragma omp for schedule(dynamic)
                for (jsh = jsh0; jsh < jsh1; jsh++) {
                        const int jpointer = atm[
                                PTR_COORD + bas[ATOM_OF + jsh * BAS_SLOTS] * ATM_SLOTS];
                        const int dj = ao_loc[jsh + 1] - ao_loc[jsh];
                        const int JSH = reference_map[jsh - jsh0];
                        int image, ish;
                        for (image = 0; image < nimages; image++) {
                                shift_center(local_env, env, lattice, jpointer, image);
                                const double *rj = local_env + jpointer;
                                for (ish = ish0; ish < ish1; ish++) {
                                        const int ipointer = atm[
                                                PTR_COORD + bas[ATOM_OF + ish * BAS_SLOTS]
                                                * ATM_SLOTS];
                                        const double *ri = local_env + ipointer;
                                        const int ISH = reference_map[ish - ish0];
                                        const int pair = ISH >= JSH
                                                ? ISH * (ISH + 1) / 2 + JSH
                                                : JSH * (JSH + 1) / 2 + ISH;
                                        if (distance_square(ri, rj) >= cutoff2[pair]) {
                                                continue;
                                        }
                                        const int di = ao_loc[ish + 1] - ao_loc[ish];
                                        const int dij = di * dj;
                                        int shells[2] = {ish, jsh};
                                        int present = value_intor(
                                                value, NULL, shells, atm, natm, bas,
                                                nbas, local_env, NULL, cache);
                                        if (!present) {
                                                memset(value, 0, sizeof(double) * dij);
                                        }
                                        present = ip1_intor(
                                                ip1, NULL, shells, atm, natm, bas,
                                                nbas, local_env, NULL, cache);
                                        if (!present) {
                                                memset(ip1, 0, sizeof(double) * 3 * dij);
                                        }
                                        int i, j, x, y;
                                        for (j = 0; j < dj; j++) {
                                                const int column = ao_loc[jsh] - ao_loc[jsh0] + j;
                                                for (i = 0; i < di; i++) {
                                                        const int row = ao_loc[ish] - ao_loc[ish0] + i;
                                                        const int q = j * di + i;
                                                        const size_t output = (size_t)row * nj + column;
                                                        out[output] += value[q];
                                                        for (x = 0; x < 3; x++) {
                                                                for (y = 0; y < 3; y++) {
                                                                        out[(1 + x * 3 + y) * matrix_size + output]
                                                                                += ip1[x * dij + q]
                                                                                * (rj[y] - ri[y]);
                                                                }
                                                        }
                                                }
                                        }
                                }
                        }
                }
                free(buffer);
                free(local_env);
        }
}

void PBCsr2c_strain_kpt_drv(
        int (*value_intor)(), int (*ip1_intor)(), double complex *out,
        int nimages, double *lattice, double complex *phase, int *slice,
        int *ao_loc, const int *reference_map, const double *cutoff2,
        int *atm, int natm, int *bas, int nbas, double *env, int nenv)
{
        const int ish0 = slice[0];
        const int ish1 = slice[1];
        const int jsh0 = slice[2];
        const int jsh1 = slice[3];
        const int ni = ao_loc[ish1] - ao_loc[ish0];
        const int nj = ao_loc[jsh1] - ao_loc[jsh0];
        const size_t matrix_size = (size_t)ni * nj;
        int value_cache = GTOmax_cache_size(
                value_intor, slice, 2, atm, natm, bas, nbas, env);
        int ip_cache = GTOmax_cache_size(
                ip1_intor, slice, 2, atm, natm, bas, nbas, env);
        const int cache_size = MAX2(value_cache, ip_cache);
        const int maximum_ao = GTOmax_shell_dim(ao_loc, slice, 2);

#pragma omp parallel
        {
                double *local_env = malloc(sizeof(double) * nenv);
                double *buffer = malloc(
                        sizeof(double) * (4 * maximum_ao * maximum_ao + cache_size));
                double *value = buffer;
                double *ip1 = value + maximum_ao * maximum_ao;
                double *cache = ip1 + 3 * maximum_ao * maximum_ao;
                int jsh;

                memcpy(local_env, env, sizeof(double) * nenv);
#pragma omp for schedule(dynamic)
                for (jsh = jsh0; jsh < jsh1; jsh++) {
                        const int jpointer = atm[
                                PTR_COORD + bas[ATOM_OF + jsh * BAS_SLOTS] * ATM_SLOTS];
                        const int dj = ao_loc[jsh + 1] - ao_loc[jsh];
                        const int JSH = reference_map[jsh - jsh0];
                        int image, ish;
                        for (image = 0; image < nimages; image++) {
                                shift_center(local_env, env, lattice, jpointer, image);
                                const double *rj = local_env + jpointer;
                                const double complex image_phase = phase[image];
                                for (ish = ish0; ish < ish1; ish++) {
                                        const int ipointer = atm[
                                                PTR_COORD + bas[ATOM_OF + ish * BAS_SLOTS]
                                                * ATM_SLOTS];
                                        const double *ri = local_env + ipointer;
                                        const int ISH = reference_map[ish - ish0];
                                        const int pair = ISH >= JSH
                                                ? ISH * (ISH + 1) / 2 + JSH
                                                : JSH * (JSH + 1) / 2 + ISH;
                                        if (distance_square(ri, rj) >= cutoff2[pair]) {
                                                continue;
                                        }
                                        const int di = ao_loc[ish + 1] - ao_loc[ish];
                                        const int dij = di * dj;
                                        int shells[2] = {ish, jsh};
                                        int present = value_intor(
                                                value, NULL, shells, atm, natm, bas,
                                                nbas, local_env, NULL, cache);
                                        if (!present) {
                                                memset(value, 0, sizeof(double) * dij);
                                        }
                                        present = ip1_intor(
                                                ip1, NULL, shells, atm, natm, bas,
                                                nbas, local_env, NULL, cache);
                                        if (!present) {
                                                memset(ip1, 0, sizeof(double) * 3 * dij);
                                        }
                                        int i, j, x, y;
                                        for (j = 0; j < dj; j++) {
                                                const int column = ao_loc[jsh]
                                                        - ao_loc[jsh0] + j;
                                                for (i = 0; i < di; i++) {
                                                        const int row = ao_loc[ish]
                                                                - ao_loc[ish0] + i;
                                                        const int q = j * di + i;
                                                        const size_t output =
                                                                (size_t)row * nj + column;
                                                        out[output] += image_phase * value[q];
                                                        for (x = 0; x < 3; x++) {
                                                                for (y = 0; y < 3; y++) {
                                                                        out[(1 + x * 3 + y)
                                                                            * matrix_size + output]
                                                                                += image_phase
                                                                                * ip1[x * dij + q]
                                                                                * (rj[y] - ri[y]);
                                                                }
                                                        }
                                                }
                                        }
                                }
                        }
                }
                free(buffer);
                free(local_env);
        }
}
