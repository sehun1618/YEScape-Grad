/* Direct image-resolved two-center metric-strain contraction.
 *
 * This DC-Forge callback implements the homogeneous-strain identity
 *
 *   A_PQ M_PQ^(xy)
 *     = sum_L A_PQ (int2c2e_ip1)_x,PQ
 *                       (R_Q + L - R_P)_y
 *
 * for real-Gamma symmetric metric adjoints.  PySCF/libcint continues to own
 * the screened short-range Gaussian integral.  The callback consumes each
 * shell/image block immediately and returns only nobs x 3 x 3 scalars; it
 * never constructs a full strain derivative metric.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "cint.h"

int GTOmax_shell_dim(int *ao_loc, int *shls_slice, int ncenter);
int GTOmax_cache_size(int (*intor)(), int *shls_slice, int ncenter,
                      int *atm, int natm, int *bas, int nbas, double *env);

static double distance_squared(const double *first, const double *second)
{
        const double dx = first[0] - second[0];
        const double dy = first[1] - second[1];
        const double dz = first[2] - second[2];
        return dx * dx + dy * dy + dz * dz;
}

static void set_image_coordinate(double *local_env, const double *env,
                                 const double *lattice, int pointer,
                                 int image)
{
        local_env[pointer] = env[pointer] + lattice[3 * image];
        local_env[pointer + 1] = env[pointer + 1] + lattice[3 * image + 1];
        local_env[pointer + 2] = env[pointer + 2] + lattice[3 * image + 2];
}

void DCFORGE_sr2c_adjoint_strain(
        int (*ip1_intor)(), double *out, const double *adjoint, int nobs,
        int nimages, const double *lattice, int *slice, int *ao_loc,
        const int *reference_map, const double *cutoff_squared,
        int *atm, int natm, int *bas, int nbas, double *env, int nenv)
{
        const int ish0 = slice[0];
        const int ish1 = slice[1];
        const int jsh0 = slice[2];
        const int jsh1 = slice[3];
        const int naux = ao_loc[ish1] - ao_loc[ish0];
        const int cache_size = GTOmax_cache_size(
                ip1_intor, slice, 2, atm, natm, bas, nbas, env);
        const int maximum_ao = GTOmax_shell_dim(
                ao_loc, slice, 2);

#pragma omp parallel
        {
                double *local_env = malloc(sizeof(double) * (size_t)nenv);
                double *ip1 = malloc(sizeof(double)
                                     * (size_t)3 * maximum_ao * maximum_ao);
                double *cache = malloc(sizeof(double) * (size_t)cache_size);
                double *local_out = calloc(
                        (size_t)nobs * 9, sizeof(double));
                int jsh;

                memcpy(local_env, env, sizeof(double) * (size_t)nenv);
#pragma omp for schedule(dynamic)
                for (jsh = jsh0; jsh < jsh1; jsh++) {
                        const int jatom = bas[ATOM_OF + jsh * BAS_SLOTS];
                        const int jpointer = atm[
                                PTR_COORD + jatom * ATM_SLOTS];
                        const int dj = ao_loc[jsh + 1] - ao_loc[jsh];
                        const int jreference = reference_map[jsh - jsh0];
                        int image;

                        for (image = 0; image < nimages; image++) {
                                int ish;
                                set_image_coordinate(
                                        local_env, env, lattice, jpointer,
                                        image);
                                for (ish = ish0; ish < ish1; ish++) {
                                        const int iatom = bas[
                                                ATOM_OF + ish * BAS_SLOTS];
                                        const int ipointer = atm[
                                                PTR_COORD + iatom * ATM_SLOTS];
                                        const int ireference = reference_map[
                                                ish - ish0];
                                        const int high = ireference > jreference
                                                ? ireference : jreference;
                                        const int low = ireference > jreference
                                                ? jreference : ireference;
                                        const int pair = high * (high + 1) / 2
                                                + low;
                                        const double *ri = local_env + ipointer;
                                        const double *rj = local_env + jpointer;
                                        int shells[2] = {ish, jsh};
                                        int present;
                                        int di;
                                        int dij;
                                        int i;
                                        int j;

                                        if (distance_squared(ri, rj)
                                                >= cutoff_squared[pair]) {
                                                continue;
                                        }
                                        di = ao_loc[ish + 1] - ao_loc[ish];
                                        dij = di * dj;
                                        present = ip1_intor(
                                                ip1, NULL, shells, atm, natm,
                                                bas, nbas, local_env, NULL,
                                                cache);
                                        if (!present) {
                                                continue;
                                        }
                                        for (j = 0; j < dj; j++) {
                                                const int column =
                                                        ao_loc[jsh] - ao_loc[jsh0]
                                                        + j;
                                                for (i = 0; i < di; i++) {
                                                        const int row =
                                                                ao_loc[ish]
                                                                - ao_loc[ish0]
                                                                + i;
                                                        const int q = j * di + i;
                                                        int observable;
                                                        int axis_x;

                                                        for (observable = 0;
                                                             observable < nobs;
                                                             observable++) {
                                                                const double weight =
                                                                        adjoint[
                                                                                ((size_t)observable
                                                                                 * naux + row)
                                                                                * naux + column];
                                                                if (weight == 0.0) {
                                                                        continue;
                                                                }
                                                                for (axis_x = 0;
                                                                     axis_x < 3;
                                                                     axis_x++) {
                                                                        const double derivative =
                                                                                ip1[axis_x
                                                                                    * dij + q];
                                                                        int axis_y;
                                                                        for (axis_y = 0;
                                                                             axis_y < 3;
                                                                             axis_y++) {
                                                                                local_out[
                                                                                        observable * 9
                                                                                        + axis_x * 3
                                                                                        + axis_y] +=
                                                                                                weight
                                                                                                * derivative
                                                                                                * (rj[axis_y]
                                                                                                   - ri[axis_y]);
                                                                        }
                                                                }
                                                        }
                                                }
                                        }
                                }
                        }
                }

#pragma omp critical
                {
                        size_t index;
                        for (index = 0; index < (size_t)nobs * 9; index++) {
                                out[index] += local_out[index];
                        }
                }
                free(local_out);
                free(cache);
                free(ip1);
                free(local_env);
        }
}

static struct PyModuleDef module_definition = {
        PyModuleDef_HEAD_INIT,
        "_sr2c_adjoint_strain",
        "Direct image-resolved two-center metric-strain contraction.",
        -1,
        NULL,
};

PyMODINIT_FUNC PyInit__sr2c_adjoint_strain(void)
{
        return PyModule_Create(&module_definition);
}
