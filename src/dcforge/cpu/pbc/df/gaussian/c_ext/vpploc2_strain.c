/* Copyright 2021- The PySCF Developers. All Rights Reserved.
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
 * PySCF owns PBCnr3c1e_screened_nuc_grad_drv, its neighbor list, and the
 * libcint local-GTH integrals.  This callback retains the lattice image of
 * the differentiated first AO long enough to form a homogeneous-strain
 * contraction.  The result is contracted with all supplied real-Gamma
 * densities before leaving the shell/image loop.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <assert.h>
#include <stddef.h>
#include <string.h>
#include "cint.h"
#include "pbc/neighbor_list.h"


static void set_image_coordinate(double *local_env, const double *env,
                                 const double *lattice, int pointer,
                                 int image)
{
        local_env[pointer] = env[pointer] + lattice[3 * image];
        local_env[pointer + 1] =
                env[pointer + 1] + lattice[3 * image + 1];
        local_env[pointer + 2] =
                env[pointer + 2] + lattice[3 * image + 2];
}


/* For a symmetric Gamma density, exchange of the two primary AOs makes the
 * second-center contribution equal to the first-center contribution.  With
 * libcint's ip1 sign convention the exact fixed-fraction strain is
 *
 *   dE/d eps_xy = 2 sum D_ij I_ip1,x (R_K - R_i - L_i)_y .
 *
 * R_K is the central fake-GTH center and R_i+L_i is the actual AO image.
 * The ordinary PySCF nuclear-gradient callback sums I_ip1 before this image
 * is known to Python; this callback performs the same contraction one level
 * earlier and therefore retains the missing lattice moment.
 */
void DCFORGE_vpploc2_strain_fill_gs1(
        int (*intor)(), double *out, double *densities,
        int nkpts_ij, int nkpts, int components, int nimages,
        int ish_relative, int jsh_relative, double *buffer,
        double *local_env, double *lattice, double *unused_phase_r,
        double *unused_phase_i, int *unused_kptij_idx, int *slice,
        int *ao_loc, CINTOpt *optimizer, void *unused_pbcopt,
        int *atm, int natm, int *bas, int nbas, double *env, int nao,
        NeighborList **neighbor_list)
{
        (void)nkpts_ij;
        (void)nkpts;
        (void)unused_phase_r;
        (void)unused_phase_i;
        (void)unused_kptij_idx;
        (void)unused_pbcopt;
        assert(components > 0 && components % 9 == 0);

        const int nobs = components / 9;
        const int ish = ish_relative + slice[0];
        const int jsh = jsh_relative + slice[2];
        const int ksh0 = slice[4];
        const int ksh1 = slice[5];
        const int iptr = atm[PTR_COORD
                + bas[ATOM_OF + ish * BAS_SLOTS] * ATM_SLOTS];
        const int jptr = atm[PTR_COORD
                + bas[ATOM_OF + jsh * BAS_SLOTS] * ATM_SLOTS];
        const int di = ao_loc[ish + 1] - ao_loc[ish];
        const int dj = ao_loc[jsh + 1] - ao_loc[jsh];
        const int dij = di * dj;
        const int i0 = ao_loc[ish];
        const int j0 = ao_loc[jsh] - nao;
        const size_t density_stride = (size_t)nao * nao;
        const int jsh_offset = jsh - nbas;
        NeighborList *list = *neighbor_list;

        int shells[3] = {ish, jsh, ksh0};
        int ksh, image_i, image_j, index_i, index_j;
        for (ksh = ksh0; ksh < ksh1; ksh++) {
                shells[2] = ksh;
                const int ksh_offset = ksh - 2 * nbas;
                NeighborPair *pair_ki =
                        list->pairs[ksh_offset * nbas + ish];
                NeighborPair *pair_kj =
                        list->pairs[ksh_offset * nbas + jsh_offset];
                if (pair_ki->nimgs == 0 || pair_kj->nimgs == 0) {
                        continue;
                }

                const int dk = ao_loc[ksh + 1] - ao_loc[ksh];
                const int dijk = dij * dk;
                const int kptr = atm[PTR_COORD
                        + bas[ATOM_OF + ksh * BAS_SLOTS] * ATM_SLOTS];
                const double *rk = env + kptr;
                double *integrals = buffer;
                double *cache = buffer + 3 * dijk;

                for (index_i = 0; index_i < pair_ki->nimgs; index_i++) {
                        image_i = pair_ki->Ls_list[index_i];
                        set_image_coordinate(
                                local_env, env, lattice, iptr, image_i);
                        const double *ri = local_env + iptr;
                        for (index_j = 0; index_j < pair_kj->nimgs;
                             index_j++) {
                                image_j = pair_kj->Ls_list[index_j];
                                set_image_coordinate(
                                        local_env, env, lattice, jptr,
                                        image_j);
                                if (!(*intor)(
                                        integrals, NULL, shells, atm, natm,
                                        bas, nbas, local_env, optimizer,
                                        cache)) {
                                        continue;
                                }

                                int observable, axis_x, axis_y;
                                for (observable = 0; observable < nobs;
                                     observable++) {
                                        const double *density = densities
                                                + (size_t)observable
                                                  * density_stride;
                                        for (axis_x = 0; axis_x < 3;
                                             axis_x++) {
                                                const double *component =
                                                        integrals
                                                        + axis_x * dijk;
                                                double contraction = 0.0;
                                                int local_i, local_j,
                                                    auxiliary;
                                                for (auxiliary = 0;
                                                     auxiliary < dk;
                                                     auxiliary++) {
                                                        const double *block =
                                                                component
                                                                + auxiliary
                                                                  * dij;
                                                        for (local_i = 0;
                                                             local_i < di;
                                                             local_i++) {
                                                                const double
                                                                    *density_row =
                                                                    density
                                                                    + (size_t)(
                                                                        i0
                                                                        + local_i)
                                                                      * nao
                                                                    + j0;
                                                                for (local_j = 0;
                                                                     local_j < dj;
                                                                     local_j++) {
                                                                        contraction +=
                                                                            block[
                                                                                local_j
                                                                                * di
                                                                                + local_i]
                                                                            * density_row[
                                                                                local_j];
                                                                }
                                                        }
                                                }
                                                for (axis_y = 0; axis_y < 3;
                                                     axis_y++) {
                                                        out[observable * 9
                                                            + axis_x * 3
                                                            + axis_y] +=
                                                                2.0
                                                                * contraction
                                                                * (rk[axis_y]
                                                                   - ri[axis_y]);
                                                }
                                        }
                                }
                        }
                }
        }
}


static struct PyModuleDef module_definition = {
        PyModuleDef_HEAD_INIT,
        "_vpploc2_strain",
        "Image-resolved exact local-GTH part-2 strain contraction.",
        -1,
        NULL,
};


PyMODINIT_FUNC PyInit__vpploc2_strain(void)
{
        return PyModule_Create(&module_definition);
}
