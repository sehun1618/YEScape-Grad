/* Copyright 2014-2018,2021 The PySCF Developers. All Rights Reserved.
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
 */

/* Fused overlap and i/j-center derivatives of a Fourier AO pair.
 *
 * PySCF's Fourier driver builds the common primitive recurrence before it
 * calls this contraction callback.  Producing the overlap and both center
 * derivatives in one callback avoids traversing every shell pair and
 * rebuilding that recurrence three times.  Component 0 is the overlap,
 * components 1:4 belong to center i, and 4:7 belong to center j.
 */

#include <complex.h>
#include "cint.h"
#include "gto/ft_ao.h"

extern void GTO_ft_nabla1i(double *f, double *g, int li, int lj,
                           FTEnvVars *envs);
extern void GTO_ft_nabla1j(double *f, double *g, int li, int lj,
                           FTEnvVars *envs);

static void accumulate_center(double *gout, double *g0R, double *g1R,
                              int *idx, FTEnvVars *envs, int empty,
                              int component_offset)
{
        const int nf = envs->nf;
        const int bs = envs->block_size;
        const size_t g_size = envs->g_size * bs;
        double *g0I = g0R + g_size * 3;
        double *g1I = g1R + g_size * 3;
        double *goutR = gout;
        double *goutI = gout + nf * bs * 7;
        double xyR, xyI, sR[3], sI[3];
        int ix, iy, iz, n, k;

        for (n = 0; n < nf; n++) {
                ix = idx[n * 3];
                iy = idx[n * 3 + 1];
                iz = idx[n * 3 + 2];
#pragma GCC ivdep
                for (k = 0; k < bs; k++) {
                        ZMUL(sR[0], sI[0], g1, g0, g0);
                        ZMUL(sR[1], sI[1], g0, g1, g0);
                        ZMUL(sR[2], sI[2], g0, g0, g1);
                        const int base = (n * 7 + component_offset) * bs + k;
                        int axis;
                        for (axis = 0; axis < 3; axis++) {
                                const int offset = base + axis * bs;
                                if (empty) {
                                        goutR[offset] = sR[axis];
                                        goutI[offset] = sI[axis];
                                } else {
                                        goutR[offset] += sR[axis];
                                        goutI[offset] += sI[axis];
                                }
                        }
                }
        }
}


static void accumulate_overlap(double *gout, double *g0R, int *idx,
                               FTEnvVars *envs, int empty)
{
        const int nf = envs->nf;
        const int bs = envs->block_size;
        const size_t g_size = envs->g_size * bs;
        double *g0I = g0R + g_size * 3;
        double *goutR = gout;
        double *goutI = gout + nf * bs * 7;
        double xyR, xyI, valueR, valueI;
        int ix, iy, iz, n, k;

        for (n = 0; n < nf; n++) {
                ix = idx[n * 3];
                iy = idx[n * 3 + 1];
                iz = idx[n * 3 + 2];
#pragma GCC ivdep
                for (k = 0; k < bs; k++) {
                        ZMUL(valueR, valueI, g0, g0, g0);
                        const int offset = n * 7 * bs + k;
                        if (empty) {
                                goutR[offset] = valueR;
                                goutI[offset] = valueI;
                        } else {
                                goutR[offset] += valueR;
                                goutI[offset] += valueI;
                        }
                }
        }
}


static void inner_prod_ipij(double *gout, double *g, int *idx,
                            FTEnvVars *envs, int empty)
{
        const size_t g_size = envs->g_size * envs->block_size;
        double *g0R = g;
        double *g0I = g0R + g_size * 3;
        double *g1R = g0I + g_size * 3;

        accumulate_overlap(gout, g0R, idx, envs, empty);
        GTO_ft_nabla1i(g1R, g0R, envs->i_l, envs->j_l, envs);
        accumulate_center(gout, g0R, g1R, idx, envs, empty, 1);
        GTO_ft_nabla1j(g1R, g0R, envs->i_l, envs->j_l, envs);
        accumulate_center(gout, g0R, g1R, idx, envs, empty, 4);
}

static int ft_ipij_driver(double *outR, double *outI, int *shls, int *dims,
                          FPtr_eval_gz eval_gz, double complex fac,
                          double *Gv, double *b, int *gxyz, int *gs,
                          int nGv, int block_size, int *atm, int natm,
                          int *bas, int nbas, double *env, double *cache,
                          int spherical)
{
        FTEnvVars envs;
        int ng[] = {1, 1, 0, 0, 1, 1, 0, 7};
        GTO_ft_init1e_envs(&envs, ng, shls, fac, Gv, b, gxyz, gs,
                           nGv, block_size, atm, natm, bas, nbas, env);
        envs.f_gout = &inner_prod_ipij;
        void (*transform)() = spherical
                ? (void (*)())GTO_ft_c2s_sph
                : (void (*)())GTO_ft_c2s_cart;
        return GTO_ft_aopair_drv(outR, outI, dims, eval_gz, cache,
                                 transform, &envs);
}

#define DECLARE_CALLBACK(NAME, SPH)                                        \
int NAME(double *outR, double *outI, int *shls, int *dims,                 \
         FPtr_eval_gz eval_gz, double complex fac, double *Gv, double *b,  \
         int *gxyz, int *gs, int nGv, int block_size, int *atm, int natm, \
         int *bas, int nbas, double *env, double *cache)                   \
{                                                                          \
        return ft_ipij_driver(outR, outI, shls, dims, eval_gz, fac, Gv, b, \
                              gxyz, gs, nGv, block_size, atm, natm, bas,    \
                              nbas, env, cache, SPH);                       \
}

DECLARE_CALLBACK(DCDT_ft_ovlp_ipij_cart, 0)
DECLARE_CALLBACK(DCDT_ft_ovlp_ipij_sph, 1)
