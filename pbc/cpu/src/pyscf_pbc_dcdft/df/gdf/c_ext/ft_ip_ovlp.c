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
 *
 * Adapted from PySCF's ft_ao_deriv.c Fourier AO derivative implementation.
 * This version exports separate i- and j-center callbacks for periodic GDF.
 */

/*
 * Fourier AO-pair center derivatives for periodic GDF.
 *
 * PySCF exposes the PBC_ft_bvk_drv driver and the low-level
 * GTO_ft_nabla1i/GTO_ft_nabla1j recurrences, but it does not export public
 * intor callbacks combining them.  The long-range GDF source derivative needs
 * the two centers separately, so this extension supplies four callbacks:
 * Cartesian/spherical and derivative-on-i/derivative-on-j.
 *
 * The numerical convention is identical to PySCF's GTO_ft_ovlp callback.
 * Only the Gaussian recurrence used for one AO center changes.  All lattice
 * image sums, k-point phases, AO sorting, and complex output layout remain
 * owned by PySCF's PBC_ft_bvk_drv.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <complex.h>
#include "cint.h"
#include "gto/ft_ao.h"

extern void GTO_ft_nabla1i(double *f, double *g, int li, int lj,
                           FTEnvVars *envs);
extern void GTO_ft_nabla1j(double *f, double *g, int li, int lj,
                           FTEnvVars *envs);

static void contract_ip(double *gout, double *g, int *idx,
                        FTEnvVars *envs, int empty, int on_j)
{
        const int nf = envs->nf;
        const int bs = envs->block_size;
        const size_t g_size = envs->g_size * bs;
        double *g0R = g;
        double *g0I = g0R + g_size * 3;
        double *g1R = g0I + g_size * 3;
        double *g1I = g1R + g_size * 3;
        double *goutR = gout;
        double *goutI = gout + nf * bs * 3;
        double xyR, xyI, sR[3], sI[3];
        int ix, iy, iz, n, k;

        if (on_j) {
                GTO_ft_nabla1j(g1R, g0R, envs->i_l, envs->j_l, envs);
        } else {
                GTO_ft_nabla1i(g1R, g0R, envs->i_l, envs->j_l, envs);
        }
        for (n = 0; n < nf; n++) {
                ix = idx[n * 3];
                iy = idx[n * 3 + 1];
                iz = idx[n * 3 + 2];
#pragma GCC ivdep
                for (k = 0; k < bs; k++) {
                        ZMUL(sR[0], sI[0], g1, g0, g0);
                        ZMUL(sR[1], sI[1], g0, g1, g0);
                        ZMUL(sR[2], sI[2], g0, g0, g1);
                        int offset = (n * 3) * bs + k;
                        if (empty) {
                                goutR[offset] = sR[0];
                                goutR[offset + bs] = sR[1];
                                goutR[offset + 2 * bs] = sR[2];
                                goutI[offset] = sI[0];
                                goutI[offset + bs] = sI[1];
                                goutI[offset + 2 * bs] = sI[2];
                        } else {
                                goutR[offset] += sR[0];
                                goutR[offset + bs] += sR[1];
                                goutR[offset + 2 * bs] += sR[2];
                                goutI[offset] += sI[0];
                                goutI[offset + bs] += sI[1];
                                goutI[offset + 2 * bs] += sI[2];
                        }
                }
        }
}

static void inner_prod_ipi(double *gout, double *g, int *idx,
                           FTEnvVars *envs, int empty)
{
        contract_ip(gout, g, idx, envs, empty, 0);
}

static void inner_prod_ipj(double *gout, double *g, int *idx,
                           FTEnvVars *envs, int empty)
{
        contract_ip(gout, g, idx, envs, empty, 1);
}

static int ft_ip_driver(double *outR, double *outI, int *shls, int *dims,
                        FPtr_eval_gz eval_gz, double complex fac,
                        double *Gv, double *b, int *gxyz, int *gs,
                        int nGv, int block_size, int *atm, int natm,
                        int *bas, int nbas, double *env, double *cache,
                        int on_j, int spherical)
{
        FTEnvVars envs;
        int ng_i[] = {1, 0, 0, 0, 1, 1, 0, 3};
        int ng_j[] = {0, 1, 0, 0, 1, 1, 0, 3};
        int *ng = on_j ? ng_j : ng_i;
        GTO_ft_init1e_envs(&envs, ng, shls, fac, Gv, b, gxyz, gs,
                           nGv, block_size, atm, natm, bas, nbas, env);
        envs.f_gout = on_j ? &inner_prod_ipj : &inner_prod_ipi;
        void (*transform)() = spherical
                ? (void (*)())GTO_ft_c2s_sph
                : (void (*)())GTO_ft_c2s_cart;
        return GTO_ft_aopair_drv(outR, outI, dims, eval_gz, cache,
                                 transform, &envs);
}

#define DECLARE_CALLBACK(NAME, ON_J, SPH)                                  \
int NAME(double *outR, double *outI, int *shls, int *dims,                  \
         FPtr_eval_gz eval_gz, double complex fac, double *Gv, double *b,   \
         int *gxyz, int *gs, int nGv, int block_size, int *atm, int natm,  \
         int *bas, int nbas, double *env, double *cache)                    \
{                                                                           \
        return ft_ip_driver(outR, outI, shls, dims, eval_gz, fac, Gv, b,    \
                            gxyz, gs, nGv, block_size, atm, natm, bas, nbas, \
                            env, cache, ON_J, SPH);                          \
}

DECLARE_CALLBACK(GTO_ft_ipovlp_i_cart, 0, 0)
DECLARE_CALLBACK(GTO_ft_ipovlp_i_sph,  0, 1)
DECLARE_CALLBACK(GTO_ft_ipovlp_j_cart, 1, 0)
DECLARE_CALLBACK(GTO_ft_ipovlp_j_sph,  1, 1)

static struct PyModuleDef module_definition = {
        PyModuleDef_HEAD_INIT,
        "_ft_ip_ovlp",
        "PySCF-compatible periodic Fourier AO-pair derivative callbacks.",
        -1,
        NULL,
};

PyMODINIT_FUNC PyInit__ft_ip_ovlp(void)
{
        return PyModule_Create(&module_definition);
}
