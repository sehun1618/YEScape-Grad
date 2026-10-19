/* Original-Becke moving-grid derivative, following PySCF grad/rks.py.
 * Two pair passes: p_B first, then contract directly into d(p_A/sum p_B).
 * Each grid point is independent; no atom-pair derivative tensor is stored.
 */
#include <Python.h>
#include <math.h>
#include <stdlib.h>
#include <stddef.h>
#include <omp.h>

static inline void switching(double g, double a, double *s, double *u, double *t) {
    double p0 = g + a*(1-g*g);
    double p1 = (3-p0*p0)*p0*.5;
    double p2 = (3-p1*p1)*p1*.5;
    double p3 = (3-p2*p2)*p2*.5;
    *s = .5*(1-p3+1e-200);
    *u = .5*(1+p3+1e-200);
    *t = (27./8.)*(1-p2*p2)*(1-p1*p1)*(1-p0*p0)*(.5*(1-2*a*g));
}

int partition(int n, int ng, int owner, int threads, const double *atoms,
              const double *coords, const double *vol, const double *adjust,
              const double *inv_r, double *w0, double *w1) {
    double *scratch = malloc((size_t)threads*5*n*sizeof(double));
    if (!scratch) return 1;
    #pragma omp parallel num_threads(threads)
    {
        double *d = scratch + (size_t)omp_get_thread_num()*5*n;
        double *v = d+n, *p = v+3*n;
        #pragma omp for schedule(static)
        for (int g=0; g<ng; ++g) {
            for (int i=0; i<n; ++i) {
                double r2=0;
                for (int x=0; x<3; ++x) {
                    v[3*i+x] = atoms[3*i+x]-coords[3*g+x];
                    r2 += v[3*i+x]*v[3*i+x];
                    w1[((size_t)i*3+x)*ng+g]=0;
                }
                d[i]=sqrt(r2)+1e-200; p[i]=1;
                for (int x=0; x<3; ++x) v[3*i+x]/=d[i];
            }
            for (int i=0; i<n; ++i) for (int j=0; j<i; ++j) {
                double s,u,t;
                switching((d[i]-d[j])*inv_r[i*n+j],adjust[i*n+j],&s,&u,&t);
                p[i]*=s; p[j]*=u;
            }
            double total=0;
            for (int i=0; i<n; ++i) total+=p[i];
            double z=1/total, pa=p[owner]*z;
            w0[g]=vol[g]*pa;
            for (int i=0; i<n; ++i) for (int j=0; j<i; ++j) {
                double r=inv_r[i*n+j],s,u,t;
                switching((d[i]-d[j])*r,adjust[i*n+j],&s,&u,&t);
                double ci=((i==owner)-pa)*(p[i]*z)*(-t/s);
                double cj=((j==owner)-pa)*(p[j]*z)*(t/u);
                double c=(ci+cj)*vol[g];
                for (int x=0; x<3; ++x) {
                    double offset=(atoms[3*i+x]-atoms[3*j+x])*(r*r*r)*(d[i]-d[j]);
                    double di=r*v[3*i+x]-offset, dj=r*v[3*j+x]-offset;
                    w1[((size_t)i*3+x)*ng+g]+=c*(i==owner?dj:di);
                    w1[((size_t)j*3+x)*ng+g]-=c*(j==owner?di:dj);
                    if (i!=owner && j!=owner)
                        w1[((size_t)owner*3+x)*ng+g]-=c*r*(v[3*i+x]-v[3*j+x]);
                }
            }
        }
    }
    free(scratch);
    return 0;
}

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_grid_response_becke",
    "OpenMP Becke moving-grid response (loaded through ctypes).", -1, NULL
};
PyMODINIT_FUNC PyInit__grid_response_becke(void) {
    return PyModule_Create(&module);
}
