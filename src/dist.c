#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <cblas.h>
#include <lapacke.h>

#include "dist.h"


mvn_output_t*
mvn_output_new(int nrow)
{
    mvn_output_t* out = malloc(sizeof(mvn_output_t));
    if (out != NULL) {
        out->v = malloc(nrow * sizeof(double));
        out->cov = calloc(nrow * nrow, sizeof(double));
    }
    return out;
}


void
mvn_output_free(mvn_output_t* a)
{
    free(a->v);
    free(a->cov);
    free(a);
}


double
uniform_rand(rng_t* rng, double low, double high)
{
    double u = rng->next_double(rng->base);
    return low + (high - low) * u;
}


double
std_normal_rand(rng_t* rng)
{
    double s, u, v, z;
    static double x, y;
    static int cached = 0;

    if (cached) {
        cached--;
        return y;
    }

    do {
        u = 2.0 * rng->next_double(rng->base) - 1;
        v = 2.0 * rng->next_double(rng->base) - 1;
        s = u * u + v * v;
    } while (s >= 1.0);

    z = sqrt(-2 * log(s) / s);
    y = v * z;
    x = u * z;
    cached++;
    return x; 
}


void
std_normal_rand_fill(rng_t* rng, int n, double* out)
{
    for (int i = 0; i < n; i++)
        out[i] = std_normal_rand(rng);
}


int
mv_normal_rand(rng_t* rng, const double* mean, const double* cov, int nrow,
               bool diag, double* out)
{
    lapack_int info = 0;

    if (diag) {
        for (int i = 0; i < nrow; i++)
            out[i] = mean[i] + sqrt(cov[nrow * i + i]) * std_normal_rand(rng);
        return info;
    }
    
    double* factor = malloc(sizeof(double) * nrow * nrow);
    if (factor == NULL)
        return HTNORM_ALLOC_ERROR;

    cblas_dcopy(nrow * nrow, cov, 1, factor, 1);

    // do cholesky factorization and store result in lower tringular part
    info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', nrow, factor, nrow);  
    if (!info) {
        // triangular matrix-vector product. L * z.
        std_normal_rand_fill(rng, nrow, out);
        cblas_dtrmv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                    nrow, factor, nrow, out, 1);
        // out = out + mean, where out = L * z
        cblas_daxpy(nrow, 1.0, mean, 1, out, 1);
    }

    free(factor);
    return info;
}


int
mv_normal_rand_prec(rng_t* rng, const double* prec, int nrow, bool diag,
                    mvn_output_t* out)
{
    lapack_int info = 0;
    int i;

    std_normal_rand_fill(rng, nrow, out->v);

    // if precision is diagonal then use a direct way to calculate output.
    if (diag) {
        size_t diag_index; 
        for (i = 0; i < nrow; i++) {
            diag_index = nrow * i + i;
            out->cov[diag_index] = 1.0 / prec[diag_index]; 
            out->v[i] *= out->cov[diag_index];
        }
        return info;
    }
    
    double* factor = malloc(sizeof(double) * nrow * nrow);
    if (factor == NULL)
        return HTNORM_ALLOC_ERROR;

    cblas_dcopy(nrow * nrow, prec, 1, factor, 1);
    // do cholesky factorization and store result in Lower tringular part
    info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', nrow, factor, nrow);
    if (!info) {
        // solve system using cholesky factor to get the normal variate
        info = LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', nrow, 1,
                              factor, nrow, out->v, nrow); 
        if (!info) {
            // make an Identity matrix so as to compute inverse of prec
            for (i = 0; i < nrow; i++)
                out->cov[nrow * i + i] = 1.0;

            info = LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'U', nrow, nrow,
                                  factor, nrow, out->cov, nrow); 
        }
    }

    free(factor);
    return info;
}
