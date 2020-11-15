#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include <cblas.h>
#include <lapacke.h>

#include "dist.h"


static double
std_normal_rand(rng_t* rng)
{
    double s, u, v, z;
    static double x, y;
    static bool cached = false;

    if (cached) {
        cached = false;
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
    cached = true;
    return x; 
}


static inline void
std_normal_rand_fill(rng_t* rng, int n, double* out)
{
    for (size_t i = n; i--; )
        out[i] = std_normal_rand(rng);
}


inline mvn_output_t*
mvn_output_new(int nrow)
{
    mvn_output_t* out = malloc(sizeof(mvn_output_t));
    if (out != NULL) {
        out->v = malloc(nrow * sizeof(double));
        out->cov = calloc(nrow * nrow, sizeof(double));
    }
    return out;
}


inline void
mvn_output_free(mvn_output_t* a)
{
    free(a->v);
    free(a->cov);
    free(a);
}


int
mv_normal_rand(rng_t* rng, const double* mean, const double* cov, int nrow,
               bool diag, double* out)
{
    lapack_int info = 0;
    int i;

    if (diag) {
        for (i = nrow; i--; )
            out[i] = mean[i] + sqrt(cov[nrow * i + i]) * std_normal_rand(rng);
        return info;
    }
    
    const size_t factor_size = nrow * nrow * sizeof(double);
    double* factor = malloc(factor_size);
    if (factor == NULL)
        return HTNORM_ALLOC_ERROR;

    memcpy(factor, cov, factor_size); 
    // do cholesky factorization and store result in lower tringular part
    info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', nrow, factor, nrow);  
    if (!info) {
        // triangular matrix-vector product. L * z.
        std_normal_rand_fill(rng, nrow, out);
        cblas_dtrmv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                    nrow, factor, nrow, out, 1);
        // out = out + mean, where out = L * z
        for (i = 0; i < nrow; i++)
            out[i] += mean[i];
    }

    free(factor);
    return info;
}


int
mv_normal_rand_prec(rng_t* rng, const double* prec, int nrow, bool diag,
                    mvn_output_t* out, bool full_inv)
{
    lapack_int info = 0;
    int i;

    // if precision is diagonal then use a direct way to calculate output.
    if (diag) {
        size_t diag_index; 
        for (i = nrow; i--; ) {
            diag_index = nrow * i + i;
            out->cov[diag_index] = 1.0 / prec[diag_index]; 
            out->v[i] = std_normal_rand(rng) / sqrt(out->cov[diag_index]);
        }
        return info;
    }

    const size_t factor_size = nrow * nrow * sizeof(double);
    double* factor = malloc(factor_size);
    if (factor == NULL)
        return HTNORM_ALLOC_ERROR;

    memcpy(factor, prec, factor_size); 
    // do cholesky factorization and store result in Lower tringular part
    info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', nrow, factor, nrow);
    if (!info) {
        // sample from N(0, prec) using cholesky factor (i.e, calculate L * z)
        std_normal_rand_fill(rng, nrow, out->v);
        cblas_dtrmv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                    nrow, factor, nrow, out->v, 1);
        // solve system using cholesky factor to get: out ~ N(0, prec_inv)
        info = LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', nrow, 1,
                              factor, nrow, out->v, nrow); 
        if (!info) {
            // calculate the explicit inverse needed with the output
            info = LAPACKE_dpotri(LAPACK_ROW_MAJOR, 'L', nrow, factor, nrow);
            // replace cov with factor to avoid a copy of the contents
            free(out->cov);
            out->cov = factor;
            if (full_inv)
                for (i = 0; i < nrow; i++)
                    for (int j = 0; j < i; j++)
                        out->cov[nrow * j + i] = out->cov[nrow * i + j];

            return info;
        }
    }

    free(factor);
    return info;
}
