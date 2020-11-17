/* Copyright (c) 2020, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "blas.h"
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
mvn_output_new(size_t nrow)
{
    mvn_output_t* out = malloc(sizeof(mvn_output_t));
    if (out != NULL) {
        out->v = calloc(nrow, sizeof(*out->v));
        out->cov = calloc(nrow * nrow, sizeof(*out->cov));
    }
    return out;
}


inline void
mvn_output_free(mvn_output_t* a)
{
    free(a->cov);
    free(a->v);
    free(a);
}


int
mv_normal_rand(rng_t* rng, const double* mean, const double* cov, size_t nrow,
               bool diag, double* out)
{
    lapack_int info = 0;
    size_t i;

    if (diag) {
        for (i = nrow; i--; )
            out[i] = mean[i] + sqrt(cov[nrow * i + i]) * std_normal_rand(rng);
        return info;
    }
    
    double* factor = calloc(nrow * nrow, sizeof(*factor));
    if (factor == NULL)
        return HTNORM_ALLOC_ERROR;

    // do cholesky factorization.
    memcpy(factor, cov, nrow * nrow * sizeof(*cov));
    info = POTRF(nrow, factor, nrow);  
    if (!info) {
        // triangular matrix-vector product. L * z.
        std_normal_rand_fill(rng, nrow, out);
        TRMV(nrow, factor, nrow, out, 1);
        // out = out + mean, where out = L * z
        for (i = nrow; i--; )
            out[i] += mean[i];
    }

    free(factor);
    return info;
}


int
mv_normal_rand_prec(rng_t* rng, const double* prec, size_t nrow, bool diag,
                    mvn_output_t* out, bool full_inv)
{
    lapack_int info = 0;
    // if precision is diagonal then use a direct way to calculate output.
    if (diag) {
        size_t diag_index; 
        for (size_t i = nrow; i--; ) {
            diag_index = nrow * i + i;
            out->cov[diag_index] = 1.0 / prec[diag_index]; 
            out->v[i] = std_normal_rand(rng) / sqrt(out->cov[diag_index]);
        }
        return info;
    }

    double* factor = calloc(nrow * nrow, sizeof(*factor));
    if (factor == NULL)
        return HTNORM_ALLOC_ERROR;

    memcpy(factor, prec, nrow * nrow * sizeof(*prec));
    info = POTRF(nrow, factor, nrow);
    if (!info) {
        // sample from N(0, prec) using cholesky factor (i.e, calculate L * z)
        std_normal_rand_fill(rng, nrow, out->v);
        TRMV(nrow, factor, nrow, out->v, 1);
        // solve system using cholesky factor to get: out ~ N(0, prec_inv)
        info = POTRS(nrow, 1, factor, nrow, out->v, nrow); 
        if (!info) {
            // calculate the explicit inverse needed with the output
            info = POTRI(nrow, factor, nrow);
            // replace cov with factor to avoid a copy of the contents
            free(out->cov);
            out->cov = factor;
            if (full_inv)
                for (size_t i = 0; i < nrow; i++)
                    for (size_t j = 0; j < i; j++)
                        out->cov[nrow * j + i] = out->cov[nrow * i + j];

            return info;
        }
    }

    free(factor);
    return info;
}
