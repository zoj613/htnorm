/* Copyright (c) 2020, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include <math.h>
#include <stddef.h>
#include <string.h>

#include "blas.h"
#include "dist.h"


/* Generate a sample from the standard normal distribution using the 
 * Marsaglia-Polar method.
 *
 * TODO: Think about using a faster one, maybe the Ziggurat method?*/
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
    } while (!(s < 1.0));

    z = sqrt(-2 * log(s) / s);
    y = v * z;
    x = u * z;
    cached = true;
    return x;
}


static inline void
std_normal_rand_fill(rng_t* rng, size_t n, double* out)
{
    for (size_t i = n; i--; )
        out[i] = std_normal_rand(rng);
}


inline mvn_output_t*
mvn_output_new(size_t nrow)
{
    mvn_output_t* out = malloc(sizeof(mvn_output_t));
    if (out != NULL) {
        out->v = malloc(nrow * sizeof(*out->v));
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
mvn_rand_cov(rng_t* rng, const double* mean, const double* cov, size_t nrow,
             bool diag, double* out)
{
    lapack_int info = 0;
    size_t i;

    if (diag) {
        for (i = nrow; i--; )
            out[i] = mean[i] + sqrt(cov[nrow * i + i]) * std_normal_rand(rng);
        return info;
    }

    double* factor = malloc(nrow * nrow * sizeof(*factor));
    if (factor == NULL)
        return HTNORM_ALLOC_ERROR;

    // do cholesky factorization.
    memcpy(factor, cov, nrow * nrow * sizeof(*factor));
    info = POTRF(nrow, factor, nrow);
    if (!info) {
        // triangular matrix-vector product. U^T * z.
        std_normal_rand_fill(rng, nrow, out);
        TRMV_T(nrow, factor, nrow, out, 1);
        // out = out + mean, where out = L * z
        for (i = nrow; i--; )
            out[i] += mean[i];
    }

    free(factor);
    return info;
}


int
mvn_rand_prec(rng_t* rng, const double* prec, size_t nrow, type_t type,
              mvn_output_t* out, bool full_inv)
{
    lapack_int info = 0;
    // if precision is diagonal then use a direct way to calculate output.
    if (type == IDENTITY) {
        std_normal_rand_fill(rng, nrow, out->v);
        return info;
    }
    else if (type == DIAGONAL) {
        size_t diag_index;
        for (size_t i = nrow; i--; ) {
            diag_index = nrow * i + i;
            out->cov[diag_index] = 1.0 / prec[diag_index];
            out->v[i] = std_normal_rand(rng) * sqrt(out->cov[diag_index]);
        }
        return info;
    }

    double* factor = malloc(nrow * nrow * sizeof(*factor));
    if (factor == NULL)
        return HTNORM_ALLOC_ERROR;

    memcpy(factor, prec, nrow * nrow * sizeof(*prec));
    info = POTRF(nrow, factor, nrow);
    if (!info) {
        // sample from N(0, prec) using cholesky factor (i.e, calculate U^T * z)
        std_normal_rand_fill(rng, nrow, out->v);
        TRMV_T(nrow, factor, nrow, out->v, 1);
        // solve system using cholesky factor to get: out ~ N(0, prec_inv)
        info = POTRS(nrow, 1, factor, nrow, out->v, 1);
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
