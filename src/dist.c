/* Copyright (c) 2020, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include <math.h>
#include <stddef.h>
#include <string.h>

#include "blas.h"
#include "dist.h"

#define std_normal_rand_fill(rng_t, arr_size, arr) \
    for (size_t inc = (arr_size); inc--;) (arr)[inc] = std_normal_rand((rng_t))

extern ALWAYS_INLINE(mvn_output_t*) mvn_output_new(size_t nrow, type_t factor_type);
extern ALWAYS_INLINE(void) mvn_output_free(mvn_output_t* a);

/* Generate a sample from the standard normal distribution using the 
 * Marsaglia-Polar method.
 *
 * TODO: Think about using a faster one, maybe the Ziggurat method?*/
static ALWAYS_INLINE(double)
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
        std_normal_rand_fill(rng, nrow, out);
        // triangular matrix-vector product. U^T * z.
#ifdef HTNORM_COLMAJOR
        TRMV(nrow, factor, nrow, out, 1);
#else
        TRMV_T(nrow, factor, nrow, out, 1);
#endif
        // out = out + mean, where out = L * z
        for (i = nrow; i--; )
            out[i] += mean[i];
    }

    free(factor);
    return info;
}


int
mvn_rand_prec(rng_t* rng, const double* prec, size_t nrow, type_t prec_type,
              mvn_output_t* out)
{
    lapack_int info = 0;
    switch (prec_type) {
        case IDENTITY:
            // if precision is diagonal then use a direct way to calculate output.
            std_normal_rand_fill(rng, nrow, out->v);
            for (size_t i = nrow; i--; )
                out->factor[i] = 1.0;
            return info;
        case DIAGONAL:
            // we save the factor as the precision since it is diagonal. This
            // way we get to save computation steps later when required to
            // reconstruct the full precision by squaring it's factor.
            for (size_t i = nrow; i--; ) {
                out->factor[i] = prec[nrow * i + i];
                out->v[i] = std_normal_rand(rng) / sqrt(prec[nrow * i + i]);
            }
            return info;
        default:
            // when precision matrix is neither diagonal nor identity
            memcpy(out->factor, prec, nrow * nrow * sizeof(*prec));
            info = POTRF(nrow, out->factor, nrow);
            if (!info) {
                std_normal_rand_fill(rng, nrow, out->v);
                // solve a triangular system Ux = v to get a sample from N(0, prec)
                // where `U` is the cholesky factor of the precision matrix
#ifdef HTNORM_COLMAJOR
                TRTRS_T(nrow, 1, out->factor, nrow, out->v, nrow);
#else
                TRTRS(nrow, 1, out->factor, nrow, out->v, 1);
#endif
            }
            return info;
    }
}
