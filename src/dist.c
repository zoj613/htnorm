/* Copyright (c) 2020, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include <math.h>
#include <stddef.h>
#include <string.h>

#include "blas.h"
#include "dist.h"
#include "zig_constants.h"

#define std_normal_rand_fill(rng_t, arr_size, arr) \
    for (size_t inc = (arr_size); inc--;) (arr)[inc] = std_normal_rand((rng_t))


// Generate a sample from the standard normal distribution using the Ziggurat method.
// This uses numpy's implementation of the algorithm.
static ALWAYS_INLINE(double)
std_normal_rand(rng_t* rng)
{
    uint64_t r, rabs;
    int sign, idx;
    double x, xx, yy;

    for (;;) {
        r = rng->next_int(rng->base);
        idx = r & 0xff;
        r >>= 8;
        sign = r & 0x1;
        rabs = (r >> 1) & 0x000fffffffffffff;
        x = rabs * wi_double[idx];

        if (sign & 0x1)
            x = -x;
        if (rabs < ki_double[idx])
            return x; /* 99.3% of the time return here */

        if (idx == 0) {
            // use tail sampling method.
            for (;;) {
            xx = -ziggurat_nor_inv_r * log(1.0 - rng->next_double(rng->base));
            yy = -log(1.0 - rng->next_double(rng->base));
            if (yy + yy > xx * xx)
                return ((rabs >> 8) & 0x1) ? -(ziggurat_nor_r + xx) : ziggurat_nor_r + xx;
            }
        }
        else {
            // try and get a sample at the wedge.
            if (((fi_double[idx - 1] - fi_double[idx]) *
                rng->next_double(rng->base) + fi_double[idx]) < exp(-0.5 * x * x))
            return x;
        }
    }
} 


int
mvn_rand_cov(rng_t* rng, const double* mean, const double* cov, size_t nrow,
             bool diag, double* restrict out)
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
