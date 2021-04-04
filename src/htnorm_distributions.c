/* Copyright (c) 2020, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include <math.h>
#include <stddef.h>
#include <string.h>

#include "htnorm_blas.h"
#include "htnorm_distributions.h"
#include "htnorm_ziggurat_constants.h"

#define std_normal_rand_fill(rng, arr_size, arr) \
    for (int ii = (arr_size); ii--;) (arr)[ii] = std_normal_rand((rng))


// Generate a sample from the standard normal distribution using the Ziggurat method.
// This uses numpy's implementation of the algorithm.
static ALWAYS_INLINE(double)
std_normal_rand(rng_t* rng)
{
    uint64_t r, rabs;
    int sign, idx;
    double x, xx, yy;

    for (;;) {
        r = rng->next_uint64(rng->base);
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


// sample from a multivariate-normal distribution N(mean, cov)
int
mvn_rand_cov(rng_t* rng, const double* mean, const double* cov, int nrow,
             bool diag, double* restrict out)
{
    if (diag) {
        for (size_t i = nrow; i--; )
            out[i] = mean[i] + sqrt(cov[nrow * i + i]) * std_normal_rand(rng);
        return 0;
    }

    double* factor = malloc(nrow * nrow * sizeof(*factor));
    if (factor == NULL)
        return HTNORM_ALLOC_ERROR;

    int info;
    static const int incx = 1;
    static const double one = 1;
    memcpy(factor, cov, nrow * nrow * sizeof(*factor));
    // do cholesky factorization.
    POTRF(nrow, factor, nrow, info);
    if (!info) {
        std_normal_rand_fill(rng, nrow, out);
        // triangular matrix-vector product. L * z.
        TRMV(nrow, factor, nrow, out, incx);
        // mean + L * z
        AXPY(nrow, one, mean, out);
    }

    free(factor);
    return info;
}


int
mvn_rand_prec(rng_t* rng, const double* prec, int nrow, type_t prec_type,
              mvn_output_t* out)
{
    switch (prec_type) {
        case IDENTITY:
            // if precision is diagonal then use a direct way to calculate output.
            std_normal_rand_fill(rng, nrow, out->v);
            memset(out->factor, 1, nrow * sizeof(*out->factor));
            return 0;
        case DIAGONAL:
            // we save the factor as the precision since it is diagonal. This
            // way we get to save computation steps later when required to
            // reconstruct the full precision by squaring it's factor.
            for (size_t i = nrow; i--; ) {
                out->factor[i] = prec[nrow * i + i];
                out->v[i] = std_normal_rand(rng) / sqrt(prec[nrow * i + i]);
            }
            return 0;
        default: {
            // when precision matrix is neither diagonal nor identity
            int info;
            static const int incx = 1;
            memcpy(out->factor, prec, nrow * nrow * sizeof(*prec));
            POTRF(nrow, out->factor, nrow, info);
            if (!info) {
                // solve a triangular system L^T * x = v to get a sample from N(0, prec)
                // where `L` is the cholesky factor of the precision matrix
                std_normal_rand_fill(rng, nrow, out->v);
                TRSV(nrow, out->factor, nrow, out->v, incx);
            }
            return info;
        }
    }
}
