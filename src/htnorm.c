/* Copyright (c) 2020, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

#include "blas.h"
#include "dist.h"
#include "../include/htnorm.h"

// special case for when g matrix has dimensions 1 by n (a 1d array)
static int
htnorm_rand_g_a_vec(const matrix_t* cov, bool diag, const double* g,
                    double r, double* out)
{
    const size_t cncol = cov->ncol;
    double alpha = 0, g_cov_g = 0;
    size_t i, j;

    double* cov_g = calloc(cncol, sizeof(*cov_g));
    if (cov_g == NULL)
        return HTNORM_ALLOC_ERROR;

    // compute: r - g * y; where y ~ N(mean, cov)
    for (i = cncol; i--; )
        alpha +=  g[i] * out[i]; 
    alpha = r - alpha;

    // compute: cov * g^T
    if (diag) {
        // optimize when cov if diagonal
        for (i = cncol; i--; )
            cov_g[i] = cov->mat[cncol * i + i] * g[i];
    }
    else {
        SYMV(cncol, 1.0, cov->mat, cncol, g, 1, 0.0, cov_g, 1);
    }

    // out = y + cov * g^T * alpha, where alpha = (r - g * y) / (g * cov * g^T)
    for (i = cncol; i--; ) {
        g_cov_g = 0;
        for (j = cncol; j--; )
            g_cov_g += g[j] * cov_g[j];
        out[i] += alpha * (cov_g[i] / g_cov_g);
    } 

    free(cov_g);
    return 0;
}


int
htnorm_rand(rng_t* rng, const double* mean, const matrix_t* cov,
            bool diag, const matrix_t* g, const double* r, double* out)
{
    const size_t gncol = g->ncol;
    const size_t gnrow = g->nrow;
    const double* gmat = g->mat;

    lapack_int info = mv_normal_rand(rng, mean, cov->mat, gncol, diag, out);
    // early return upon failure
    if (info)
        return info;
    // check if g's number of rows is 1 and use an optimized function
    if (gnrow == 1)
        return htnorm_rand_g_a_vec(cov, diag, gmat, *r, out);

    double* gy = calloc(gnrow, sizeof(*gy));
    if (gy == NULL)
        return HTNORM_ALLOC_ERROR;

    double* cov_g = calloc(gnrow * gncol, sizeof(*cov_g));
    if (cov_g == NULL) {
        info = HTNORM_ALLOC_ERROR;
        goto covg_failure_cleanup;
    }

    double* g_cov_g = calloc(gnrow * gnrow, sizeof(*g_cov_g));
    if (g_cov_g == NULL) {
        info = HTNORM_ALLOC_ERROR;
        goto gcovg_failure_cleanup;
    }

    // compute: r - g*y
    memcpy(gy, r, gnrow * sizeof(*gy)); 
    GEMV(gnrow, gncol, -1.0, gmat, gncol, out, 1, 1.0, gy, 1);
    // compute: cov * g^T
    if (diag) {
        for (size_t i = 0; i < gncol; i++)
            for (size_t j = 0; j < gnrow; j++)
                cov_g[gnrow * i + j] = cov->mat[gncol * i + i] * gmat[gncol * j + i];
    }
    else {
        GEMM_NT(gncol, gnrow, gncol, 1.0, cov->mat, gncol, gmat,
                gncol, 0.0, cov_g, gnrow); 
    }
    // compute: g * cov * g^T
    GEMM(gnrow, gnrow, gncol, 1.0, gmat, gncol, cov_g, gnrow, 0.0, g_cov_g, gnrow);
    // factorize g_cov_g using cholesky method.
    info = POTRF(gnrow, g_cov_g, gnrow);
    if (!info) {                      
        // solve a system of linear equations: g * cov * g^T * alpha = r - g*y
        // value of alpha is store in `gy` array.
        info = POTRS(gnrow, 1, g_cov_g, gnrow, gy, 1); 
        // compute: out = cov * g^T * alpha + out
        if (!info)  
            GEMV(gncol, gnrow, 1.0, cov_g, gnrow, gy, 1, 1.0, out, 1);
    }

    free(g_cov_g); 
gcovg_failure_cleanup:
    free(cov_g);
covg_failure_cleanup:
    free(gy);

    return info;
} 


int
htnorm_rand2(rng_t* rng, const double* mean, const matrix_t* a, bool a_diag,
             const matrix_t* phi, const matrix_t* omega, bool o_diag, double* out)
{
    lapack_int info;
    const size_t pnrow = phi->nrow;
    const size_t pncol = phi->ncol;
    const double* pmat = phi->mat;

    mvn_output_t* y1 = mvn_output_new(pncol);
    if (y1 == NULL || y1->v == NULL || y1->cov == NULL) {
        info = HTNORM_ALLOC_ERROR;
        goto y1_failure_cleanup;
    }

    mvn_output_t* y2 = mvn_output_new(pnrow);
    if (y2 == NULL || y2->v == NULL || y2->cov == NULL) {
        info = HTNORM_ALLOC_ERROR;
        goto y2_failure_cleanup;
    }

    if((info = mv_normal_rand_prec(rng, a->mat, pncol, a_diag, y1, false)) ||
        (info = mv_normal_rand_prec(rng, omega->mat, pnrow, o_diag, y2, true)))
        goto y2_failure_cleanup;

    double* x = calloc(pnrow * pncol, sizeof(*x));
    if (x == NULL) {
        info = HTNORM_ALLOC_ERROR;
        goto y2_failure_cleanup;
    }

    // compute: x = A_inv * phi^T
    GEMM_NT(pncol, pnrow, pncol, 1.0, a->mat, pncol, pmat, pncol, 0.0, x, pnrow);
    // compute: phi * A_inv * phi^T + omega_inv
    GEMM(pnrow, pnrow, pncol, 1.0, pmat, pncol, x, pnrow, 1.0, y2->cov, pnrow);
    // compute: phi * y1 + y2
    GEMV(pnrow, pncol, 1.0, pmat, pncol, y1->v, 1, 1.0, y2->v, 1);

    // compute: mean + y1
    for (size_t i = pncol; i--; )
        out[i] = y1->v[i] + mean[i];

    // solve for alpha: (omega_inv + phi * A_inv * phi^T) * alpha = phi * y1 + y2
    info = POTRF(pnrow, y2->cov, pnrow);
    if (!info) {
        info = POTRS(pnrow, 1, y2->cov, pnrow, y2->v, 1);
        if (!info)
            // compute: -A_inv * phi^T * alpha + mean + y1
            GEMV(pncol, pnrow, -1.0, x, pnrow, y2->v, 1, 1.0, out, 1);
    } 

    free(x);
y2_failure_cleanup:
    mvn_output_free(y2);
y1_failure_cleanup:
    mvn_output_free(y1);
 
    return info;
}
