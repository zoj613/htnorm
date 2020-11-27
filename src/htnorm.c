/* Copyright (c) 2020, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include <stddef.h>
#include <string.h>

#include "blas.h"
#include "dist.h"

// special case for when g matrix has dimensions 1 by n (a 1d array)
static int
hyperplane_truncated_norm_1d_g(const ht_config_t* conf, double* out)
{
    bool diag = conf->diag;
    size_t ncol = conf->gncol;
    double r = *(conf->r);
    const double* cov = conf->mean;
    const double* g = conf->g;

    double alpha = 0, g_cov_g = 0;
    size_t i, j;

    double* cov_g = malloc(ncol * sizeof(*cov_g));
    if (cov_g == NULL)
        return HTNORM_ALLOC_ERROR;

    // compute: r - g * y; where y ~ N(mean, cov)
    for (i = ncol; i--; )
        alpha +=  g[i] * out[i]; 
    alpha = r - alpha;

    // compute: cov * g^T
    if (diag) {
        // optimize when cov if diagonal
        for (i = ncol; i--; )
            cov_g[i] = cov[ncol * i + i] * g[i];
    }
    else {
        SYMV(ncol, 1.0, cov, ncol, g, 1, 0.0, cov_g, 1);
    }

    // out = y + cov * g^T * alpha, where alpha = (r - g * y) / (g * cov * g^T)
    for (i = ncol; i--; ) {
        g_cov_g = 0;
        for (j = ncol; j--; )
            g_cov_g += g[j] * cov_g[j];
        out[i] += alpha * (cov_g[i] / g_cov_g);
    } 

    free(cov_g);
    return 0;
}


int
htn_hyperplane_truncated_mvn(rng_t* rng, const ht_config_t* conf, double* out)
{
    const size_t gncol = conf->gncol;  // equal to the dimension of the covariance
    const size_t gnrow = conf->gnrow;
    const bool diag = conf->diag;
    const double* mean = conf->mean;
    const double* cov = conf->cov;
    const double* g = conf->g;
    const double* r = conf->r;

    lapack_int info = mvn_rand_cov(rng, mean, cov, gncol, diag, out);
    // early return upon failure
    if (info)
        return info;

    // check if g's number of rows is 1 and use an optimized function
    if (gnrow == 1)
        return hyperplane_truncated_norm_1d_g(conf, out);

    double* gy = malloc(gnrow * sizeof(*gy));
    if (gy == NULL)
        return HTNORM_ALLOC_ERROR;

    double* cov_g = malloc(gnrow * gncol * sizeof(*cov_g));
    if (cov_g == NULL) {
        info = HTNORM_ALLOC_ERROR;
        goto covg_failure_cleanup;
    }

    double* g_cov_g = malloc(gnrow * gnrow * sizeof(*g_cov_g));
    if (g_cov_g == NULL) {
        info = HTNORM_ALLOC_ERROR;
        goto gcovg_failure_cleanup;
    }

    // compute: r - g*y
    memcpy(gy, r, gnrow * sizeof(*gy)); 
    GEMV(gnrow, gncol, -1.0, g, gncol, out, 1, 1.0, gy, 1);
    // compute: cov * g^T
    if (diag) {
        for (size_t i = 0; i < gncol; i++)
            for (size_t j = 0; j < gnrow; j++)
                cov_g[gnrow * i + j] = cov[gncol * i + i] * g[gncol * j + i];
    }
    else {
        GEMM_NT(gncol, gnrow, gncol, 1.0, cov, gncol, g, gncol, 0.0, cov_g, gnrow); 
    }
    // compute: g * cov * g^T
    GEMM(gnrow, gnrow, gncol, 1.0, g, gncol, cov_g, gnrow, 0.0, g_cov_g, gnrow);
    // factorize g_cov_g using cholesky method and store in upper triangular part.
    // solve a positive definite system of linear equations: g * cov * g^T * alpha = r - g*y
    info = POSV(gnrow, 1, g_cov_g, gnrow, gy, 1);
    if (!info)
        // compute: out = cov * g^T * alpha + out
        GEMV(gncol, gnrow, 1.0, cov_g, gnrow, gy, 1, 1.0, out, 1);

    free(g_cov_g); 
gcovg_failure_cleanup:
    free(cov_g);
covg_failure_cleanup:
    free(gy);

    return info;
} 


int
htn_structured_precision_mvn(rng_t* rng, const sp_config_t* conf, double* out)
{
    lapack_int info;
    const size_t pnrow = conf->pnrow;
    const size_t pncol = conf->pncol;
    const type_t a_type = conf->a_id;
    const double* mean = conf->mean;
    const double* a = conf->a;
    const double* phi = conf->phi;
    const double* omega = conf->omega;

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

    if((info = mvn_rand_prec(rng, a, pncol, a_type , y1, false)) ||
        (info = mvn_rand_prec(rng, omega, pnrow, conf->o_id, y2, true)))
        goto y2_failure_cleanup;

    double* x = malloc(pnrow * pncol * sizeof(*x));
    if (x == NULL) {
        info = HTNORM_ALLOC_ERROR;
        goto y2_failure_cleanup;
    }

    // compute: x = A_inv * phi^T
    if (a_type == DIAGONAL) {
        for (size_t i = 0; i < pncol; i++)
            for (size_t j = 0; j < pnrow; j++)
                x[pnrow * i + j] = y1->cov[pncol * i + i] * phi[pncol * j + i];
    }
    else if (a_type == NORMAL) {
        GEMM_NT(pncol, pnrow, pncol, 1.0, y1->cov, pncol, phi, pncol, 0.0, x, pnrow);
    }
    // compute: phi * A_inv * phi^T + omega_inv
    if (a_type == IDENTITY) {
        GEMM_NT(pnrow, pnrow, pncol, 1.0, phi, pncol, phi, pncol, 1.0, y2->cov, pnrow);
    }
    else {
        GEMM(pnrow, pnrow, pncol, 1.0, phi, pncol, x, pnrow, 1.0, y2->cov, pnrow);
    }
    // compute: phi * y1 + y2
    GEMV(pnrow, pncol, 1.0, phi, pncol, y1->v, 1, 1.0, y2->v, 1);

    double coef = 1.0;
    if (conf->struct_mean) {
        // compute: t - (phi * y1 + y2)
        for (size_t i = pncol; i--; ) {
            if (i < pnrow)
                y2->v[i] = mean[i] - y2->v[i]; 
            out[i] = y1->v[i];
        }
        coef = -1.0;
    }
    else {
        // compute: mean + y1
        for (size_t i = pncol; i--; )
            out[i] = y1->v[i] + mean[i];
    }
    // solve for alpha: (omega_inv + phi * A_inv * phi^T) * alpha = b 
    // where b = t - (phi * y1 + y2) or b = phi * y1 + y2
    info = POSV(pnrow, 1, y2->cov, pnrow, y2->v, 1);
    // compute: (coef) * (A_inv * phi^T * alpha) + c, where c = y1 or (mean + y1) 
    if (!info && (a_type == IDENTITY)) {
        GEMV_T(pnrow, pncol, coef, phi, pncol, y2->v, 1, 1.0, out, 1);
    }
    else if (!info) {
        GEMV(pncol, pnrow, coef, x, pnrow, y2->v, 1, 1.0, out, 1);
    }

    free(x);
y2_failure_cleanup:
    mvn_output_free(y2);
y1_failure_cleanup:
    mvn_output_free(y1);
 
    return info;
}
