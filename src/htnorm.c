/* Copyright (c) 2020, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * the HTNORM_COLMAJOR preprocessor macro is used for calling this library in
 * languages that store array data in column-major format (e.g. Fortran, R,
 * Julia). At the expence of readability, it is necessary to have to add this
 * macro for several lines per function because matrix manipulation with BLAS
 * / LAPACK in C heavily depends on the memory layout of the input.
 * */
#include <string.h>

#include "htnorm_blas.h"
#include "htnorm_distributions.h"


ALWAYS_INLINE(void)
init_ht_config(ht_config_t* conf, size_t gnrow, size_t gncol, const double* mean,
               const double* cov, const double* g, const double* r, bool diag)
{
    conf->gnrow = gnrow;
    conf->gncol = gncol;
    conf->mean = mean;
    conf->cov = cov;
    conf->g = g;
    conf->r = r;
    conf->diag = diag;
}


ALWAYS_INLINE(void)
init_sp_config(sp_config_t* conf, size_t pnrow, size_t pncol, const double* mean,
               const double* a, const double* phi, const double* omega,
               bool struct_mean, type_t a_id, type_t o_id)
{
    conf->pnrow = pnrow;
    conf->pncol = pncol;
    conf->mean = mean;
    conf->a = a;
    conf->phi = phi;
    conf->omega = omega;
    conf->struct_mean = struct_mean;
    conf->a_id = a_id;
    conf->o_id = o_id;
}


// special case for when g matrix has dimensions 1 by n (a 1d array)
static ALWAYS_INLINE(int)
hyperplane_truncated_norm_1d_g(const ht_config_t* conf, double* restrict out)
{
    static const double a = 1.0, b = 0.0;

    const int ncol = conf->gncol;
    const double* const cov = conf->cov;
    const double* const g = conf->g;

    double alpha = 0;

    double* cov_g = malloc(ncol * sizeof(*cov_g));
    if (cov_g == NULL)
        return HTNORM_ALLOC_ERROR;

    // compute: r - g * y; where y ~ N(mean, cov)
    alpha = *conf->r - DOT(ncol, g, out);

    // compute: cov * g^T
    if (conf->diag) {
        for (size_t i = ncol; i--; )
            cov_g[i] = cov[ncol * i + i] * g[i];
    }
    else {
        SYMV(ncol, a, cov, ncol, g, b, cov_g);
    }

    // out = y + cov * g^T * alpha, where alpha = (r - g * y) / (g * cov * g^T)
    alpha /= DOT(ncol, g, cov_g);
    AXPY(ncol, alpha, cov_g, out);

    free(cov_g);
    return 0;
}


int
htn_hyperplane_truncated_mvn(rng_t* rng, const ht_config_t* conf, double* restrict out)
{
    static const int one = 1;

    const int gncol = conf->gncol;  // equal to the dimension of the covariance
    const int gnrow = conf->gnrow;
    const bool diag = conf->diag;
    const double* mean = conf->mean;
    const double* cov = conf->cov;
    const double* g = conf->g;
    const double* r = conf->r;
    
    // blas/lapack matrix coefficient scalars used in several routines
    double a = -1, b = 1;
    int info;


    if ((info = mvn_rand_cov(rng, mean, cov, gncol, diag, out)))
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


    memcpy(gy, r, gnrow * sizeof(*gy)); 
#ifdef HTNORM_COLMAJOR
    // compute: r - g*y
    GEMV(gnrow, gncol, a, g, gnrow, out, b, gy);
#else
    // compute: r - y^T * g^T
    GEMM(one, gnrow, gncol, a, out, one, g, gncol, b, gy, one);
#endif


    b = 0;
    a = 1;
    if (diag) {
        for (int i = 0; i < gnrow; i++)
            for (int j = 0; j < gncol; j++)
#ifdef HTNORM_COLMAJOR
                // compute: g * cov
                cov_g[gnrow * j + i] = cov[gncol * j + j] * g[gnrow * j + i];
#else
                // compute: cov * g^T
                cov_g[gncol * i + j] = cov[gncol * j + j] * g[gncol * i + j];
#endif
    }
    else {
#ifdef HTNORM_COLMAJOR
        // compute: g * cov
        SYMM_R(gnrow, gncol, a, cov, gncol, g, gnrow, b, cov_g, gnrow);
#else
        SYMM(gncol, gnrow, a, cov, gncol, g, gncol, b, cov_g, gncol); 
#endif
    }


#ifdef HTNORM_COLMAJOR
    // g * cov * g^T == (g * cov) * g^T
    GEMM_NT(gnrow, gnrow, gncol, a, cov_g, gnrow, g, gnrow, b, g_cov_g, gnrow);
#else
    // g * cov * g^T == (cov * g^T)^T * g^T
    GEMM_TN(gnrow, gnrow, gncol, a, cov_g, gncol, g, gncol, b, g_cov_g, gnrow);
#endif


    // solve: g * cov * g^T * alpha = r - g*y
    POSV(gnrow, one, g_cov_g, gnrow, gy, gnrow, info);
    if (!info)
#ifdef HTNORM_COLMAJOR
        // out = (g * cov)^T * alpha + out
        GEMV_T(gnrow, gncol, a, cov_g, gnrow, gy, a, out);
#else
        // out = (cov * g^T) * alpha + out
        GEMV(gncol, gnrow, a, cov_g, gncol, gy, a, out);
#endif


    free(g_cov_g); 
gcovg_failure_cleanup:
    free(cov_g);
covg_failure_cleanup:
    free(gy);

    return info;
} 


static ALWAYS_INLINE(int)
compute_precision_inverse(type_t mat_type, int n, mvn_output_t* out)
{
    switch (mat_type) {
        case IDENTITY:
        case DIAGONAL:
            // convert `factor` from a 1d array into a diagonal nxn matrix
            // for later use with LAPACK/BLAS routines. `factor` here is the
            // precision matrix diagonal entries stored in a 1d array.
            out->factor = realloc(out->factor, n * n * sizeof(*out->factor));
            if (out->factor == NULL)
                return HTNORM_ALLOC_ERROR;
            memset(out->factor + n, 0, n * (n - 1) * sizeof(*out->factor)); 
            for (size_t i = n; i--; )
                out->factor[n * i + i] = 1 / out->factor[i];
            // set all the old values in first row to 0 except for the first
            memset(out->factor + 1, 0, (n - 1) * sizeof(*out->factor));
            return 0;
        default: {
            int info;
            POTRI(n, out->factor, n, info);
            return info;
        }
    }
}


int
htn_structured_precision_mvn(rng_t* rng, const sp_config_t* conf, double* restrict out)
{
    static const int one = 1;

    const int pnrow = conf->pnrow;
    const int pncol = conf->pncol;
    const type_t a_type = conf->a_id;
    const type_t o_type = conf->o_id;
    const double* mean = conf->mean;
    const double* phi = conf->phi;

    double a = 1, b = 1;
    int i, info;


    mvn_output_t* y1 = mvn_output_new(pncol, a_type);
    if (y1 == NULL || y1->v == NULL || y1->factor == NULL) {
        info = HTNORM_ALLOC_ERROR;
        goto y1_failure_cleanup;
    }

    mvn_output_t* y2 = mvn_output_new(pnrow, o_type);
    if (y2 == NULL || y2->v == NULL || y2->factor == NULL) {
        info = HTNORM_ALLOC_ERROR;
        goto y2_failure_cleanup;
    }

    if ((info = mvn_rand_prec(rng, conf->a, pncol, a_type , y1)) ||
        (info = mvn_rand_prec(rng, conf->omega, pnrow, conf->o_id, y2)))
        goto y2_failure_cleanup;

    double* x = malloc(pnrow * pncol * sizeof(*x));
    if (x == NULL) {
        info = HTNORM_ALLOC_ERROR;
        goto y2_failure_cleanup;
    }

    // omega inverse
    if ((info = compute_precision_inverse(o_type, pnrow, y2)))
        goto inverse_fail_cleanup;


    // compute: (x = A_inv * phi^T) and (phi * A_inv * phi^T + omega_inv)
    switch (a_type) {
        case IDENTITY:
#ifdef HTNORM_COLMAJOR
            SYRK(pnrow, pncol, a, phi, pnrow, b, y2->factor, pnrow);
#else
            // (phi^T)^T * (phi^T) + omega_inv
            SYRK_T(pnrow, pncol, a, phi, pncol, b, y2->factor, pnrow);
#endif
            break;

        case DIAGONAL:
            for (i = 0; i < pnrow; i++)
                for (int j = 0; j < pncol; j++)
#ifdef HTNORM_COLMAJOR
                    // compute: phi * A_inv
                    x[pnrow * j + i] = phi[pnrow * j + i] / y1->factor[j];
#else
                    // compute: A_inv * phi^T
                    x[pncol * i + j] = phi[pncol * i + j] / y1->factor[j];
#endif

#ifdef HTNORM_COLMAJOR
            GEMM_NT(pnrow, pnrow, pncol, a, x, pnrow, phi, pnrow, b, y2->factor, pnrow);
#else
            GEMM_TN(pnrow, pnrow, pncol, a, x, pncol, phi, pncol, b, y2->factor, pnrow);
#endif
            break;

        default:
            if ((info = compute_precision_inverse(a_type, pncol, y1)))
                goto inverse_fail_cleanup;
            b = 0;
#ifdef HTNORM_COLMAJOR
            SYMM_R(pnrow, pncol, a, y1->factor, pncol, phi, pnrow, b, x, pnrow);
            b = 1;
            GEMM_NT(pnrow, pnrow, pncol, a, x, pnrow, phi, pnrow, b, y2->factor, pnrow);
#else
            SYMM(pncol, pnrow, a, y1->factor, pncol, phi, pncol, b, x, pncol); 
            b = 1;
            GEMM_TN(pnrow, pnrow, pncol, a, x, pncol, phi, pncol, b, y2->factor, pnrow);
#endif
    }


#ifdef HTNORM_COLMAJOR
    // compute: phi * y1 + y2
    GEMV(pnrow, pncol, a, phi, pnrow, y1->v, b, y2->v);
#else
    // compute: y1^T * phi^T + y2
    GEMM(one, pnrow, pncol, a, y1->v, one, phi, pncol, b, y2->v, one);
#endif


    if (conf->struct_mean) {
        // compute: t - (phi * y1 + y2)
        for (i = pnrow; i--; )
            y2->v[i] = mean[i] - y2->v[i]; 
        for (i = pncol; i--; )
            out[i] = y1->v[i];
        a = -1.0;
    }
    else {
        // compute: mean + y1
        for (i = pncol; i--; )
            out[i] = y1->v[i] + mean[i];
    }

    // solve for alpha: (omega_inv + phi * A_inv * phi^T) * alpha = b 
    // where b = t - (phi * y1 + y2) or b = phi * y1 + y2
    POSV(pnrow, one, y2->factor, pnrow, y2->v, pnrow, info);
    // compute: (coef) * (A_inv * phi^T * alpha) + c, where c = y1 or (mean + y1) 
    if (!info && (a_type == IDENTITY)) {
#ifdef HTNORM_COLMAJOR
        GEMV_T(pnrow, pncol, a, phi, pnrow, y2->v, b, out);
#else
        GEMM(pncol, one, pnrow, a, phi, pncol, y2->v, pnrow, b, out, pncol);
#endif
    }
    else if (!info) {
#ifdef HTNORM_COLMAJOR
        GEMV_T(pnrow, pncol, a, x, pnrow, y2->v, b, out);
#else
        GEMV(pncol, pnrow, a, x, pncol, y2->v, b, out);
#endif
    }


inverse_fail_cleanup:
    free(x);
y2_failure_cleanup:
    mvn_output_free(y2);
y1_failure_cleanup:
    mvn_output_free(y1);
 
    return info;
}
