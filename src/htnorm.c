#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <cblas.h>
#include <lapacke.h>

#include "dist.h"
#include "../include/htnorm.h"


// special case for when g matrix has dimensions 1 by n (a 1d array)
static int
htnorm_rand_g_a_vec(const matrix_t* cov, bool diag, const double* g,
                    double r, double* out)
{
    const int cncol = cov->ncol;

    // compute: r - g * y; where y ~ N(mean, cov)
    double alpha = r - cblas_ddot(cncol, g, 1, out, 1);

    // compute: cov * g^T
    double* cov_g = malloc(cncol * sizeof(double));
    if (cov_g == NULL)
        return HTNORM_ALLOC_ERROR;

    // optimize when cov if diagonal
    if (diag) {
        for (int i = 0; i < cncol; i++)
            cov_g[i] = cov->mat[cncol * i + i] * g[i];
    }
    else {
        cblas_dsymv(CblasRowMajor, CblasLower, cncol, 1.0,
                    cov->mat, cncol, g, 1, 0.0, cov_g, 1);
    }

    alpha /= cblas_ddot(cncol, g, 1, cov_g, 1);
    
    cblas_daxpy(cncol, alpha, cov_g, 1, out, 1);

    free(cov_g);
    return 0;
}


int
htnorm_rand(rng_t* rng, const double* mean, const matrix_t* cov,
            bool diag, const matrix_t* g, const double* r, double* out)
{
    const int cnrow = cov->nrow;
    const int gnrow = g->nrow;

    lapack_int info = mv_normal_rand(rng, mean, cov->mat, cnrow, diag, out);
    // early return upon failure
    if (info)
        return info;

    // check if g's number of rows is 1 and use an optimized function
    if (gnrow == 1)
        return htnorm_rand_g_a_vec(cov, diag, g->mat, *r, out);

    double* gy = malloc(gnrow * sizeof(double));
    if (gy == NULL)
        return HTNORM_ALLOC_ERROR;

    double* cov_g = malloc(gnrow * cnrow * sizeof(double));
    if (cov_g == NULL) {
        info = HTNORM_ALLOC_ERROR;
        goto covg_failure_cleanup;
    }

    double* g_cov_g = malloc(gnrow * gnrow * sizeof(double));
    if (g_cov_g == NULL) {
        info = HTNORM_ALLOC_ERROR;
        goto gcovg_failure_cleanup;
    }

    lapack_int* ipiv = malloc(gnrow * sizeof(lapack_int));
    if (ipiv == NULL) {
        info = HTNORM_ALLOC_ERROR;
        goto ipiv_failure_cleanup;
    }

    // compute: r - g*y
    cblas_dcopy(gnrow, r, 1, gy, 1);
    cblas_dgemv(CblasRowMajor, CblasNoTrans, gnrow, cnrow,
                -1.0, g->mat, gnrow, out, 1, 1.0, gy, 1);

    // TODO: add code for case when cov is diagonal?
    // compute: g * cov
    cblas_dsymm(CblasRowMajor, CblasRight, CblasUpper, gnrow, cnrow,
                1.0, cov->mat, cnrow, g->mat, gnrow, 0.0, cov_g, gnrow); 

    // compute: g * cov * g^T
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, gnrow, gnrow, cnrow,
                1.0, cov_g, cnrow, g->mat, cnrow, 0.0, g_cov_g, gnrow);

    // compute LU factorization to get ipiv (pivoting indices)
    info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, gnrow, gnrow, g_cov_g, gnrow, ipiv);
    if (!info) {                      
        // solve a system of linear equations: g * cov * g^T * alpha = r - g*y
        // value of alpha is store in `gy` array.
        info = LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', gnrow, 1,
                              g_cov_g, gnrow, ipiv, gy, gnrow);
        if (!info) { 
            // compute: out = cov * g^T * alpha + out
            cblas_dgemv(CblasRowMajor, CblasTrans, gnrow, cnrow,
                        1.0, cov_g, cnrow, gy, 1, 1.0, out, 1);
        }
    }

    free(ipiv);
ipiv_failure_cleanup:
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
    const int pnrow = phi->nrow;
    const int pncol = phi->ncol;
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

    if((info = mv_normal_rand_prec(rng, a->mat, pncol, a_diag, y1)) ||
        (info = mv_normal_rand_prec(rng, omega->mat, pnrow, o_diag, y2)))
        goto y2_failure_cleanup;

    double* x = malloc(pnrow * pncol * sizeof(double));
    if (x == NULL) {
        info = HTNORM_ALLOC_ERROR;
        goto y2_failure_cleanup;
    }

    lapack_int* ipiv = malloc(pnrow * sizeof(lapack_int));
    if (ipiv == NULL) {
        info = HTNORM_ALLOC_ERROR;
        goto ipiv_failure_cleanup;
    }

    // compute: x = phi * A_inv
    cblas_dsymm(CblasRowMajor, CblasRight, CblasUpper, pnrow, pncol,
                1.0, y1->cov, pncol, pmat, pnrow, 0.0, x, pnrow); 

    // compute: phi * A_inv * phi^T + omega_inv
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, pnrow, pnrow, pncol,
                1.0, x, pncol, pmat, pncol, 1.0, y2->cov, pnrow);

    // compute: phi * y1 + y2
    cblas_dgemv(CblasRowMajor, CblasNoTrans, pnrow, pncol,
                1.0, pmat, pncol, y1->v, 1, 1.0, y2->v, 1);

    // compute: mean + y1
    cblas_daxpy(pncol, 1.0, mean, 1, y1->v, 1);

    // solve for alpha: (omega_inv + phi * A_inv * phi^T) * alpha = phi * y1 + y2
    // compute LU factorization to get ipiv (pivoting indices)
    info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, pnrow, pnrow, y2->cov, pnrow, ipiv);
    if (!info) {
        info = LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', pnrow, 1, y2->cov,
                              pnrow, ipiv, y2->v, pnrow);
        if (!info) {
            // compute: A_inv * phi^T * alpha + mean + y1
            cblas_dgemv(CblasRowMajor, CblasTrans, pnrow, pncol,
                        -1.0, x, pncol, y2->v, 1, 1.0, y1->v, 1);

            cblas_dcopy(pnrow, y1->v, 1, out, 1);
        }
    } 

    free(ipiv);
ipiv_failure_cleanup:
    free(x);
y2_failure_cleanup:
    mvn_output_free(y2);
y1_failure_cleanup:
    mvn_output_free(y1);
 
    return info;
}
