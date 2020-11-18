/* Copyright (c) 2020, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#ifndef HTNORM_BLAS_H
#define HTNORM_BLAS_H

#include <cblas.h>
#include <lapacke.h>

/* CBLAS macros */
#define TRMV(n, a, lda, x, incx) \
    cblas_dtrmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, \
    (n), (a), (lda), (x), (incx))

#define TRMV_T(n, a, lda, x, incx) \
    cblas_dtrmv(CblasRowMajor, CblasUpper, CblasTrans, CblasNonUnit, \
    (n), (a), (lda), (x), (incx))

#define SYMV(n, alpha, a, lda, x, incx, beta, y, incy) \
    cblas_dsymv(CblasRowMajor, CblasUpper, (n), (alpha), (a), (lda), (x), \
    (incx), (beta), (y), (incy))

#define GEMV(m, n, alpha, a, lda, x, incx, beta, y, incy) \
    cblas_dgemv(CblasRowMajor, CblasNoTrans, (m), (n), (alpha), (a), (lda), \
    (x), (incx), (beta), (y), (incy))

#define GEMV_T(m, n, alpha, a, lda, x, incx, beta, y, incy) \
    cblas_dgemv(CblasRowMajor, CblasTrans, (m), (n), (alpha), (a), (lda), \
    (x), (incx), (beta), (y), (incy))

#define SYMM(m, n, alpha, a, lda, b, ldb, beta, c, ldc) \
    cblas_dsymm(CblasRowMajor, CblasRight, CblasUpper, (m), (n), (alpha), (a), \
    (lda), (b), (ldb), (beta), (c), (ldc))

#define GEMM(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) \
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (m), (n), (k), \
    (alpha), (a), (lda), (b), (ldb), (beta), (c), (ldc))

#define GEMM_NT(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) \
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, (m), (n), (k), \
    (alpha), (a), (lda), (b), (ldb), (beta), (c), (ldc))



/* LAPACKE macros */
#define GETRF(m, n, a, lda, ipiv) \
    LAPACKE_dgetrf(LAPACK_ROW_MAJOR, (m), (n), (a), (lda), (ipiv))

#define GETRS(n, nrhs, a, lda, ipiv, b, ldb) \
    LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', (n), (nrhs), (a), (lda), (ipiv), (b), (ldb))

#define POTRF(n, a, lda) \
    LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'U', (n), (a), (lda))

#define POTRS(n, nrhs, a, lda, b, ldb) \
    LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'U', (n), (nrhs), (a), (lda), (b), (ldb))

#define POTRI(n, a, lda) \
    LAPACKE_dpotri(LAPACK_ROW_MAJOR, 'U', (n), (a), (lda))

#endif
