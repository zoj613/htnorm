/* Copyright (c) 2020, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 *
 * This module contains macros that are meant to make calls to thw BLAS and
 * LAPACK C API's easier. Notice that NaN-checking is turned off for LAPACK
 * routines for performance reasons. The checks can be turned on again using
 * compiler flags to undefine the `LAPACK_DISABLE_NAN_CHECK`.
 */
#ifndef HTNORM_BLAS_H
#define HTNORM_BLAS_H

#include <cblas.h>
// disable nan-checking in all lapacke routines
#define LAPACK_DISABLE_NAN_CHECK
#include <lapacke.h>

#ifdef HTNORM_COLMAJOR
    #define MatrixLayout CblasColMajor
    #define MatrixPart CblasLower
    #define LapackeLayout LAPACK_COL_MAJOR
    #define LapackePart 'L'
#else
    #define MatrixLayout CblasRowMajor
    #define MatrixPart CblasUpper
    #define LapackeLayout LAPACK_ROW_MAJOR
    #define LapackePart 'U'
#endif


/* CBLAS macros */
#define TRMV(n, a, lda, x, incx) \
    cblas_dtrmv(MatrixLayout, MatrixPart, CblasNoTrans, CblasNonUnit, \
    (n), (a), (lda), (x), (incx))

#define TRMV_T(n, a, lda, x, incx) \
    cblas_dtrmv(MatrixLayout, MatrixPart, CblasTrans, CblasNonUnit, \
    (n), (a), (lda), (x), (incx))

#define SYMV(n, alpha, a, lda, x, incx, beta, y, incy) \
    cblas_dsymv(MatrixLayout, MatrixPart, (n), (alpha), (a), (lda), (x), \
    (incx), (beta), (y), (incy))

#define GEMV(m, n, alpha, a, lda, x, incx, beta, y, incy) \
    cblas_dgemv(MatrixLayout, CblasNoTrans, (m), (n), (alpha), (a), (lda), \
    (x), (incx), (beta), (y), (incy))

#define GEMV_T(m, n, alpha, a, lda, x, incx, beta, y, incy) \
    cblas_dgemv(MatrixLayout, CblasTrans, (m), (n), (alpha), (a), (lda), \
    (x), (incx), (beta), (y), (incy))

#define GEMM(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) \
    cblas_dgemm(MatrixLayout, CblasNoTrans, CblasNoTrans, (m), (n), (k), \
    (alpha), (a), (lda), (b), (ldb), (beta), (c), (ldc))

#define GEMM_NT(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) \
    cblas_dgemm(MatrixLayout, CblasNoTrans, CblasTrans, (m), (n), (k), \
    (alpha), (a), (lda), (b), (ldb), (beta), (c), (ldc))

#define SYRK(n, k, alpha, a, lda, beta, c, ldc) \
    cblas_dsyrk(MatrixLayout, MatrixPart, CblasNoTrans, (n), (k), (alpha), \
    (a), (lda), (beta), (c), (ldc))


/* LAPACKE macros */
#define TRTRS(n, nrhs, a, lda, b, ldb) \
    LAPACKE_dtrtrs(LapackeLayout, LapackePart, 'N', 'N', (n), (nrhs), (a), (lda), (b), (ldb))

#define TRTRS_T(n, nrhs, a, lda, b, ldb) \
    LAPACKE_dtrtrs(LapackeLayout, LapackePart, 'T', 'N', (n), (nrhs), (a), (lda), (b), (ldb))

#define POTRF(n, a, lda) \
    LAPACKE_dpotrf(LapackeLayout, LapackePart, (n), (a), (lda))

#define POTRS(n, nrhs, a, lda, b, ldb) \
    LAPACKE_dpotrs(LapackeLayout, LapackePart, (n), (nrhs), (a), (lda), (b), (ldb))

#define POTRI(n, a, lda) \
    LAPACKE_dpotri(LapackeLayout, LapackePart, (n), (a), (lda))

#define POSV(n, nrhs, a, lda, b, ldb) \
    LAPACKE_dposv(LapackeLayout, LapackePart, (n), (nrhs), (a), (lda), (b), (ldb))

#endif
