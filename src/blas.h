/* Copyright (c) 2020, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */
#ifndef HTNORM_BLAS_H
#define HTNORM_BLAS_H

// blas
extern double ddot_(const int* n, const double* x, const int* incx,
                    const double* y, const int* incy);
extern void daxpy_(const int* n, const double* a, const double* x,
                   const int* incx, double* restrict y, const int* incy);
extern void dtrmv_(const char* uplo, const char* trans, const char* diag,
                   const int* n, const double* a, const int* lda,
                   double* restrict x, const int* incx);
extern void dtrsv_(const char* uplo, const char* trans, const char* diag,
                   const int* n, const double* a, const int* lda,
                   double* restrict x, const int* incx);
extern void dsymv_(const char* uplo, const int* n, const double* alpha,
                   const double* a, const int* lda, const double* x,
                   const int* incx, const double* beta, double* restrict y,
                   const int* incy);
extern void dgemv_(const char* trans, const int* m, const int* n,
                   const double* alpha, const double* a, const int* lda,
                   double* x, const int* incx, const double* beta,
                   double* restrict y, const int* incy);
extern void dgemm_(const char* transa, const char* transb, const int* m,
                   const int* n, const int* k, const double* alpha,
                   const double* a, const int* lda, const double* b,
                   const int* ldb, const double* beta, double* restrict c,
                   const int* ldc);
extern void dsymm_(const char* side, const char* uplo, const int* m, const int* n,
                   const double* alpha, const double* a, const int* lda,
                   const double* b, const int* ldb, const double* beta,
                   double* restrict c, const int* ldc);
extern void dsyrk_(const char* uplo, const char* trans, const int* n, const int* k,
                   const double* alpha, const double* a, const int* lda,
                   const double* beta, double* restrict c, const int* ldc);

// lapack
extern void dpotrf_(const char* uplo, const int* n, double* restrict a,
                    const int* lda, int* info);
extern void dpotrs_(const char* uplo, const int* n, const int* nrhs, double* a,
                    const int* lda, double* restrict b, const int* ldb, int* info);
extern void dpotri_(const char* uplo, const int* n, double* restrict a,
                    const int* lda, int* info);
extern void dposv_(const char* uplo, const int* n, const int* nrhs,
                   const double* a, const int* lda, double* restrict b,
                   const int* ldb, int* info);


static const char L = 'L';  // lower triangual part
static const char T = 'T';  // Transpose
static const char N = 'N';  // Non-Transpose / Non-Unit
static const char R = 'R';  // right side
static const int inc = 1;  // element increment


/* CBLAS macros */
#define DOT(n, x, y) \
    ddot_(&(n), (x), &(inc), (y), &(inc))

#define AXPY(n, a, x, y) \
    daxpy_(&(n), &(a), (x), &(inc), (y), &(inc))

#define TRMV(n, a, lda, x, incx) \
    dtrmv_(&(L), &(N), &(N), &(n), (a), &(lda), x, &(incx))

#define SYMV(n, alpha, a, lda, x, beta, y) \
    dsymv_(&(L), &(n), &(alpha), (a), &(lda), x, &(inc), &(beta), y, &(inc))

#define GEMV(m, n, alpha, a, lda, x, beta, y) \
    dgemv_(&(N), &(m), &(n), &(alpha), (a), &(lda), (x), &(inc), &(beta), \
    (y), &(inc))

#define GEMV_T(m, n, alpha, a, lda, x, beta, y) \
    dgemv_(&(T), &(m), &(n), &(alpha), (a), &(lda), (x), &(inc), &(beta), \
    (y), &(inc))

#define SYMM(m, n, alpha, a, lda, b, ldb, beta, c, ldc) \
    dsymm_(&(L), &(L), &(m), &(n), &(alpha), (a), &(lda), (b), &(ldb), \
    &(beta), (c), &(ldc))

#define SYMM_R(m, n, alpha, a, lda, b, ldb, beta, c, ldc) \
    dsymm_(&(R), &(L), &(m), &(n), &(alpha), (a), &(lda), (b), &(ldb), \
    &(beta), (c), &(ldc))

#define GEMM(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) \
    dgemm_(&(N), &(N), &(m), &(n), &(k), &(alpha), (a), &(lda), (b), &(ldb), \
    &(beta), (c), &(ldc))

#define GEMM_NT(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) \
    dgemm_(&(N), &(T), &(m), &(n), &(k), &(alpha), (a), &(lda), (b), &(ldb), \
    &(beta), (c), &(ldc))

#define GEMM_TN(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) \
    dgemm_(&(T), &(N), &(m), &(n), &(k), &(alpha), (a), &(lda), (b), &(ldb), \
    &(beta), (c), &(ldc))

#define SYRK(n, k, alpha, a, lda, beta, c, ldc) \
    dsyrk_(&(L), &(N), &(n), &(k), &(alpha), (a), &(lda), &(beta), (c), &(ldc))

#define SYRK_T(n, k, alpha, a, lda, beta, c, ldc) \
    dsyrk_(&(L), &(T), &(n), &(k), &(alpha), (a), &(lda), &(beta), (c), &(ldc))

#define TRSV(n, a, lda, x, incx) \
    dtrsv_(&(L), &(T), &(N), &(n), (a), &(lda), (x), &(incx))


/* LAPACKE macros */
#define POTRF(n, a, lda, info) \
    dpotrf_(&(L), &(n), (a), &(lda), &(info))

#define POTRS(n, nrhs, a, lda, b, ldb, info) \
    dpotrs_(&(L), &(n), &(nrhs), (a), &(lda), (b), &(ldb), &(info))

#define POTRI(n, a, lda, info) \
    dpotri_(&(L), &(n), (a), &(lda), &(info))

#define POSV(n, nrhs, a, lda, b, ldb, info) \
    dposv_(&(L), &(n), &(nrhs), (a), &(lda), (b), &(ldb), &(info))

#endif
