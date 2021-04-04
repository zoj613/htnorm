/* Copyright (c) 2020, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause 
 *
 * This header contains declarations for helper functions that aid in
 * generating samples from the multivariate normal disribution, given either
 * a covariance or a precision matrix.
 */
#ifndef HTNORM_DIST_H
#define HTNORM_DIST_H

#include <stdlib.h>

#include "../include/htnorm_rng.h"
#include "../include/htnorm.h"
#include "htnorm_always_inline.h"

// error number for failed memory allocation throughout the library
#define HTNORM_ALLOC_ERROR -100

/* Stores the output of sampling from a MVN using the precision matrix.
 * `v` is the sampled vector and `factor` is the computed cholesky factor of 
 * the precision */
typedef struct {
    double* restrict v;
    double* restrict factor;
} mvn_output_t;

/* Get an instance of the `mvn_output_t` struct pointer whose elements have
 * dimension `nrow`. The members are allocated memeory on the heap, and thus
 * need to free'd using `mvn_output_free` when no longer needed.*/
static ALWAYS_INLINE(mvn_output_t*)
mvn_output_new(int nrow, type_t factor_type)
{
    mvn_output_t* out = malloc(sizeof(mvn_output_t));
    if (out != NULL) {
        out->v = malloc(nrow * sizeof(*out->v));
        // allocate space needed based on factor matrix type
        switch(factor_type) {
            case DIAGONAL:
            case IDENTITY:
                out->factor = malloc(nrow * sizeof(*out->factor));
                break;
            default:
                out->factor = calloc(nrow * nrow, sizeof(*out->factor));
        }
    }
    return out;
}


static ALWAYS_INLINE(void)
mvn_output_free(mvn_output_t* a)
{
    free(a->factor);
    free(a->v);
    free(a);
}

/* Generate a vector from a MVN of specied mean and covariance.
 *
 * Paramaters
 * ----------
 *  rng:
 *      an instance of a random number generator.
 *  mean:
 *      The mean of the distribution.
 *  mat:
 *      The covariance matrix.
 *  nrow:
 *      The dimension of the matrix.
 *  diag:
 *      Whether the covariance is diagonal. When set to True, an optimized
 *      algorithm is used to generate the samples faster.
 *  out:
 *      The array to store the generated sample
 *
 *  Returns
 *  -------
 *  And integer to indicate whether the function completed successfully. A
 *  value of zero means the sampling was successful, else it failed.
 */
int mvn_rand_cov(rng_t* rng, const double* mean, const double* mat, int nrow,
                 bool diag, double* restrict out);

/* Generate a vector from a MVN of zero mean and a specified precision matrix.
 *
 * Paramaters
 * ----------
 *  rng:
 *      an instance of a random number generator.
 *  prec:
 *      The precision matrix.
 *  nrow:
 *      The dimension of the matrix.
 *  prec_type:
 *      Whether the precision is NORMAL, DIAGONAL or IDENTITY.
 *  out:
 *      The array to store the generated sample.
 *
 *  Returns
 *  -------
 *  And integer to indicate whether the function completed successfully. A
 *  value of zero means the sampling was successful, else it failed.
 */
int mvn_rand_prec(rng_t* rng, const double* prec, int nrow, type_t prec_type,
                  mvn_output_t* out);

#endif
