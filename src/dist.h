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

#include <stdbool.h>
#include <stdlib.h>

#include "../include/rng.h"
#include "../include/htnorm.h"

// error number for failed memory allocation throughout the library
#define HTNORM_ALLOC_ERROR -100

/* Stores the output of sampling from a MVN using the precision matrix.
 * `v` is the sampled vector and `cov` is the computed inverse of the precision */
typedef struct mvn_output {
    double* v;
    double* cov;
} mvn_output_t;

/* Get an instance of the `mvn_output_t` struct pointer whose elements have
 * dimension `nrow`. The members are allocated memeory on the heap, and thus
 * need to free'd using `mvn_output_free` when no longer needed.*/
mvn_output_t* mvn_output_new(size_t nrow);
void mvn_output_free(mvn_output_t* a);

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
int mvn_rand_cov(rng_t* rng, const double* mean, const double* mat, size_t nrow,
                 bool diag, double* out);

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
 *  type:
 *      Whether the precision is NORMAL, DIAGONAL or IDENTITY.
 *  out:
 *      The array to store the generated sample.
 *
 *  Returns
 *  -------
 *  And integer to indicate whether the function completed successfully. A
 *  value of zero means the sampling was successful, else it failed.
 */
int mvn_rand_prec(rng_t* rng, const double* prec, size_t nrow, type_t type,
                  mvn_output_t* out, bool full_inv);

#endif
