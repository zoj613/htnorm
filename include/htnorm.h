/* Copyright (c) 2020, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause 
 *
 * Fast Simulation of Hyperplane-Truncated Multivaariate Normal Distributions.
 *
 * This library implements the algorithms 2, 4 and the one in example 4 of
 * Cong, Y., Chen, B., & Zhou, M. (2017).
 *
 * Algorithm 2 allows fast and exact sampling from a multivariate normal (MVN)
 * distribution that is trancated on a hyperplane.
 *
 * Algorthm 4 allows efficient sampling from a MVN with a structured precision
 * matrix.
 *
 * Algorithm described in example 4 allows efficient sampling from a MVN with
 * a structured precision matrix and a structured mean dependent on the
 * structure of the precision matrix.
 *
 * References:
 * 1) Cong, Y., Chen, B., & Zhou, M. (2017). Fast simulation of
 *    hyperplane-truncated multivariate normal distributions. Bayesian Analysis,
 *    12(4), 1017-1037.
 */
#ifndef HTNORM_HTNORM_H
#define HTNORM_HTNORM_H

#include <stdbool.h>
#include <stddef.h>

#include "htnorm_rng.h"


typedef enum {NORMAL, DIAGONAL, IDENTITY} type_t;

typedef struct {
    // number of rows of the G matrix
    size_t gnrow;
    // number of columns of the G matrix
    size_t gncol;
    // mean vector
    const double* mean;
    // covariance
    const double* cov;
    // G matrix with LHS of the hyperplane
    const double* g;
    // r vector with the RHS of the equation: Gx = r
    const double* r;
    // whether the covariance is diagonal;
    bool diag;
} ht_config_t;

// helper function to initialize values of the ht_config_t struct
void init_ht_config(ht_config_t* conf, size_t gnrow, size_t gncol,
                    const double* mean, const double* cov, const double* g,
                    const double* r, bool diag);

typedef struct {
    // Whether matrix A is regular (0), diagonal (1) or identity (2)
    type_t a_id;
    // Whether matrix Omega is regular (0), diagonal (1) or identity (2)
    type_t o_id;
    // number of rows of phi matrix
    size_t pnrow;
    // number of columns of phi matrix
    size_t pncol;
    // mean vector
    const double* mean;
    // array with elements of the A matrix
    const double* a;
    // array with elements of the phi matrix
    const double* phi;
    // array with elements of the omega matrix
    const double* omega;
    // whether the mean is structure (i.e. made up of the covariance matrix)
    bool struct_mean; 
} sp_config_t;

// helper function to initialize values of the sp_config_t struct
void init_sp_config(sp_config_t* conf, size_t pnrow, size_t pncol, const double* mean,
                    const double* a, const double* phi, const double* omega,
                    bool struct_mean, type_t a_id, type_t o_id);

/* Sample from a multivariate normal distribution truncated on a hyperplane.
 *
 * Generate sample x from the distribution: N(mean, cov) truncated on the plane
 * {x | Gx = r}, where the rank of G is less than that of the covariance.
 *
 * Paramaters
 * ----------
 *  rng:
 *      A pointer to the `rng_t` struct (The random number generator).
 *  conf:
 *      A pointer to the input configuration struct `ht_config_t`.
 *  out:
 *      An array to store the generated samples.
 *
 *  Returns
 *  -------
 *  Zero if the sampling was successful. A positive integer is returned if the
 *  sampled failed because the covariance is not positive definite or if a
 *  factorization of the covariance was not successful. A negative integer is
 *  returned if one of the inputs contains an illegal value (non-numerical/NaN).
 */
int htn_hyperplane_truncated_mvn(rng_t* rng, const ht_config_t* conf, double* restrict out);


/* Sample from a MVN with a structured precision matrix and/or structured mean.
 *
 * Sample from a MVN: N(mean, (A + phi^T * Omega * phi)^-1) or
 * N((A + phi^T * Omega * phi)^-1 * phi^T * t, (A + phi^T * Omega * phi)^-1))
 * This algorithm in very efficient and ideal when the dimension of matrix A
 * is greater than that of matrix Omega.
 *
 * Parameters
 * ----------
 *  rng:
 *      A pointer to the `rng_t` struct (The random number generator).
 *  conf:
 *      A pointer to the input configuration struct `sp_config_t`.
 *  out:
 *      An array to store the generated samples.
 *
 *  Returns
 *  -------
 *  Zero if the sampling was successful. A positive integer is returned if the
 *  sampled failed because the covariance is not positive definite or if a
 *  factorization of the covariance was not successful. A negative integer is
 *  returned if one of the inputs contains an illegal value (non-numerical/NaN).
 * */
int htn_structured_precision_mvn(rng_t* rng, const sp_config_t* conf, double* restrict out);

#endif
