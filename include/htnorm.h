/* Copyright (c) 2020, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#ifndef HTNORM_HTNORM_H
#define HTNORM_HTNORM_H

#include <stdbool.h>
#include <stddef.h>

#include "rng.h"

#define INIT_HT_CONFIG(x) (x) = {.diag = false}

#define INIT_SP_CONFIG(x) \
    (x) = {.struct_mean = false, .a_id = NORMAL, .o_id = NORMAL} 

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

typedef struct {
    // Whether matrix A is regular (0), diagonal (1) or identity (2)
    type_t a_id;
    // Whether matrix A is regular (0), diagonal (1) or identity (2)
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


int htn_hyperplane_truncated_mvn(rng_t* rng, const ht_config_t* conf, double* out);
int htn_structured_precision_mvn(rng_t* rng, const sp_config_t* conf, double* out);

#endif
