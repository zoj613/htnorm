/* Copyright (c) 2020, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#ifndef HTNORM_HTNORM_H
#define HTNORM_HTNORM_H

#include <stdbool.h>
#include <stddef.h>

#include "rng.h"

typedef enum {NORMAL, DIAGONAL, IDENTITY} type_t;

typedef struct {
    size_t gnrow;
    size_t gncol;
    bool diag;
} ht_config_t;

#define INIT_HP_CONFIG(x) (x) = {.diag = false}

typedef struct {
    // whether the mean is structure (i.e. made up of the covariance matrix)
    bool struct_mean; 
    // Whether matrix A is regular (0), diagonal (1) or identity (2)
    type_t a_id;
    // Whether matrix A is regular (0), diagonal (1) or identity (2)
    type_t o_id;
    // number of rows of phi matrix
    size_t pnrow;
    // number of columns of phi matrix
    size_t pncol;
} sp_config_t;

#define INIT_SP_CONFIG(x) \
    (x) = {.struct_mean = false, .a_id = NORMAL, .o_id = NORMAL} 


int htn_hyperplane_truncated_mvn(rng_t* rng, const ht_config_t* conf,
                                 const double* mean, const double* cov,
                                 const double* g, const double* r, double* out);
int htn_structured_precision_mvn(rng_t* rng, const sp_config_t* conf,
                                 const double* mean, const double* a,
                                 const double* phi, const double* omega,
                                 double* out);

#endif
