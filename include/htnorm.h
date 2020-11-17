/* Copyright (c) 2020, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#ifndef HTNORM_HTNORM_H
#define HTNORM_HTNORM_H

#include <stdbool.h>

#include "../src/dist.h"

typedef struct matrix {
    int nrow;
    int ncol;
    double* mat;
} matrix_t;

int htnorm_rand(rng_t* rng, const double* mean, const matrix_t* cov, bool diag,
                const matrix_t* g, const double* r, double* out);
int htnorm_rand2(rng_t* rng, const double* mean, const matrix_t* a, bool a_diag,
                 const matrix_t* phi, const matrix_t* omega, bool o_diag,
                 double* out);

#endif
