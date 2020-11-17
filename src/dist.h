#ifndef HTNORM_DIST_H
#define HTNORM_DIST_H

#include <stdbool.h>
#include <stdlib.h>

#include "../include/rng.h"

#define HTNORM_ALLOC_ERROR -100


typedef struct mvn_output {
    double* v;
    double* cov;
} mvn_output_t;


mvn_output_t* mvn_output_new(size_t nrow);
void mvn_output_free(mvn_output_t* a);

int mv_normal_rand(rng_t* rng, const double* mean, const double* mat,
                   size_t nrow, bool diag, double* out);
int mv_normal_rand_prec(rng_t* rng, const double* prec, size_t nrow,
                        bool diag, mvn_output_t* out, bool full_inv);

#endif
