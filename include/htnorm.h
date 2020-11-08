#ifndef HTNORM_H
#define HTNORM_H

#include <stdbool.h>

#include "rng.h"


typedef struct matrix {
    int nrow;
    int ncol;
    double* mat;
} matrix_t;


int htnorm_rand(rng_t* rng, const double* mean, const matrix_t* cov, bool diag,
                const matrix_t* g, const double* r, double* out);
int htnorm_rand2(rng_t* rng, const double* mean, const matrix_t* a, bool a_diag,
                 const matrix_t* phi, const matrix_t* omega, bool o_diag, double* out);

#endif
