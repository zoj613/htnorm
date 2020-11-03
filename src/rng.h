#ifndef HTNORM_RNG_H
#define HTNORM_RNG_H

#include "pcg.h"

typedef pcg32_random_t rng_t;

rng_t* rng_new(void);
rng_t* rng_new_seeded(uint64_t state, uint64_t seq);
inline double std_uniform_rand(rng_t* rng_ptr);
inline double uniform_rand(rng_t* rng_ptr, const double lo, const double up);
double std_normal_rand(rng_t* rng_ptr);

#endif
