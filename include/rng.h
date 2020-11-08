#ifndef HTNORM_RNG_H
#define HTNORM_RNG_H

#include "../src/pcg.h"


typedef pcg32_random_t rng_t;

rng_t* rng_new(void);
rng_t* rng_new_seeded(uint64_t state, uint64_t seq);

#endif
