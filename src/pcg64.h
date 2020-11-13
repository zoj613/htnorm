#ifndef HTNORM_PCG64_H
#define HTNORM_PCG64_H

#include <stdint.h>
#include "pcg32_minimal.h"

typedef struct pcg64_rand{
    pcg32_random_t gen[2];
    uint64_t (*next_int)(void* pcg64);
    double (*next_double)(void* pcg64);
} pcg64_random_t;

pcg64_random_t* pcg64_init(uint64_t seed);

#endif
