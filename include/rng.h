/* Copyright (c) 2020, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#ifndef HTNORM_RNG_H
#define HTNORM_RNG_H

#include <stdint.h>

typedef struct bitgen {
    void* base;
    uint64_t (*next_int)(void* base);
    double (*next_double)(void* base);
} rng_t;

void rng_free(rng_t* rng);


rng_t* rng_pcg64_new(void);
rng_t* rng_pcg64_new_seeded(uint64_t seed);

rng_t* rng_xrs128p_new(void);
rng_t* rng_xrs128p_new_seeded(uint64_t seed);

#endif
