/* Copyright (c) 2020, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#ifndef HTNORM_PCG64_H
#define HTNORM_PCG64_H

#include "htnorm_pcg32_minimal.h"
#include "htnorm_splitmix64.h"

typedef struct {pcg32_random_t gen[2];} pcg64_random_t;

// the splitmax64 bit generator is used to seed the PCG32 bitgenerator
static inline void
pcg32_init(pcg32_random_t* rng, uint64_t* seed)
{
    rng->state = splitmix64_next64(seed);
    rng->inc = splitmix64_next64(seed);
}

// Generate a random unsigned 64bit integer using PCG64 bit generator
// Two 32 bit integers are used as explained in:
// https://github.com/imneme/pcg-c-basic/blob/master/pcg32x2-demo.c
static ALWAYS_INLINE(uint64_t)
pcg64_next_int(void* rng)
{
    pcg64_random_t* pcg = rng;
    return ((uint64_t)(pcg32_random_r(pcg->gen)) << 32) | pcg32_random_r(pcg->gen+1);
}

// Generate a random double in the interval (0, 1) using PCG64
static ALWAYS_INLINE(double)
pcg64_next_double(void* rng)
{
    pcg64_random_t* pcg = rng;
    // adapated from: shorturl.at/fjltD
    return (pcg64_next_int(pcg) >> 11) * (1.0 / 9007199254740992.0);
}

#endif
