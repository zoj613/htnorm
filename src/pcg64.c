/* Copyright (c) 2020, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include <stdlib.h>

#include "splitmax64.h"
#include "pcg64.h"

// the splitmax64 bit generator is used to seed the PCG32 bitgenerator
static void
pcg32_init(pcg32_random_t* rng, uint64_t* seed)
{
    rng->state = splitmix64_next64(seed);
    rng->inc = splitmix64_next64(seed);
}

// Generate a random unsigned 64bit integer using PCG64 bit generator
// Two 32 bit integers are used as explained in:
// https://github.com/imneme/pcg-c-basic/blob/master/pcg32x2-demo.c
static uint64_t
pcg64_next_int(void* rng)
{
    pcg64_random_t* pcg = rng;
    return ((uint64_t)(pcg32_random_r(pcg->gen)) << 32) | pcg32_random_r(pcg->gen+1);
}


// Generate a random double in the interval (0, 1) using PCG64
static double
pcg64_next_double(void* rng)
{
    pcg64_random_t* pcg = rng;
    // adapated from: shorturl.at/fjltD
    return (pcg64_next_int(pcg) >> 11) * (1.0 / 9007199254740992.0);
}


// Initialize an instance of PCG64 bit generator using a specified seed
pcg64_random_t*
pcg64_init(uint64_t seed)
{
    pcg64_random_t* rng = malloc(sizeof(pcg64_random_t));
    if (rng != NULL) {
        pcg32_init(rng->gen, &seed);
        pcg32_init(rng->gen + 1, &seed);
        rng->next_int = pcg64_next_int;
        rng->next_double = pcg64_next_double;
    }
    return rng; 
}
