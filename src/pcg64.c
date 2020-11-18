/* Copyright (c) 2020, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include <stdlib.h>

#include "splitmax64.h"
#include "pcg64.h"


static void
pcg32_init(pcg32_random_t* rng, uint64_t* seed)
{
    rng->state = splitmix64_next64(seed);
    rng->inc = splitmix64_next64(seed);
}


static uint64_t
pcg64_next_int(void* rng)
{
    // adapted from: shorturl.at/fsDEW
    pcg64_random_t* pcg = rng;
    return ((uint64_t)(pcg32_random_r(pcg->gen)) << 32) | pcg32_random_r(pcg->gen+1);
}


static double
pcg64_next_double(void* rng)
{
    // adapated from: shorturl.at/fjltD
    pcg64_random_t* pcg = rng;
    return (pcg64_next_int(pcg) >> 11) * (1.0 / 9007199254740992.0);
}


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
