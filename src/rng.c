/* Copyright (c) 2020, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include <stdlib.h>
#include <time.h>

#include "pcg64.h"
#include "splitmax64.h"
#include "xoroshiro128p.h"
#include "../include/rng.h"


inline void
rng_free(rng_t* rng)
{
    free(rng->base);
    free(rng);
}


rng_t*
rng_pcg64_new_seeded(uint64_t seed)
{
    rng_t* rng = malloc(sizeof(rng_t));
    if (rng != NULL) {
        pcg64_random_t* base = pcg64_init(seed);
        rng->base = base;
        rng->next_int = base->next_int;
        rng->next_double = base->next_double;
    }
    return rng;
}


inline rng_t*
rng_pcg64_new(void)
{
    uint64_t seed = time(NULL);
    return rng_pcg64_new_seeded((intptr_t)&seed);
}


rng_t*
rng_xrs128p_new_seeded(uint64_t seed)
{
    rng_t* rng = malloc(sizeof(rng_t));
    if (rng != NULL) {
        xrs128p_random_t* base = xrs128p_init(seed);
        rng->base = base;
        rng->next_int = base->next_int;
        rng->next_double = base->next_double;
    }
    return rng;
}


inline rng_t*
rng_xrs128p_new(void)
{
    uint64_t seed = time(NULL);
    return rng_xrs128p_new_seeded((intptr_t)&seed);
}
