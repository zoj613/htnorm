/* Copyright (c) 2020, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#include <stdlib.h>
#include <time.h>

#include "htnorm_pcg64.h"
#include "htnorm_xoroshiro128p.h"
#include "../include/htnorm_rng.h"


ALWAYS_INLINE(void)
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
        pcg64_random_t* pcg = malloc(sizeof(pcg64_random_t));
        if (pcg != NULL) {
            pcg32_init(pcg->gen, &seed);
            pcg32_init(pcg->gen + 1, &seed);
            rng->base = pcg;
            rng->next_uint64 = pcg64_next_int;
            rng->next_double = pcg64_next_double;
        }
    }
    return rng;
}


ALWAYS_INLINE(rng_t*)
rng_pcg64_new(void)
{
    return rng_pcg64_new_seeded((uint64_t)time(NULL));
}


rng_t*
rng_xrs128p_new_seeded(uint64_t seed)
{
    rng_t* rng = malloc(sizeof(rng_t));
    if (rng != NULL) {
        xrs128p_random_t* xrs = malloc(sizeof(xrs128p_random_t));
        if (xrs != NULL) {
            xrs->s[0] = splitmix64_next64(&seed);
            xrs->s[1] = splitmix64_next64(&seed);
            rng->base = xrs;
            rng->next_uint64 = xrs128p_next_int;
            rng->next_double = xrs128p_next_double;
        }
    }
    return rng;
}


ALWAYS_INLINE(rng_t*)
rng_xrs128p_new(void)
{
    return rng_xrs128p_new_seeded((uint64_t)time(NULL));
}
