#include <stdlib.h>
#include <time.h>

#include "../include/rng.h"


rng_t*
rng_new(void)
{
    rng_t* rng = malloc(sizeof(rng_t));
    if (rng != NULL)
        pcg32_srandom_r(rng, time(NULL), (intptr_t)&rng);
    return rng;
}


rng_t*
rng_new_seeded(uint64_t state, uint64_t seq)
{
    rng_t* rng = malloc(sizeof(rng_t));
    if (rng != NULL)
        pcg32_srandom_r(rng, state, seq);
    return rng;
}

