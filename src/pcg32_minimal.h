// *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)
//
// Modifications: Make the function always inlineale to reduce all overhead

#ifndef HTNORM_PCG32_H
#define HTNORM_PCG32_H

#include <stdint.h>

#include "always_inline.h"

typedef struct pcg32_rand {
    uint64_t state;
    uint64_t inc;
} pcg32_random_t;


static ALWAYS_INLINE(uint32_t)
pcg32_random_r(pcg32_random_t* rng)
{
    uint64_t oldstate = rng->state;
    // Advance internal state
    rng->state = oldstate * 6364136223846793005ULL + (rng->inc|1);
    // Calculate output function (XSH RR), uses old state for max ILP
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

#endif
