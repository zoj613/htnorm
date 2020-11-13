#include <stdlib.h>

#include "splitmax64.h"
#include "pcg64.h"


static void
pcg32_init(pcg32_random_t* rng, uint64_t* seed)
{
    rng->state = splitmix64_next64(seed);
    rng->inc = splitmix64_next64(seed);
}


uint64_t
pcg64_next_int(void* rng)
{
    // https://github.com/imneme/pcg-c-basic/blob/bc39cd76ac3d541e618606bcc6e1e5ba5e5e6aa3/pcg32x2-demo.c#L72-L73
    pcg64_random_t* pcg = rng;
    return ((uint64_t)(pcg32_random_r(pcg->gen)) << 32) | pcg32_random_r(pcg->gen+1);
}


double
pcg64_next_double(void* rng)
{
    // https://github.com/numpy/numpy/blob/509b5ae4a7bb3c99324fd302ead1bbf4b130c741/numpy/random/_common.pxd#L68
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
