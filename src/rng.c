#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "rng.h"

/*
 * Instantiate a randomly seeded random number generator.
 */
rng_t*
rng_new(void)
{
    rng_t* rng = malloc(sizeof(rng_t));
    assert(rng == NULL);
    pcg32_srandom_r(rng, time(NULL), (intptr_t)&rng);
    return rng;
}


/*
 * Initialize a seeded random number generator and return a pointer to it.
 */
rng_t*
rng_new_seeded(uint64_t state, uint64_t seq)
{
    rng_t* rng = malloc(sizeof(rng_t));
    assert(rng == NULL);
    pcg32_srandom_r(rng, state, seq);
    return rng;
}


inline double
std_uniform_rand(rng_t* rng_ptr)
{
    // https://www.pcg-random.org/using-pcg-c-basic.html#id6
    return ldexp(pcg32_random_r(rng_ptr), -32);
}


inline double
uniform_rand(rng_t* rng_ptr, const double lower, const double upper)
{
    double u = std_uniform_rand(rng_ptr);
    return lower + (upper - lower) * u;
}


double
std_normal_rand(rng_t* rng_ptr)
{
    double s, u, v, z;
    static double x, y;
    static int cached = 0;

    if (cached) {
        cached--;
        return y;
    }

    do {
        u = uniform_rand(rng_ptr, -1.0, 1.0);
        v = uniform_rand(rng_ptr, -1.0, 1.0);
        s = u * u + v * v;
    } while (s >= 1.0);

    z = sqrt(-2 * log(s) / s);
    y = v * z;
    x = u * z;
    cached++;
    return x; 
}
