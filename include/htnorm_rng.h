/* Copyright (c) 2020, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause 
 *
 * The public API of the underlying random number generator. The `rng_t`
 * struct is passed as a parameter to both sampling functions in the htnorm
 * header.
 */
#ifndef HTNORM_RNG_H
#define HTNORM_RNG_H

#include <stdint.h>

/* The bitgenerator interface. Users can use this struct in order to define
 * their own bitgenerator, The minimum requirement is to provide a function
 * that generate 64bit unsigned integers and doubles in the range (0, 1).
 */
typedef struct {
    // the bsse bitgenerator
    void* base;
    // a function pointer that takes `base` as an input an returns a positive intgger
    uint64_t (*next_uint64)(void* base);
    // a function pointer that takes `base` as an input an returns a double in the range (0, 1)
    double (*next_double)(void* base);
} rng_t;

// Helper function to deallocate a `rng_t` pointer.
void rng_free(rng_t* rng);

/* The following functions provide an interface to easily generate random
 * integers or standard uniform variables using either PCG64 or Xoroshiro128plus
 * bit generators.
 *
 * For more details, see: 
 * 1) https://www.pcg-random.org/
 * 2) https://www.pcg-random.org/posts/visualizing-the-heart-of-some-prngs.html
 * 3) https://www.pcg-random.org/posts/bounded-rands.html
 */

// Return a pointer to rng_t based on PCG64 bit generator that is randomly seeded.
rng_t* rng_pcg64_new(void);
// Return a pointer to rng_t based n PCG64 bit generator with a specified seed.
rng_t* rng_pcg64_new_seeded(uint64_t seed);

// Return a pointer to rng_t based on Xoroshiro128plus that is randomly seeded.
rng_t* rng_xrs128p_new(void);
// Return a pointer to rng_t based on Xoroshiro128plus with a specified seed.
rng_t* rng_xrs128p_new_seeded(uint64_t seed);

#endif
