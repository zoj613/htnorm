/* Copyright (c) 2020, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#ifndef HTNORM_PCG64_H
#define HTNORM_PCG64_H

#include "pcg32_minimal.h"

typedef struct pcg64_rand{
    pcg32_random_t gen[2];
    // a function pointer to generate a positive integer
    uint64_t (*next_int)(void* pcg64);
    // a function pointer to generate a double in the interval (0, 1).
    double (*next_double)(void* pcg64);
} pcg64_random_t;

/* Get a pointer to an instance of a PCG64 bit generator.
 *
 * Paramaters
 * ----------
 *  seed:
 *      The seed of the generator.
 *
 *  Returns
 *  -------
 *  A pointer to an instance of `pcg64_random_t` struct.
 */
pcg64_random_t* pcg64_init(uint64_t seed);

#endif
