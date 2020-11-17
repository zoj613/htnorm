/* Copyright (c) 2020, Zolisa Bleki
 *
 * SPDX-License-Identifier: BSD-3-Clause */
#ifndef HTNORM_XOROSHIRO128P_H
#define HTNORM_XOROSHIRO128P_H

#include <stddef.h>
#include <stdint.h>

typedef struct xrs128p_rng {
    uint64_t s[2];
    uint64_t (*next_int)(void* xrs128p);
    double (*next_double)(void* xrs128p);
} xrs128p_random_t;


// Initialize a xoroshiro128p state using a specified seed
xrs128p_random_t* xrs128p_init(uint64_t seed);
/* This is the jump function for the generator. It is equivalent
   to 2^64 calls to next() if JUMP array is used and 2^96 is LONG_JUMP is used.
   If passed JUMP: It can be used to generate 2^64 non-overlapping
   subsequences for parallel computations. 
   If passed LONG_JUMP: it can be used to generate 2^32 starting points,
   from each of which a call to this function with JUMP will generate 2^32
   non-overlapping subsequences for parallel distributed computations.
*/
void xrs128p_jump(xrs128p_random_t* rng, size_t jmp);

#endif
