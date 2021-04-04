/* Originally written by in 2016-2018 by D. Blackman and S. Vigna and
 * the original code is published under public domain. See:
 * http://xorshift.di.unimi.it/xoroshiro128plus.c
 *
 * Modifications and content in this file is published under the BSD-3 license.
 * SPDX-License-Identifier: BSD-3-Clause */
#ifndef HTNORM_XOROSHIRO128P_H
#define HTNORM_XOROSHIRO128P_H

#include <stdint.h>

#include "htnorm_always_inline.h"

typedef struct {uint64_t s[2];} xrs128p_random_t;


static ALWAYS_INLINE(uint64_t)
rotl(const uint64_t x, int k)
{
	return (x << k) | (x >> (64 - k));
}


static ALWAYS_INLINE(uint64_t)
xrs128p_next_int(void* rng)
{
    xrs128p_random_t* xrs = rng;
	const uint64_t s0 = xrs->s[0];
	uint64_t s1 = xrs->s[1];
	const uint64_t result = s0 + s1;

	s1 ^= s0;
	xrs->s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16); // a, b
	xrs->s[1] = rotl(s1, 37); // c

	return result;
}


static ALWAYS_INLINE(double)
xrs128p_next_double(void* rng)
{
    xrs128p_random_t* xrs = rng;
    return (xrs128p_next_int(xrs) >> 11) * (1.0 / 9007199254740992.0);
}

#endif
