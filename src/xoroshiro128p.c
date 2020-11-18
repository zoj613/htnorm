/* Originally written by in 2016-2018 by D. Blackman and S. Vigna and
 * the original code is published under public domain. See:
 * http://xorshift.di.unimi.it/xoroshiro128plus.c
 *
 * Modifications and content in this file is published under the BSD-3 license.
 */
#include <stdlib.h>

#include "splitmax64.h"
#include "xoroshiro128p.h"


static const uint64_t JUMPS[2][2] = {
    {0xdf900294d8f554a5, 0x170865df4b3201fc},  // normal jump
    {0xd2a98b26625eee7b, 0xdddf9b1090aa7ac1}  // long jump
};
static const size_t JUMP_SIZE = sizeof(JUMPS[0]) / sizeof(*JUMPS[0]);


static inline uint64_t
rotl(const uint64_t x, int k)
{
	return (x << k) | (x >> (64 - k));
}


static uint64_t
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


static inline double
xrs128p_next_double(void* rng)
{
    // adapated from: shorturl.at/fjltD
    xrs128p_random_t* xrs = rng;
    return (xrs128p_next_int(xrs) >> 11) * (1.0 / 9007199254740992.0);
}


xrs128p_random_t*
xrs128p_init(uint64_t seed)
{
    xrs128p_random_t* rng = malloc(sizeof(xrs128p_random_t));
    if (rng != NULL) {
        rng->s[0] = splitmix64_next64(&seed);
        rng->s[1] = splitmix64_next64(&seed);
        rng->next_int = xrs128p_next_int;
        rng->next_double = xrs128p_next_double;
    }
    return rng;
}
    

void
xrs128p_jump(xrs128p_random_t* rng, size_t jmp)
{
	uint64_t s0 = 0;
	uint64_t s1 = 0;
	for(size_t i = 0; i < JUMP_SIZE; i++)
		for(size_t b = 0; b < 64; b++) {
			if (JUMPS[jmp][i] & 1ULL<< b) {
				s0 ^= rng->s[0];
				s1 ^= rng->s[1];
			}
			xrs128p_next_int(rng);
		}
	rng->s[0] = s0;
	rng->s[1] = s1;
}
