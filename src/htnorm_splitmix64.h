/*  Written in 2015 by Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */
// Original code at: http://xorshift.di.unimi.it/splitmix64.c
#ifndef HTNORM_SPLITMAX64_H
#define HTNORM_SPLITMAX64_H

#include <stdint.h>

#include "htnorm_always_inline.h"

// generate a postive integer using the splitmix64 bit generator
static ALWAYS_INLINE(uint64_t)
splitmix64_next64(uint64_t* state)
{
	uint64_t z = (*state += 0x9e3779b97f4a7c15);
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
	return z ^ (z >> 31);
}

#endif
