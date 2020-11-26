#ifndef HTNORM_SPLITMAX64_H
#define HTNORM_SPLITMAX64_H

#include <stdint.h>

// generate a postive integer using the splitmix64 bit generator
uint64_t splitmix64_next64(uint64_t* state);

#endif
