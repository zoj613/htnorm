# cython: language_level=3
"""
Copyright (c) 2020, Zolisa Bleki

SPDX-License-Identifier: BSD-3-Clause */
"""

from libc.stdint cimport uint32_t, uint64_t


cdef extern from "../include/rng.h":
    ctypedef struct rng_t:
        void* state
        uint32_t (*next)(void* state) nogil
        double (*next_double)(void* state) nogil

    void rng_free(rng_t* rng) nogil

    rng_t* rng_pcg64_new() nogil
    rng_t* rng_pcg64_new_seeded(uint64_t state) nogil

    rng_t* rng_xrs128p_new() nogil
    rng_t* rng_xrs128p_new_seeded(uint64_t seed) nogil


cdef extern from "../include/htnorm.h":

    ctypedef enum mat_type "type_t":
        NORMAL
        DIAGONAL
        IDENTITY

    ctypedef struct ht_config_t:
        size_t gnrow
        size_t gncol
        const double* mean
        const double* cov
        const double* g
        const double* r
        bint is_diag "diag"

    ctypedef struct sp_config_t:
        mat_type a_id
        mat_type o_id
        size_t pnrow
        size_t pncol
        const double* mean
        const double* a
        const double* phi
        const double* omega
        bint struct_mean

    int hplane_mvn "htn_hyperplane_truncated_mvn" (
        rng_t* rng, const ht_config_t* conf, double* out
    ) nogil

    int str_prec_mvn "htn_structured_precision_mvn"(
        rng_t* rng, const sp_config_t* conf, double* out
    ) nogil
