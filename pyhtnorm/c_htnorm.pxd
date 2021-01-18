# cython: language_level=3
"""
Copyright (c) 2020-2021, Zolisa Bleki

SPDX-License-Identifier: BSD-3-Clause */
"""
from libc.stdint cimport uint64_t


cdef extern from "../include/rng.h":
    ctypedef struct rng_t:
        void* base
        uint64_t (*next_uint64)(void* state) nogil
        double (*next_double)(void* state) nogil


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

    void init_ht_config(ht_config_t* conf, size_t gnrow, size_t gncol,
                        const double* mean, const double* cov, const double* g,
                        const double* r, bint diag) nogil

    void init_sp_config(sp_config_t* conf, size_t pnrow, size_t pncol,
                        const double* mean, const double* a, const double* phi,
                        const double* omega, bint struct_mean, mat_type a_id,
                        mat_type o_id) nogil

    int htn_hyperplane_truncated_mvn(rng_t* rng, const ht_config_t* conf, double* out) nogil

    int htn_structured_precision_mvn(rng_t* rng, const sp_config_t* conf, double* out) nogil
