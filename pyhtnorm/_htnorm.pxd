# cython: language_level=3
"""
Copyright (c) 2020, Zolisa Bleki

SPDX-License-Identifier: BSD-3-Clause */
"""
from cpython cimport array

from pyhtnorm.c_htnorm cimport rng_t


cdef class HTNGenerator:
    cdef rng_t* rng
    cdef array.array pyarr
    cpdef hyperplane_truncated_mvnorm(
        self,
        double[:] mean,
        double[:,::1] cov,
        double[:,::1] g,
        double[:] r,
        bint diag=*,
        double[:] out=*
    )
    cpdef structured_precision_mvnorm(
        self,
        double[:] mean,
        double[:,::1] a,
        double[:,::1] phi,
        double[:,::1] omega,
        bint mean_structured=*,
        int a_type=*,
        int o_type=*,
        double[:] out=*
    )
