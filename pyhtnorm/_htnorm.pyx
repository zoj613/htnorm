# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
"""
Copyright (c) 2020, Zolisa Bleki

SPDX-License-Identifier: BSD-3-Clause */
"""
import array
from cpython cimport array
from libc.stdint cimport uint64_t

from pyhtnorm.c_htnorm cimport *


cdef extern from "../src/dist.h":
    int HTNORM_ALLOC_ERROR


cdef validate_return_info(info):
    if info == HTNORM_ALLOC_ERROR:
        raise MemoryError("Not enough memory to allocate resources.")
    elif info < 0:
        raise ValueError("Possible illegal value in one of the inputs.")
    elif info > 0:
        raise ValueError(
            f"""Either the leading minor of the {info}'th order is not
            positive definite (meaning the covariance matrix is also not
            positive definite), or factorization of one of the inputs
            returned a `U` with a zero in the {info}'th diagonal."""
        )


cdef inline rng_t* get_pcg_instance(seed=None):
    return rng_pcg64_new_seeded(<uint64_t>seed) if seed else rng_pcg64_new()


cdef inline rng_t* get_xrs_instance(seed=None):
    return rng_xrs128p_new_seeded(<uint64_t>seed) if seed else rng_xrs128p_new()


cdef object _VALID_MATRIX_TYPES = {NORMAL, DIAGONAL, IDENTITY}


cdef class HTNGenerator:

    def __cinit__(self, seed=None, gen=None):
        if seed and not (isinstance(seed, int) and seed >= 0):
            raise ValueError('`seed` needs to be an int and non-negative.')
        if gen not in {None, 'pcg', 'xrs'}:
            raise ValueError(f'bitgenerator {gen} is not supported.')

        self.rng = get_pcg_instance(seed) if gen == 'pcg' else get_xrs_instance(seed)
        if self.rng is NULL:
            raise MemoryError('Not enough memory to allocate for `HTNGenerator`')

        self.pyarr = array.array('d', [])  # template for python return output

    def __dealloc__(self):
        if self.rng is not NULL:
            rng_free(self.rng)

    cpdef hyperplane_truncated_mvnorm(
        self,
        double[:] mean,
        double[:,::1] cov,
        double[:,::1] g,
        double[:] r,
        bint diag=False,
        double[:] out=None
    ):
        """
        hyperplane_truncated_mvnorm(mean, cov, g, r, diag=False, out=False)
        """
        if g.shape[1] != cov.shape[0]:
            raise RuntimeError(
                '`G` number of cols need to be equal to `cov` number of rows'
            )

        cdef ht_config_t config
        cdef int info;

        config.gnrow = g.shape[0]
        config.gncol = g.shape[1]
        config.is_diag = diag
        config.mean = &mean[0]
        config.cov = &cov[0, 0]
        config.g = &g[0, 0]
        config.r = &r[0]

        if out is None:
            out = array.clone(self.pyarr, mean.shape[0], zero=False)

        with nogil:
            info = hplane_mvn(self.rng, &config, &out[0])

        validate_return_info(info)
        # return the base object of the memoryview
        return out.base

    cpdef structured_precision_mvnorm(
        self,
        double[:] mean,
        double[:,::1] a,
        double[:,::1] phi,
        double[:,::1] omega,
        bint mean_structured=False,
        int a_type=0,
        int o_type=0,
        double[:] out=None
    ):
        """
        structured_precision_mvnorm(mean, a, phi, omega, mean_structured=False,
                                    a_type=0, o_type=0, out=None)
        """
        if (omega.shape[0] != omega.shape[1]) or (a.shape[0] != a.shape[1]):
            raise ValueError('`omega` and `a` both need to be square matrices')

        if (phi.shape[1] != omega.shape[0]) or (phi.shape[0] != a.shape[1]):
            raise ValueError(
                "Shapes of `phi`, `omega` and `a` are not consistent"
            )

        if not {a_type, o_type}.issubset(_VALID_MATRIX_TYPES):
            raise ValueError(
                "`a_type` and `o_type` must be one of "
                f"({NORMAL}, {DIAGONAL}, {IDENTITY})"
            )

        cdef sp_config_t config

        config.struct_mean = mean_structured
        config.a_id = <mat_type>a_type
        config.o_id = <mat_type>o_type
        config.pnrow = phi.shape[0]
        config.pncol = phi.shape[1]
        config.mean = &mean[0]
        config.a = &a[0, 0]
        config.phi = &phi[0, 0]
        config.omega = &omega[0, 0]

        if out is None:
            out = array.clone(self.pyarr, mean.shape[0], zero=False)

        with nogil:
            info = str_prec_mvn(self.rng, &config, &out[0])

        validate_return_info(info)
        return out.base
