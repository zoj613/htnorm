# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
"""
Copyright (c) 2020-2021, Zolisa Bleki
SPDX-License-Identifier: BSD-3-Clause


This module implements algorithms 2, 4 and the one in example 4 of [1]_
Where one can generate efficiently samples from a MVN truncated on a hyperplace.

.. math::
    \mathbf{G}\mathbf{x} = \mathbf{r},
where :math:`rank(\mathbf{G}) = k_2 < rank(\mathbf{\Sigma})`.
This class also implements an efficient way that leverages high dimensional
truncated MVN to sample from a MVN with a structured precision matrix, as
described in algorithm 4 of [1]_, where the precision
:math:`\mathbf{\Lambda}` can be written as
.. math::
    \mathbf{\Lambda} = (\mathbf{A} + \mathbf{Phi}^T\mathbf{\Omega}\mathbf{Phi}),
    \mathbf{\Phi} \in \R^{n \times p}, \mathbf{\Omega} \in \R^{n \times n}
    and \mathbf{A} \in \R^{p \times p}.

Algorithm 4 can be extended to sample from a MVN whose mean depends on the
structure of the precision, as shown in example 4 of [1]_ .

References
----------
.. [1] Cong, Yulai; Chen, Bo; Zhou, Mingyuan. Fast Simulation of Hyperplane-
   Truncated Multivariate Normal Distributions. Bayesian Anal. 12 (2017),
   no. 4, 1017--1037. doi:10.1214/17-BA1052. https://projecteuclid.org/euclid.ba/1488337478

"""
from cpython.pycapsule cimport PyCapsule_GetPointer
from libc.stdint cimport uint64_t

cimport numpy as np
from numpy.random cimport BitGenerator, bitgen_t

from numpy.random import default_rng


np.import_array()

cdef extern from "htnorm_distributions.h":
    int HTNORM_ALLOC_ERROR


cdef extern from "htnorm_rng.h" nogil:
    ctypedef struct rng_t:
        void* base
        uint64_t (*next_uint64)(void* state)
        double (*next_double)(void* state)


cdef extern from "htnorm.h" nogil:
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
        bint diag

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
                        const double* r, bint diag)

    void init_sp_config(sp_config_t* conf, size_t pnrow, size_t pncol,
                        const double* mean, const double* a, const double* phi,
                        const double* omega, bint struct_mean, mat_type a_id,
                        mat_type o_id)

    int htn_hyperplane_truncated_mvn(rng_t* rng, const ht_config_t* conf, double* out)

    int htn_structured_precision_mvn(rng_t* rng, const sp_config_t* conf, double* out)


cdef inline int validate_return_info(int info) except? -1:
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


cdef dict MAT_TYPE = {"regular": NORMAL, "diagonal": DIAGONAL, "identity": IDENTITY}
cdef const char* BITGEN_NAME = "BitGenerator"


cdef inline void initialize_rng(object bitgenerator, rng_t* htnorm_rng):
    cdef bitgen_t* bitgen

    bitgen = <bitgen_t*>PyCapsule_GetPointer(bitgenerator.capsule, BITGEN_NAME)
    htnorm_rng.base = bitgen.state
    htnorm_rng.next_uint64 = bitgen.next_uint64
    htnorm_rng.next_double = bitgen.next_double


def hyperplane_truncated_mvnorm(
    double[:] mean,
    double[:,::1] cov,
    double[:,::1] g,
    double[:] r,
    *,
    bint diag=False,
    double[:] out=None,
    random_state=None
):
    """
    hyperplane_truncated_mvnorm(mean, cov, g, r, diag=False, out=False,
                                random_state=None)

    Sample from a multivariate normal truncated on a hyperplane

    .. math::
        \mathbf{G}\mathbf{x} = \mathbf{r},

    where :math:`rank(\mathbf{G}) = k_2 < rank(\mathbf{\Sigma})`.

    Parameters
    ----------
    mean : 1d array
        The mean of the distribution.
    cov : 2d array
        The covariance of the distribution.
    g : 2d array
        The ``G`` matrix representing the left-hand-side of the plane.
    r : 1d array
        The right-hand side of the hyperplane.
    diag : bool, optional, default=False
        Setting this option to True if ``cov`` is a diagonal matrix can
        speed up sampling.
    out : 1d array, optional, default=None
        An array of the same shape as `mean` to store the samples. If not
        set the a new python array object is created and the samples are
        copied into before the method returns.
    random_state : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}, optional
        A seed to initialize the random number generator. If None, then fresh,
        unpredictable entropy will be pulled from the OS. If an ``int`` or
        ``array_like[ints]`` is passed, then it will be passed to
        `SeedSequence` to derive the initial `BitGenerator` state. One may also
        pass in a `SeedSequence` instance.
        Additionally, when passed a `BitGenerator`, it will be wrapped by
        `Generator`. If passed a `Generator`, it will be returned unaltered.

    Returns
    -------
    out : np.ndarray
        A sample from this distribution. If `out` is given then the return
        value will be `out`, else a new numpy array is returned.

    Raises
    ------
    ValueError
        When the dimensions of the input arrays do not match.
        If the sampling was not successful because of faulty input or if
        the algorithm could not successfully generate the samples.

    """
    cdef rng_t rng
    cdef ht_config_t config
    cdef int info
    cdef bint has_out = True if out is not None else False

    if g.shape[1] != cov.shape[0]:
        raise ValueError('`G` number of columns must equal `cov` number of rows')
    elif has_out and out.shape[0] != mean.shape[0]:
        raise ValueError("`out` must have the same size as the mean array.")
    elif not has_out:
        dims = <np.npy_intp>(mean.shape[0])
        out = np.PyArray_EMPTY(1, &dims, np.NPY_DOUBLE, 0)

    init_ht_config(&config, g.shape[0], g.shape[1], &mean[0],
                   &cov[0, 0], &g[0, 0], &r[0], diag)

    bitgenerator = default_rng(random_state)._bit_generator
    initialize_rng(bitgenerator, &rng)

    with bitgenerator.lock, nogil:
        info = htn_hyperplane_truncated_mvn(&rng, &config, &out[0])

    validate_return_info(info)
    if not has_out:
        return out.base


def structured_precision_mvnorm(
    double[:] mean,
    double[:,::1] a,
    double[:,::1] phi,
    double[:,::1] omega,
    bint mean_structured=False,
    str a_type="regular",
    str o_type="regular",
    double[:] out=None,
    random_state=None
):
    """
    structured_precision_mvnorm(mean, a, phi, omega, mean_structured=False,
                                a_type="regular", o_type="regular",
                                out=None, random_state=None)

    Sample from a MVN with a structured precision matrix :math:`\Lambda`
    .. math::
         \mathbf{\Lambda} = (\mathbf{A} + \mathbf{Phi}^T\mathbf{\Omega}\mathbf{Phi})

    Parameters
    ----------
    mean : 1d array
        The mean of the distribution.
    a : 2d array
        matrix ``A`` in the precision matrix structure.
    phi : 2d array
        matrix ``Phi`` in the precision matrix structure.
    omega : 2d array
        matrix ``omega`` in the precision matrix structure.
    mean_structured : bool, optional, default=False
        whether the mean is also structured and depends on the precision
        such than ``mean = (precision)^-1 * phi^T * omega * t``. If this
        is set to True, then the `mean` parameter is assumed to contain the
        array ``t``.
    a_type : {"regular", "diagonal", "identity"}, optional, default="regular"
        Whether `a` ia a normal, diagonal or identity matrix.
    o_type : {"regular", "diagonal", "identity"}, optional, default="regular"
        Whether `omega` ia a normal, diagonal or identity matrix.
    out : 1d array, optional, default=None
        An array of the same shape as `mean` to store the samples. If not
        set the a new python array object is created and the samples are
        copied into before the method returns.
    random_state : {None, int, array_like[ints], SeedSequence, BitGenerator, Generator}, optional
        A seed to initialize the random number generator. If None, then fresh,
        unpredictable entropy will be pulled from the OS. If an ``int`` or
        ``array_like[ints]`` is passed, then it will be passed to
        `SeedSequence` to derive the initial `BitGenerator` state. One may also
        pass in a `SeedSequence` instance.
        Additionally, when passed a `BitGenerator`, it will be wrapped by
        `Generator`. If passed a `Generator`, it will be returned unaltered.

    Returns
    -------
    out : np.ndarray
        A sample from this distribution. If `out` is given then the return
        value will be `out`, else a new numpy array is returned.

    Raises
    ------
    ValueError
        When the dimensions of the input arrays do not match.
        If the sampling was not successful because of faulty input or if
        the algorithm could not successfully generate the samples.

    """
    cdef rng_t rng
    cdef sp_config_t config
    cdef int info
    cdef bint has_out = True if out is not None else False

    if (omega.shape[0] != omega.shape[1]) or (a.shape[0] != a.shape[1]):
        raise ValueError('`omega` and `a` both need to be square matrices')
    elif (phi.shape[0] != omega.shape[0]) or (phi.shape[1] != a.shape[0]):
        raise ValueError('Shapes of `phi`, `omega` and `a` are not consistent')
    elif not {a_type, o_type}.issubset(MAT_TYPE):
        raise ValueError(f"`a_type` & `o_type` must be one of {set(MAT_TYPE)}")
    elif has_out and out.shape[0] != mean.shape[0]:
        raise ValueError("`out` must have the same size as the mean array.")
    elif not has_out:
        dims = <np.npy_intp>(mean.shape[0])
        out = np.PyArray_EMPTY(1, &dims, np.NPY_DOUBLE, 0)

    init_sp_config(&config, phi.shape[0], phi.shape[1], &mean[0], &a[0, 0],
                   &phi[0, 0], &omega[0, 0], mean_structured,
                   MAT_TYPE[a_type], MAT_TYPE[o_type])

    bitgenerator = default_rng(random_state)._bit_generator
    initialize_rng(bitgenerator, &rng)

    with bitgenerator.lock, nogil:
        info = htn_structured_precision_mvn(&rng, &config, &out[0])

    validate_return_info(info)
    if not has_out:
        return out.base
