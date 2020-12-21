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
    """
    Sample from a multivariate normal truncated on a hyperplane

    Parameters
    ----------
    seed : int, optional, default=None
        A random seed for the underlying bitgenerator.If not set, then the
        eenerator will be randomly seeded.
    gen : str, optional, default=None
        The name of the generator to use for random number generation. The
        value needs to be one of {'pcg', 'xrs'}, where 'pcg' is PCG64 and 'xrs'
        is the Xoroshiro128plus bit generator.

    Methods
    -------
    hyperplane_truncated_mvnorm(mean, cov, g, r, diag=False, out=False)
    structured_precision_mvnorm(mean, a, phi, omega, mean_structured=False,
                                a_type=0, o_type=0, out=None)

    Notes
    -----
    This class implements algorithms 2, 4 and the one in example 4 of [1]_
    Where one can generate efficiently samples from a MVN truncated on a
    hyperplace

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

        Returns
        -------
        out : {array.array, array_like}
            A sample from this distribution. If `out` is given then the return
            value will be `out`, else a new python array object is returned.

        Raises
        ------
        RuntimeError
            When the dimensions of the input arrays do not match.
        MemoryError
            If the program runs out of memory while sampling.
        ValueError
            If the sampling was not successful because of faulty input or if
            the algorithm could not successfully generate the samples.

        """
        if g.shape[1] != cov.shape[0]:
            raise RuntimeError(
                '`G` number of cols need to be equal to `cov` number of rows'
            )

        cdef ht_config_t config
        cdef int info

        config.gnrow = g.shape[0]
        config.gncol = g.shape[1]
        config.is_diag = diag
        config.mean = &mean[0]
        config.cov = &cov[0, 0]
        config.g = &g[0, 0]
        config.r = &r[0]

        if out is None:
            out = array.clone(self.pyarr, mean.shape[0], zero=False)
            self.noreturn = False
        else:
            self.noreturn = True

        with nogil:
            info = hplane_mvn(self.rng, &config, &out[0])

        validate_return_info(info)
        if not self.noreturn:
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
        a_type : {0, 1, 2}, optional, default=0
            Whether `a` ia a normal, diagonal or identity matrix.
        o_type : {0, 1, 2}, optional, default=0
            Whether `omega` ia a normal, diagonal or identity matrix.
        out : 1d array, optional, default=None
            An array of the same shape as `mean` to store the samples. If not
            set the a new python array object is created and the samples are
            copied into before the method returns.

        Returns
        -------
        out : {array.array, array_like}
            A sample from this distribution. If `out` is given then the return
            value will be `out`, else a new python array object is returned.

        Raises
        ------
        RuntimeError
            When the dimensions of the input arrays do not match.
        MemoryError
            If the program runs out of memory while sampling.
        ValueError
            If the sampling was not successful because of faulty input or if
            the algorithm could not successfully generate the samples.

        """
        if (omega.shape[0] != omega.shape[1]) or (a.shape[0] != a.shape[1]):
            raise ValueError('`omega` and `a` both need to be square matrices')

        if (phi.shape[0] != omega.shape[0]) or (phi.shape[1] != a.shape[0]):
            raise ValueError(
                "Shapes of `phi`, `omega` and `a` are not consistent"
            )

        if not {a_type, o_type}.issubset(_VALID_MATRIX_TYPES):
            raise ValueError(
                "`a_type` and `o_type` must be one of "
                f"({NORMAL}, {DIAGONAL}, {IDENTITY})"
            )

        cdef sp_config_t config
        cdef int info

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
            self.noreturn = False
        else:
            self.noreturn = True

        with nogil:
            info = str_prec_mvn(self.rng, &config, &out[0])

        validate_return_info(info)
        if not self.noreturn:
            # return the base object of the memoryview
            return out.base
