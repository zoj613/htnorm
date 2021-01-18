"""
Copyright (c) 2020, Zolisa Bleki

SPDX-License-Identifier: BSD-3-Clause


This module provides a pure numpy implementation of the samplers implemented
by htnorm.
"""
import numpy as np


def hyperplane_truncated_mvnorm(mean, cov, G, r, diag=False, random_state=None):
    rng = np.random.default_rng(random_state)
    if diag:
        cov_diag = np.diag(cov)
        y = mean + cov_diag * rng.standard_normal(mean.shape[0])
        covg = cov_diag[:, None] * G.T
    else:
        y = rng.multivariate_normal(mean, cov, method='cholesky')
        covg = cov @ G.T
    gcovg = G @ covg
    alpha = np.linalg.solve(gcovg, r - G @ y)
    return y + covg @ alpha


def structured_precision_mvnorm(mean, a, phi, omega, a_type=0, o_type=0, random_state=None):
    rng = np.random.default_rng(random_state)
    if a_type:
        Ainv = 1 / np.diag(a)
        y1 = rng.standard_normal(a.shape[0]) / np.sqrt(Ainv)
        ainv_phi = Ainv[:, None] * phi.T
    else:
        Ainv = np.linalg.inv(a)
        y1 = rng.multivariate_normal(
            np.zeros(a.shape[0]), Ainv, method='cholesky'
        )
        ainv_phi = Ainv @ phi.T
    if o_type:
        omegainv = 1 / np.diag(omega)
        y2 = rng.standard_normal(omega.shape[0]) / np.sqrt(omegainv)
        alpha = np.linalg.solve(
            np.diag(omegainv) + phi @ ainv_phi, phi @ y1 + y2
        )
    else:
        omegainv = np.linalg.inv(omega)
        y2 = rng.multivariate_normal(
            np.zeros(omega.shape[0]), omegainv, method='cholesky'
        )
        alpha = np.linalg.solve(omegainv + phi @ ainv_phi, phi @ y1 + y2)
    return mean + y1 - ainv_phi @ alpha
