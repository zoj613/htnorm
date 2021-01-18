import numpy as np
import pytest

from pyhtnorm import hyperplane_truncated_mvnorm, structured_precision_mvnorm


@pytest.fixture
def hypertruncated_mvn_data():
    k1 = 100
    k2 = 20
    gg = np.random.default_rng(0)
    mean = gg.random(k1)
    a = gg.random((k1, k1))
    cov = a @ a.T
    cov = cov + np.diag(gg.random(k1))
    G = gg.standard_normal((k2, k1))
    r = gg.standard_normal(k2)
    return mean, cov, G, r


@pytest.fixture
def structured_mvn_data(hypertruncated_mvn_data):
    mean, cov, _, r = hypertruncated_mvn_data
    k1 = mean.shape[0]
    k2 = r.shape[0]
    gg = np.random.default_rng(0)
    omega, phi = np.linalg.eigh(cov)
    phi = phi[:k2, :]
    omega = np.diag(omega[:k2])
    a = np.diag(gg.random(k1))
    return mean, a, omega, phi


def test_hypertruncated_mvn(hypertruncated_mvn_data):
    mean, cov, G, r = hypertruncated_mvn_data
    # raise error when non-C-contiguous array is passed
    with pytest.raises(ValueError, match='ndarray is not C-contiguous'):
        hyperplane_truncated_mvnorm(mean, cov.T, G, r)

    # check if G and cov dimensions match
    with pytest.raises(ValueError):
        G2 = np.ascontiguousarray(G[:, :10])
        hyperplane_truncated_mvnorm(mean, cov, G2, r)

    # test for non numerical input
    mean2 = mean.copy()
    mean2[0] = np.nan
    assert np.alltrue(np.isnan(hyperplane_truncated_mvnorm(mean2, cov, G, r)))

    # test consistency of output when `diag=True` is used for same seed
    cov_diag = np.diag(np.random.rand(cov.shape[0]))
    rng = np.random.default_rng(10)
    arr1 = hyperplane_truncated_mvnorm(mean, cov_diag, G, r, random_state=rng)
    rng = np.random.default_rng(10)
    arr2 = hyperplane_truncated_mvnorm(mean, cov_diag, G, r, diag=True, random_state=rng)
    assert np.allclose(arr1, arr2)
    # test results of passing output array through the `out` parameter
    hyperplane_truncated_mvnorm(mean, cov, G, r, out=arr1)
    assert not np.allclose(arr1, arr2)
    # test results of samples truncated on the hyperplane sum(x) = 0
    G = np.ones((1, G.shape[1]))
    r = np.zeros(1)
    hyperplane_truncated_mvnorm(mean, cov, G, r, out=arr1)
    assert np.allclose(sum(arr1), 0)


def test_structured_mvn(structured_mvn_data):
    mean, a, omega, phi = structured_mvn_data
    # check if sampling works
    structured_precision_mvnorm(mean, a, phi, omega)
    # raise error when non-C-contiguous array is passed
    with pytest.raises(ValueError, match='ndarray is not C-contiguous'):
        structured_precision_mvnorm(mean, a.T, phi, omega)
    # raise error if dimensions of phi, omega, and a dont match
    with pytest.raises(ValueError):
        phi2 = np.ascontiguousarray(phi[1:, 1:])
        structured_precision_mvnorm(mean, a, phi2, omega)
    # raise error if omega and a are not square matrices
    with pytest.raises(ValueError):
        omega2 = np.ascontiguousarray(omega[1:, :])
        a2 = np.ascontiguousarray(a[1:, :])
        structured_precision_mvnorm(mean, a2, phi, omega2)
    # raise error if invalid matrix structure is specified
    with pytest.raises(ValueError):
        structured_precision_mvnorm(mean, a, phi, omega, a_type=-1000)
    # rest for non-numerical values
    a2 = a.copy()
    a2[0] = np.nan
    assert np.alltrue(
        np.isnan(structured_precision_mvnorm(mean, a2, phi, omega))
    )
    # test consistency of output when `a_type` or `o_type` is given
    rng = np.random.default_rng(10)
    arr1 = structured_precision_mvnorm(mean, a, phi, omega, random_state=rng)
    rng = np.random.default_rng(10)
    arr2 = structured_precision_mvnorm(
        mean, a, phi, omega, o_type=1, a_type=1, random_state=rng
    )
    assert np.allclose(arr1, arr2)
    # test results of passing output array through the `out` parameter
    structured_precision_mvnorm(mean, a, phi, omega, out=arr1)
    assert not np.allclose(arr1, arr2)


def test_generator_seed(hypertruncated_mvn_data):
    mean, cov, G, r = hypertruncated_mvn_data
    rng1 = np.random.default_rng(10)
    arr1 = hyperplane_truncated_mvnorm(mean, cov, G, r, random_state=rng1)
    rng2 = np.random.default_rng(10)
    arr2 = hyperplane_truncated_mvnorm(mean, cov, G, r, random_state=rng2)
    rng3 = np.random.default_rng(20)
    arr3 = hyperplane_truncated_mvnorm(mean, cov, G, r, random_state=rng3)
    # test reproducability via seeding
    assert np.allclose(arr1, arr2)
    assert not np.allclose(arr1, arr3)
    # test if the errors are raised when the wrong input for seed is given
    with pytest.raises(ValueError):
        hyperplane_truncated_mvnorm(mean, cov, G, r, random_state=-100)
    with pytest.raises(TypeError):
        hyperplane_truncated_mvnorm(mean, cov, G, r, random_state='100')

