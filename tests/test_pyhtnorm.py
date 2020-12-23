import numpy as np
import pytest

from pyhtnorm import HTNGenerator


@pytest.fixture
def hypertruncated_mvn_data():
    k1 = 100
    k2 = 20
    gg = np.random.Generator(np.random.PCG64(0))
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
    gg = np.random.Generator(np.random.PCG64(0))
    omega, phi = np.linalg.eigh(cov)
    phi = phi[:k2, :]
    omega = np.diag(omega[:k2])
    a = np.diag(gg.random(k1))
    return mean, a, omega, phi


def test_wrong_rng_name():
    with pytest.raises(ValueError):
        HTNGenerator(gen='blah')
    # sanity check
    HTNGenerator()
    HTNGenerator(gen='pcg')
    HTNGenerator(gen='xrs')


def test_hypertruncated_mvn(hypertruncated_mvn_data):
    mean, cov, G, r = hypertruncated_mvn_data
    g = HTNGenerator(10)
    # raise error when non-C-contiguous array is passed
    with pytest.raises(ValueError, match='ndarray is not C-contiguous'):
        g.hyperplane_truncated_mvnorm(mean, cov.T, G, r)

    # check if G and cov dimensions match
    with pytest.raises(RuntimeError):
        G2 = np.ascontiguousarray(G[:, :10])
        g.hyperplane_truncated_mvnorm(mean, cov, G2, r)

    # test for non numerical input
    mean2 = mean.copy()
    mean2[0] = np.nan
    assert np.alltrue(np.isnan(g.hyperplane_truncated_mvnorm(mean2, cov, G, r)))

    # test consistency of output when `diag=True` is used for same seed
    cov_diag = np.diag(np.random.rand(cov.shape[0]))
    g = HTNGenerator(10)
    arr1 = g.hyperplane_truncated_mvnorm(mean, cov_diag, G, r)
    g = HTNGenerator(10)
    arr2 = g.hyperplane_truncated_mvnorm(mean, cov_diag, G, r, diag=True)
    assert np.allclose(arr1, arr2)
    # test results of passing output array through the `out` parameter
    g.hyperplane_truncated_mvnorm(mean, cov, G, r, out=arr1)
    assert not np.allclose(arr1, arr2)
    # test results of samples truncated on the hyperplane sum(x) = 0
    G = np.ones((1, G.shape[1]))
    r = np.zeros(1)
    g.hyperplane_truncated_mvnorm(mean, cov, G, r, out=arr1)
    assert np.allclose(sum(arr1), 0)


def test_structured_mvn(structured_mvn_data):
    mean, a, omega, phi = structured_mvn_data
    g = HTNGenerator(10)
    # check if sampling works
    g.structured_precision_mvnorm(mean, a, phi, omega)
    # raise error when non-C-contiguous array is passed
    with pytest.raises(ValueError, match='ndarray is not C-contiguous'):
        g.structured_precision_mvnorm(mean, a.T, phi, omega)
    # raise error if dimensions of phi, omega, and a dont match
    with pytest.raises(ValueError):
        phi2 = np.ascontiguousarray(phi[1:, 1:])
        g.structured_precision_mvnorm(mean, a, phi2, omega)
    # raise error if omega and a are not square matrices
    with pytest.raises(ValueError):
        omega2 = np.ascontiguousarray(omega[1:, :])
        a2 = np.ascontiguousarray(a[1:, :])
        g.structured_precision_mvnorm(mean, a2, phi, omega2)
    # raise error if invalid matrix structure is specified
    with pytest.raises(ValueError):
        g.structured_precision_mvnorm(mean, a, phi, omega, a_type=-1000)
    # rest for non-numerical values
    a2 = a.copy()
    a2[0] = np.nan
    assert np.alltrue(
        np.isnan(g.structured_precision_mvnorm(mean, a2, phi, omega))
    )
    # test consistency of output when `a_type` or `o_type` is given
    g = HTNGenerator(10)
    arr1 = g.structured_precision_mvnorm(mean, a, phi, omega)
    g = HTNGenerator(10)
    arr2 = g.structured_precision_mvnorm(mean, a, phi, omega, o_type=1, a_type=1)
    assert np.allclose(arr1, arr2)
    # test results of passing output array through the `out` parameter
    g.structured_precision_mvnorm(mean, a, phi, omega, out=arr1)
    assert not np.allclose(arr1, arr2)


def test_generator_seed(hypertruncated_mvn_data):
    mean, cov, G, r = hypertruncated_mvn_data
    g1 = HTNGenerator(5)
    arr1 = np.array(g1.hyperplane_truncated_mvnorm(mean, cov, G, r))
    g2 = HTNGenerator(5)
    arr2 = np.array(g2.hyperplane_truncated_mvnorm(mean, cov, G, r))
    g3 = HTNGenerator(20)
    arr3 = np.array(g3.hyperplane_truncated_mvnorm(mean, cov, G, r))
    # test reproducability via seeding
    assert np.allclose(arr1, arr2)
    assert not np.allclose(arr1, arr3)
    # test if the errors are raised when the wrong input for seed is given
    with pytest.raises(ValueError):
        HTNGenerator(seed=-100)
    with pytest.raises(ValueError):
        HTNGenerator(seed='100')

