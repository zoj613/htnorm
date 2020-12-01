# Copyright (c) 2020, Zolisa Bleki
# SPDX-License-Identifier: BSD-3-Clause


hypertruncated_mvn_data <- function() {
    k1 <- 100
    k2 <- 20
    out <- list()
    out$mean <- rnorm(k1)
    cov <- matrix(rnorm(k1 * k1), ncol = k1)
    out$cov <- cov %*% t(cov)
    out$g <- matrix(rep(k2, k1), ncol = k1)
    out$r <- rnorm(k2)
    out$gnrow <- k2
    out$gncol <- k1

    out
}

structured_mvn_data <- function() {
    out <- hypertruncated_mvn_data()
    k1 <- length(out$mean)
    eig <- eigen(out$cov)
    out$phi <- eig$vectors
    out$omega <- diag(eig$values)
    out$a <- diag(runif(k1))

    out
}

test_that("wrong generator name parameter", {
    expect_that(HTNGenerator(gen = "blah"), throws_error())
    expect_that(HTNGenerator()$rng, is_a("externalptr"))
    expect_that(HTNGenerator(gen = "xrs128p"), is_a("HTNGenerator"))
    expect_that(HTNGenerator(gen = "pcg64"), is_a("HTNGenerator"))
})


test_that("hyperplane truncated norm method", {
    out <- hypertruncated_mvn_data()
    mean <- out$mean
    cov <- out$cov
    g <- out$g
    r <- out$r

    gen <- HTNGenerator(10)
    hpmvn <- gen$hyperplane_truncated_mvnorm
    expect_that(hpmvn(matrix(mean), cov, g, r), throws_error())
    expect_that(hpmvn(mean[2:out$gncol], cov, g, r), throws_error())
    expect_that(hpmvn(mean, cov, g[, 2:out$gncol], r), throws_error())
    # test consistency of output when `diag=True` is used for same seed
    cov_diag <- diag(runif(out$gncol))
    gen1 <- HTNGenerator(10)$hyperplane_truncated_mvnorm(mean, cov_diag, g, r)
    gen2 <- HTNGenerator(10)$hyperplane_truncated_mvnorm(
        mean, cov_diag, g, r, diag = TRUE
    )
    expect_equal(gen1, gen2)
    # test results of passing output array through the `out` parameter
    res <- rep(1, out$gncol)
    hpmvn(mean, cov, g, r, out = res)
    expect_false(length(sum(res == rep(1, out$gncol))) == length(res))
    # test results of samples truncated on the hyperplane sum(x) = 0
    gg <- matrix(rep(1, out$gncol), ncol = out$gncol)
    r <- c(0)
    expect_equal(sum(hpmvn(mean, cov, gg, r)), 0)
    # test for non-SPD covariance input
    c <- diag(rnorm(out$gncol))
    expect_that(hpmvn(mean, c, g, r), throws_error())
})


test_that("structured precision normal", {
    out <- structured_mvn_data()
    mean <- out$mean
    a <- out$a
    phi <- out$phi
    omega <- out$omega

    gen <- HTNGenerator(10)
    spmvn <- gen$structured_precision_mvnorm
    expect_that(spmvn(matrix(mean), a, phi, omega), throws_error())
    expect_that(spmvn(mean, a, phi[2:nrow(phi), ], omega), throws_error())
    # raise error if invalid matrix structure is specified
    expect_that(spmvn(mean, a, phi, omega, a_type = -1000), throws_error())
    # test consistency of output when `a_type` or `o_type` is given
    gen1 <- HTNGenerator(10)$structured_precision_mvnorm(mean, a, phi, omega)
    gen2 <- HTNGenerator(10)$structured_precision_mvnorm(
        mean, a, phi, omega, a_type = 1, o_type = 1
    )
    expect_equal(gen1, gen2)
    # test results of passing output array through the `out` parameter
    m <- length(mean)
    res <- rep(0, m)
    spmvn(mean, a, phi, omega, out = res)
    expect_false(length(sum(res == rep(0, m))) == length(res))
    # test for non-SPD a input
    aa <- diag(rnorm(ncol(phi)))
    expect_that(hpmvn(mean, aa, g, r), throws_error())
})


test_that("reproducability via seeding", {
    out <- hypertruncated_mvn_data()
    mean <- out$mean
    cov <- out$cov
    g <- out$g
    r <- out$r

    gen1 <- HTNGenerator(10)
    res1 <- gen1$hyperplane_truncated_mvnorm(mean, cov, g, r)
    gen2 <- HTNGenerator(10)
    res2 <- gen2$hyperplane_truncated_mvnorm(mean, cov, g, r)
    gen3 <- HTNGenerator(233)
    res3 <- gen3$hyperplane_truncated_mvnorm(mean, cov, g, r)
    expect_equal(res1, res2)
    expect_false(length(sum(res1 == res3)) == length(res1))
    # test if the errors are raised when the wrong input for seed is given
    expect_that(HTNGenerator(-100), throws_error("cannot be negative"))
    expect_that(HTNGenerator("100"), throws_error("non-numeric"))
})
