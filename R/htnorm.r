# Copyright (c) 2020, Zolisa Bleki
# SPDX-License-Identifier: BSD-3-Clause

matrix_type <- c(0, 1, 2)


validate_output <- function(res) {

    if (res$info < 0) {
        stop("Possible illegal value in one of the inputs")
    }
    else if (res$info > 0) {
        stop(
            "Either the leading minor of the ", res$info, "'th order is not
            positive definite (meaning the covariance matrix is also not
            positive definite), or factorization of one of the inputs
            returned a factor with a zero in the ", res$info, "'th diagonal."
        )
    }
    else {
        res$out
    }
}


hptrunc_mvn <- function(rng, mean, cov, g, r, diag, out) {

    stopifnot(
        is.vector(mean),
        is.matrix(cov),
        is.matrix(g),
        is.vector(r),
        is.logical(diag),
        length(mean) == nrow(cov),
        nrow(cov) == ncol(cov),
        nrow(cov) == ncol(g)
    )

    if (is.null(out))
        out <- rep(0, length(mean))

    res <- .Call(C_hpmvn, rng, mean, cov, g, r, diag, out, PACKAGE = "htnorm")

    validate_output(res)
}


strprec_mvn <- function(rng, mean, a, phi, omega, str_mean, a_id, o_id, out) {

    stopifnot(
        is.vector(mean),
        is.matrix(a),
        is.matrix(phi),
        is.matrix(omega),
        is.logical(str_mean),
        nrow(omega) == ncol(omega),
        ncol(omega) == nrow(phi),
        length(mean) == nrow(a),
        nrow(a) == ncol(a)
    )

    if (!is.element(a_id, matrix_type) || !is.element(o_id, matrix_type))
        stop("`a_type` and `o_type` need to be one of {0, 1, 2}")

    a_id <- as.integer(a_id)
    o_id <- as.integer(o_id)

    if (is.null(out))
        out <- rep(0, length(mean))

    res <- .Call(
        C_spmvn,
        rng,
        mean,
        a,
        phi,
        omega,
        str_mean,
        a_id,
        o_id,
        out,
        PACKAGE = "htnorm"
    )

    validate_output(res)
}


#' Sample from a multivariate normal truncated on a hyperplane or a multivariate
#' normal with a structured precision. `HTNGenerator` returns an object that
#' can be used to sample from such a distribution.
#'
#' @param seed A random seed. It must be a positive integer. If not specified 
#'  then it defaults to NULL, which means a random seed is used.
#' @param gen The type of random number generator to use internally. It must
#'  either "xrs128p" (Xoroshiro128plus) or "pcg64" (PCG64). If not specified
#'  then this parameter defaults to "xrs128p".
#' @return A generator object that can be used to sample from the supported
#'  distributions.
#' @references Cong, Yulai; Chen, Bo; Zhou, Mingyuan. Fast Simulation of 
#'  Hyperplane-Truncated Multivariate Normal Distributions. Bayesian Anal. 
#'  12 (2017), no. 4, 1017--1037. doi:10.1214/17-BA1052.
#'  https://projecteuclid.org/euclid.ba/1488337478
#'
#' @examples
#' mean <- rnorm(1000)
#' cov <- matrix(rnorm(1000 * 1000), ncol=1000)
#' cov <- cov %*% t(cov)
#' G <- matrix(rep(1, 1000), ncol=1000)
#' r <- c(0)
#' rng <- HTNGenerator(seed=12345, gen="pcg64")
#' samples <- rng$hyperplane_truncated_mvnorm(mean, cov, G, r)
#' # verify if sampled values sum to zero
#' sum(samples)
#'
#' out <- rep(0, 1000)
#' eig <- eigen(cov)
#' phi <- eig$vectors
#' omega <- diag(eig$values)
#' a <- diag(runif(length(mean)))
#' rng$structured_precision_mvnorm(mean, a, phi, omega, a_type = 1, out = out)
HTNGenerator <- function(seed = NULL, gen = "xrs128p") {

    if (is.numeric(seed)) {
        if (seed < 0)
            stop("`seed` cannot be negative.")
        seed <- as.integer(seed)
        if (!is.finite(seed))
            stop("`seed` cannot be converted to an integer")
    }
    else if (!is.null(seed)) {
        stop("`seed` cannot be a non-numeric value")
    }

    gen <- switch(gen, "xrs128p" = as.integer(0), "pcg64" = as.integer(1))
    if (is.null(gen))
        stop("`gen` needs to one of {'xrs128p', 'pcg64'}")

    res <- list("rng" = .Call(C_get_rng, seed, gen, PACKAGE = "htnorm"))
    class(res) <- "HTNGenerator"

    res$hyperplane_truncated_mvnorm <- function(
        mean, cov, g, r, diag = FALSE, out = NULL
    ) {
        if (is.null(out)) {
            hptrunc_mvn(res$rng, mean, cov, g, r, diag, out)
        }
        else {
            invisible(hptrunc_mvn(res$rng, mean, cov, g, r, diag, out))
        }
    }

    res$structured_precision_mvnorm <- function(
        mean, a, phi, omega, str_mean = FALSE, a_type = 0, o_type = 0, out= NULL
    ) {
        if (is.null(out)) {
            strprec_mvn(
                res$rng, mean, a, phi, omega, str_mean, a_type, o_type, out
            )
        }
        else {
            invisible(
                strprec_mvn(
                    res$rng, mean, a, phi, omega, str_mean, a_type, o_type, out
                )
            )
        }
    }

    res
}
