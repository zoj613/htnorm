# htnorm

This repo provides a C implementation of a fast and exact sampling algorithm for a 
multivariate normal distribution (MVN) truncated on a hyperplane as described [here][1]

this repo implements the following from the paper:

- efficient Sampling from a MVN truncated on a hyperplane: 

    ![hptrunc](https://latex.codecogs.com/svg.latex?%5Cmathbf%7Bx%7D%20%5Csim%20%5Cmathcal%7BN%7D_%7B%5Cmathcal%7BS%7D%7D%28%5Cmathbf%7B%5Cmu%7D%2C%20%5Cmathbf%7B%5CSigma%7D%29%3B%20%5Chspace%7B2mm%7D%20%5Cmathcal%7BS%7D%20%3D%20%5C%7B%5Cmathbf%7Bx%7D%20%3A%20%5Cmathbf%7BG%7D%5Cmathbf%7Bx%7D%20%3D%20%5Cmathbf%7Br%7D%5C%7D%2C%20%5Cmathbf%7BG%7D%20%5Cin%20%5Cmathcal%7BR%7D%5E%7Bk_2%20%5Ctimes%20k%7D%2C%20rank%28%5Cmathbf%7BG%7D%29%20%3D%20k_2%20%3C%20k)

- efficient sampling from a MVN with a stuctured precision matrix that is a sum of an invertible matrix and a row rank matrix: 

    ![struc](https://latex.codecogs.com/svg.latex?%5Cmathbf%7Bx%7D%20%5Csim%20%5Cmathcal%7BN%7D%5C%5B%5Cmathbf%7B%5Cmu%7D%2C%20%28%5Cmathbf%7BA%7D%20&plus;%20%5Cmathbf%7B%5CPhi%7D%5ET%5Cmathbf%7B%5COmega%7D%5Cmathbf%7B%5CPhi%7D%29%5E%7B-1%7D%5C%5D%3B%20%5Chspace%7B2mm%7D%20%5Cmathbf%7B%5CPhi%7D%20%5Cin%20%5Cmathcal%7BR%7D%5E%7Bn%20%5Ctimes%20p%7D%2C%20%5Cmathbf%7B%5COmega%7D%20%5Cin%20%5Cmathcal%7BR%7D%5E%7Bn%20%5Ctimes%20n%7D%2C%20%5Cmathbf%7BA%7D%20%5Cin%20%5Cmathcal%7BR%7D%5E%7Bp%20%5Ctimes%20p%7D)

- efficent sampling frfom a MVN with a structured precision and mean:

    ![strucmean](https://latex.codecogs.com/svg.latex?%5Cmathbf%7Bx%7D%20%5Csim%20%5Cmathcal%7BN%7D%5CBig%5C%5B%28%5Cmathbf%7BA%7D%20&plus;%20%5Cmathbf%7B%5CPhi%7D%5ET%5Cmathbf%7B%5COmega%7D%5Cmathbf%7B%5CPhi%7D%29%5E%7B-1%7D%5Cmathbf%7B%5CPhi%7D%5ET%5Cmathbf%7B%5COmega%7D%5Cmathbf%7Bt%7D%2C%20%28%5Cmathbf%7BA%7D%20&plus;%20%5Cmathbf%7B%5CPhi%7D%5ET%5Cmathbf%7B%5COmega%7D%5Cmathbf%7B%5CPhi%7D%29%5E%7B-1%7D%5CBig%5C%5D%3B%20%5Chspace%7B2mm%7D%20%5Cmathbf%7B%5COmega%7D%20%5Cin%20%5Cmathcal%7BR%7D%5E%7Bn%20%5Ctimes%20n%7D%2C%20%5Cmathbf%7BA%7D%20%5Cin%20%5Cmathcal%7BR%7D%5E%7Bp%20%5Ctimes%20p%7D)

The algorithms implemented have the following practical applications:
- Topic models when unknown parameters can be interpreted as fractions.
- Admixture models
- discrete graphical models
- Sampling from posterior distribution of an Intrinsic Conditional Autoregressive prior [icar][8]
- Sampling from posterior conditional distributions of various bayesian regression problems.


## Dependencies

- A C compiler that implements the C99 standard or later
- An installation of `LAPACK`.

## Usage

Building a shared library of `htnorm` can be done with the following:
```bash
# optionally set path to LAPACK shared library
$ export LIBS_DIR="some/path/to/lib/"
$ make lib
```
Afterwards the shared library will be found in a `lib/` directory of the project root,
and the library can be linked dynamically via `-lhtnorm`.

The puplic API exposes the samplers through the function declarations
```C
 int htn_hyperplane_truncated_mvn(rng_t* rng, const ht_config_t* conf, double* out);
 int htn_structured_precision_mvn(rng_t* rng, const sp_config_t* conf, double* out);
```

The details of the parameters are documented in ther header files ["htnorm.h"][4].

Random number generation is done using [PCG64][2] or [Xoroshiro128plus][3] bitgenerators. 
The API allows using a custom generator, and the details are documented in the header file 
["rng.h"][5].

## Examples
```C
#include "htnorm.h"

int main (void)
{
    ...
    // instantiate a random number generator
    rng_t* rng = rng_new_pcg64_seeded(12345);
    ht_config_t config;
    init_ht_config(&config, ...);
    double* out = ...; // array to store the samples
    int res = htn_hyperplane_truncated_mvn(rng, &config, out);
    // res contains a number that indicates whether sampling failed or not.
    ...
    // finally free the RNG pointer at some point
    rng_free(rng);
    ...
    return 0;
}
```

## Python API

A high level python interface to the library is also provided. Linux users can 
install it using wheels via pip (thus not needing to worry about availability of C libraries),
```bash
pip install pyhtnorm
```
Wheels are not provided for MacOS. To install via pip, one can run the following commands:
```bash
#set the path to LAPACK shared library
export LIBS_DIR=<some directory>
pip install pyhtnorm
```
Alternatively, one can install it from source. This requires an installation of [poetry][7] and the following shell commands:

```bash
$ git clone https://github.com/zoj613/htnorm.git
$ cd htnorm/
$ poetry install
# add htnorm to python's path
$ export PYTHONPATH=$PWD:$PYTHONPATH
```

Below is an example of how to use htnorm in python to sample from a multivariate
gaussian truncated on the hyperplane ![sumzero](https://latex.codecogs.com/svg.latex?%5Cmathbf%7B1%7D%5ET%5Cmathbf%7Bx%7D%20%3D%200) (i.e. making sure the sampled values sum to zero)

```python
from pyhtnorm import HTNGenerator
import numpy as np

rng = HTNGenerator()

# generate example input
k1 = 1000
k2 = 1
npy_rng = np.random.default_rng()
temp = npy_rng.random((k1, k1))
cov = temp @ temp.T + np.diag(npy_rng.random(k1))
G = np.ones((k2, k1))
r = np.zeros(k2)
mean = npy_rng.random(k1)

samples = rng.hyperplane_truncated_mvnorm(mean, cov, G, r)
print(sum(samples))  # verify if sampled values sum to zero
# alternatively one can pass an array to store the results in
out = np.empty(k1)
rng.hyperplane_truncated_mvnorm(mean, cov, G, r, out=out)
print(out.sum())  # verify
```

For more details about the parameters of the `HTNGenerator` and its methods,
see the docstrings via python's `help` function.

The python API also exposes the `HTNGenerator` class as a Cython extension type
that can be "cimported" in a cython script.

A pure numpy implementation is demonstrated in this [example script][9].


## R API

One can also access the API in R. To install the package, use one the following 
commands:
```R
devtools::install_github("zoj613/htnorm")
pak::pkg_install("zoj613/htnorm")
```
Below is an R translation of the above python example:

```R
library(htnorm)

# make dummy data
mean <- rnorm(1000)
cov <- matrix(rnorm(1000 * 1000), ncol=1000)
cov <- cov %*% t(cov)
G <- matrix(rep(1, 1000), ncol=1000)
r <- c(0)
# initialize the Generator instance
rng <- HTNGenerator(seed=12345, gen="pcg64")
samples <- rng$hyperplane_truncated_mvnorm(mean, cov, G, r)
#verify if sampled values sum to zero
sum(samples)

# alternatively one can pass a vector to store the results in
out <- rep(0, 1000)
rng$hyperplane_truncated_mvnorm(mean, cov, G, r, out = out)
sum(out)  #verify
```

## Licensing

`htnorm` is free software made available under the BSD-3 License. For details
see the [LICENSE][6] file.


## References
- Cong, Yulai; Chen, Bo; Zhou, Mingyuan. Fast Simulation of Hyperplane-Truncated 
   Multivariate Normal Distributions. Bayesian Anal. 12 (2017), no. 4, 1017--1037. 
   doi:10.1214/17-BA1052.
- Bhattacharya, A., Chakraborty, A., and Mallick, B. K. (2016). 
  “Fast sampling with Gaussian scale mixture priors in high-dimensional regression.” 
  Biometrika, 103(4):985. 


[1]: https://projecteuclid.org/euclid.ba/1488337478
[2]: https://www.pcg-random.org/
[3]: https://en.wikipedia.org/wiki/Xoroshiro128%2B
[4]: https://github.com/zoj613/htnorm/blob/main/include/htnorm.h 
[5]: https://github.com/zoj613/htnorm/blob/main/include/rng.h
[6]: https://github.com/zoj613/htnorm/blob/main/LICENSE
[7]: https://python-poetry.org/docs/pyproject/
[8]: https://www.sciencedirect.com/science/article/abs/pii/S1877584517301600
[9]: https://github.com/zoj613/htnorm/blob/main/examples/numpy_implementation.py
