# htnorm

This repo provides a C implementation of a fast and exact sampler from a 
multivariate normal distribution (MVN) truncated on a hyperplane as described [here][1]

this repo implements the following from the paper:

- efficient Sampling from a MVN truncated on a hyperplane: 

    ![hptrunc](https://latex.codecogs.com/svg.latex?%5Cmathbf%7Bx%7D%20%5Csim%20%5Cmathcal%7BN%7D_%7B%5Cmathcal%7BS%7D%7D%28%5Cmathbf%7B%5Cmu%7D%2C%20%5Cmathbf%7B%5CSigma%7D%29%3B%20%5Chspace%7B2mm%7D%20%5Cmathcal%7BS%7D%20%3D%20%5C%7B%5Cmathbf%7Bx%7D%20%3A%20%5Cmathbf%7BG%7D%5Cmathbf%7Bx%7D%20%3D%20%5Cmathbf%7Br%7D%5C%7D%2C%20%5Cmathbf%7BG%7D%20%5Cin%20%5Cmathcal%7BR%7D%5E%7Bk_2%20%5Ctimes%20k%7D%2C%20rank%28%5Cmathbf%7BG%7D%29%20%3D%20k_2%20%3C%20k)

- efficient sampling from a MVN with a stuctured precision matrix: 

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

- a C compiler that supports the C99 standard or later
- an installation of BLAS and LAPACK that exposes its C interface via the headers `<cblas.h>` and `<lapacke.h>`
(e.g openBLAS).


## Usage

Building a shared library of `htnorm` can be done with the following:
```bash
$ cd src/
# optionally set path to CBLAS and LAPACKE headers using INCLUDE_DIRS environmental variable
$ export INCLUDE_DIRS="some/path/to/headers" 
# optionally set path to BLAS installation shared library
$ export LIBS_DIR="some/path/to/library/"
# optionally set the linker flag for your BLAS installation (e.g -lopenblas)
$ export LIBS=<flag here>
$ make lib
```
Afterwards the shared library will be found in a `lib/` directory of the project root,
and the library can be linked dynamically via `-lhtnorm`.

The puplic API exposes the samplers through the function declarations
- `int htn_hyperplane_truncated_mvn(rng_t* rng, const ht_config_t* conf, double* out)`
- `int htn_structured_precision_mvn(rng_t* rng, const sp_config_t* conf, double* out)`

The details of the parameters are documented in ther header files ["htnorm.h"][4].

Random number generation is done using [PCG64][2] or [Xoroshiro128plus][3] bitgenerators. 
The API allows using a custom generator, and the details are documented in the header file 
["rng.h"][5].

## Examples
```C
#include "htnorm.h"

int main ()
{
    ...

    // instantiate a random number generator
    rng_t* rng = rng_new_pcg64();
    ht_config_t config;
    config.g = ...; // G matrix
    config.gnrow = ...; // number of rows of G
    config.gncol = ...; // number of columns of G
    cofig.r = ...; // r array
    config.mean = ...; // mean array
    config.cov = ...; // the convariance matrix
    confi.diag = ...; // whether covariance is diagonal

    double* samples = ...; // array to store the samples
    // now call the sampler
    int res_info = htn_hyperplane_truncated_mvn(rng, &config, samples);

    // res_info contains a number that indicates whether sampling failed or not.

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
#set the path to BLAS installation headers
export INCLUDE_DIR=<path/to/headers>
#set the path to BLAS shared library
export LIBS_DIR=<some directory>
#set the name of the BLAS shared library (e.g. "openblas")
export LIBS=<lib name>
# finally install via pip so the compilation and linking can be done correctly
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
# verify if sampled values sum to zero
print(sum(samples))

# alternatively one can pass an array to store the results in
out = np.empty(k1)
rng.hyperplane_truncated_mvnorm(mean, cov, G, r, out=out)
# verify
print(out.sum())
```

For more details about the parameters of the `HTNGenerator` and its methods,
see the docstrings via python's `help` function.

The python API also exposes the `HTNGenerator` class as a Cython extension type
that can be "cimported" in a cython script.


## TODO

- Add an `R` interface to the library.


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
