from distutils.core import Extension
import os

import numpy as np


source_files = [
    "pyhtnorm/_htnorm.c",
    "src/htnorm_rng.c",
    "src/htnorm_distributions.c",
    "src/htnorm.c"
]

# get environmental variables to determine the flow of the build process
BUILD_WHEELS = os.getenv("BUILD_WHEELS", None)
LIBS_DIR = os.getenv("LIBS_DIR", '/usr/lib')

libraries = ['m']

# when building manylinux2014 wheels for pypi use different directories as
# required by CentOS, else allow the user to specify them when building from
# source distribution
if BUILD_WHEELS:
    library_dirs = ['/usr/lib64']
    libraries.append('openblas')
else:
    library_dirs = [LIBS_DIR]
    libraries.extend(['blas', 'lapack'])


extensions = [
    Extension(
        "_htnorm",
        source_files,
        include_dirs=[np.get_include(), './include', 'src'],
        library_dirs=library_dirs,
        libraries=libraries,
        define_macros=[('NPY_NO_DEPRECATED_API', 0)],
        extra_compile_args=['-std=c99']
    )
]


def build(setup_kwargs):
    """Build extension modules."""
    kwargs = dict(ext_modules=extensions, zip_safe=False)
    setup_kwargs.update(kwargs)
