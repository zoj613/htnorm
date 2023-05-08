import platform
import os
from os.path import join
from setuptools import Extension, setup

import numpy as np
from numpy.distutils.misc_util import get_info


source_files = [
    "pyhtnorm/_htnorm.pyx",
    "src/htnorm_rng.c",
    "src/htnorm_distributions.c",
    "src/htnorm.c"
]

macros = [('NPY_NO_DEPRECATED_API', 0)]
if os.getenv("BUILD_WITH_COVERAGE", None):
    macros.append(('CYTHON_TRACE_NOGIL', 1))

if platform.system() == 'Windows':
    compile_args = ['/O2']
else:
    compile_args = ['-O2', '-std=c99']

# https://numpy.org/devdocs/reference/random/examples/cython/setup.py.html
include_path = np.get_include()
lib_dirs = [
    '/usr/lib',
    join(include_path, '..', '..', 'random', 'lib'),
    *get_info('npymath')['library_dirs'],
]
extensions = [
    Extension(
        "pyhtnorm._htnorm",
        sources=source_files,
        include_dirs=[include_path, "./include", "./src"],
        library_dirs=lib_dirs,
        libraries=['npyrandom', 'npymath', 'openblas', 'lapack'],
        define_macros=macros,
        extra_compile_args=compile_args,
    ),
]


setup(ext_modules=extensions)
