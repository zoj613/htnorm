from distutils.core import Extension
import os


source_files = [
    "pyhtnorm/_htnorm.pyx",
    "src/splitmax64.c",
    "src/pcg32_minimal.c",
    "src/xoroshiro128p.c",
    "src/pcg64.c",
    "src/rng.c",
    "src/dist.c",
    "src/htnorm.c"
]

# get environmental variables to determine the flow of the build process
BUILD_WHEELS = os.getenv("BUILD_WHEELS", None)
INCLUDE_DIR = os.getenv("INCLUDE_DIR", None)
LIBS_DIR = os.getenv("LIBS_DIR", '/usr/lib')
LIBS = os.getenv("LIBS", 'openblas')

# necessary directories
include_dirs = ['/usr/include', '.include']
libraries = ['m']

# when building manylinux2014 wheels for pypi use different directories as
# required by CentOS, else allow the user to specify them when building from
# source distribution
if BUILD_WHEELS:
    include_dirs.append('/usr/include/openblas/')
    library_dirs = ['/lib64/', '/usr/lib64']
    libraries.append('openblasp')
else:
    if INCLUDE_DIR:
        include_dirs.append(INCLUDE_DIR)
    library_dirs = [LIBS_DIR]
    libraries.append(LIBS)


extensions = [
    Extension(
        "pyhtnorm._htnorm",
        source_files,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=['-std=c11']
    )
]


def build(setup_kwargs):
    """Build extension modules."""
    kwargs = dict(ext_modules=extensions, zip_safe=False)
    setup_kwargs.update(kwargs)
