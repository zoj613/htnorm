from distutils.core import Extension


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


extensions = [
    Extension(
        "pyhtnorm._htnorm",
        source_files,
        include_dirs=['/usr/include/', 'include'],
        library_dirs=['/usr/lib'],
        libraries=['cblas', 'lapacke', 'm'],
    )
]


def build(setup_kwargs):
    """Build extension modules."""
    kwargs = dict(ext_modules=extensions, zip_safe=False)
    setup_kwargs.update(kwargs)
