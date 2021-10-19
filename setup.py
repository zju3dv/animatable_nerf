try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import numpy


# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# Extensions
# pykdtree (kd tree)
pykdtree = Extension(
    'lib.csrc.libkdtree.pykdtree.kdtree',
    sources=[
        'lib/csrc/libkdtree/pykdtree/kdtree.c',
        'lib/csrc/libkdtree/pykdtree/_kdtree_core.c'
    ],
    language='c',
    extra_compile_args=['-std=c99', '-O3', '-fopenmp'],
    extra_link_args=['-lgomp'],
)

# mise (efficient mesh extraction)
mise_module = Extension(
    'lib.csrc.libmise.mise',
    sources=[
        'lib/csrc/libmise/mise.pyx'
    ],
)

# Gather all extension modules
ext_modules = [
    pykdtree,
    mise_module
]

setup(
    ext_modules=cythonize(ext_modules),
    cmdclass={
        'build_ext': BuildExtension
    }
)
