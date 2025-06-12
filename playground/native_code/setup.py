import os
import numpy as np
from setuptools import setup, Extension

## Numpy header files
numpy_lib = os.path.split(np.__file__)[0]
numpy_include = os.path.join(numpy_lib, "core/include")
include_dirs = [numpy_include, "include/"]

include_sources = [
    #               "include/array.h",
    #               "include/potential.h",
    #               "include/simple_pairwise_potential.h",
    "include/_lbfgs.cpp",
]

extra_compile_args = [
    "-Wextra",
    "-pedantic",
    "-funroll-loops",
    "-O3",
    "-march=native",
    "-mtune=native",
    "-DNDEBUG",
]

cxx_modules = [
    Extension(
        "_lj",
        ["_lj.cpp"] + include_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        language="c++",
    ),
    Extension(
        "_pele",
        ["_pele.cpp"] + include_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "_lbfgs",
        ["_lbfgs.cpp"] + include_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "_pythonpotential",
        ["_pythonpotential.cpp"] + include_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
    ),
]

# NOTE: Building Fortran extensions with setuptools requires a tool like f2py.
# This example is configured for C++ extensions. The main project's
# setup_with_cmake.py handles Fortran compilation.
cxx_f_modules = [
    Extension(
        "_lj_cython",
        ["_lj_cython.cpp", "../../pele/potentials/fortran/lj.f90"]
        + include_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
    ),
]


setup(
    ext_modules=cxx_modules + cxx_f_modules,
)
