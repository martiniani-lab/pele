from setuptools import setup, Extension

# NOTE: Building Fortran extensions with setuptools requires a tool like f2py,
# which is typically invoked via numpy.f2py.
# This setup.py is a simplified example. The main project's setup_with_cmake.py
# handles the actual Fortran compilation.

setup(ext_modules=[Extension("_mypotential", ["_mypotential.f90"])])
