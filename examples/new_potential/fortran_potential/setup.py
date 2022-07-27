from numpy.distutils.core import Extension, setup

setup(ext_modules=[Extension("_mypotential", ["_mypotential.f90"])])
