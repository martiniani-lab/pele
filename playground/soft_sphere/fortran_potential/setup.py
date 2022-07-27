from numpy.distutils.core import Extension, setup

setup(ext_modules=[Extension("_soft_sphere", ["_soft_sphere.f90"])])
