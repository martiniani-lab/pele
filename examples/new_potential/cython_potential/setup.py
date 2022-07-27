from numpy.distutils.core import Extension, setup

setup(ext_modules=[Extension("mypotential", ["mypotential.c"])])
