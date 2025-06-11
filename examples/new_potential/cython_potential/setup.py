from setuptools import setup, Extension
import numpy as np


setup(
    ext_modules=[Extension("mypotential", ["mypotential.c"], include_dirs=[np.get_include()])],
)
