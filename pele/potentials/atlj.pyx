"""
# distutils: language = C++
"""


import numpy as np
from ctypes import c_size_t as size_t

from pele.potentials import FrozenPotentialWrapper

cimport numpy as np
from cpython cimport bool

cimport pele.potentials._pele as _pele
from pele.potentials._pele cimport shared_ptr


cdef extern from "pele/atlj.hpp" namespace "pele":
    cdef cppclass  cATLJ "pele::ATLJ":
        cATLJ(double sig, double eps, double Z) except +



cdef class ATLJ(_pele.BasePotential):
    """define the python interface to the c++ LJ implementation
    """
    def __cinit__(self, eps=1.0, sig=1.0, Z=2):
        self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new cATLJ(sig, eps, Z) )
