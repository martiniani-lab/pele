"""
# distutils: language = C++
"""
import numpy as np
from ctypes import c_size_t as size_t
from libcpp cimport bool as cbool
cimport numpy as np

cimport pele.potentials._pele as _pele
from pele.potentials._pele cimport shared_ptr



#===============================================================================
# Interface to the horrid HP for Mcpele
#===============================================================================



cdef extern from "pele/HPModel.h" namespace "pele":
    cdef cppclass cHPModel "pele::HPModel":
        cHPModel(char*) except +



cdef class HP(_pele.BasePotential):
    cdef cHPModel* newptr
    def __cinit__(self, path):
        self.thisptr = _pele.shared_ptr[_pele.cBasePotential] (<_pele.cBasePotential*> new cHPModel(path))