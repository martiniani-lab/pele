"""
# distutils: language = C++
"""
import numpy as np

from ctypes import c_size_t as size_t
from libcpp cimport bool as cbool
cimport numpy as np

cimport pele.potentials._pele as _pele
from pele.potentials._pele cimport shared_ptr
from pele.potentials._pele cimport array_wrap_np, array_wrap_np_size_t


cdef extern from "pele/rosenbrock.hpp" namespace "pele":
    cdef cppclass cPoweredCosineSum "pele::PoweredCosineSum":
        cPoweredCosineSum(size_t dim, double period, double cpower, double coffset) except +
        
        
        
cdef class _Cdef_PoweredCosineSum(_pele.BasePotential):
    """Define the python interface to the c++ PoweredCosineSum potential implementation
    """
    cdef int cdim
    cdef double cperiod
    cdef double cpower
    cdef double coffset
    
    
    
    def __cinit__(self, int dim, double period, double power=0.5, double offset = 0.0):
        self.cdim = dim
        self.cperiod = period
        self.cpower = power
        self.coffset = offset
        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*>new cPoweredCosineSum(self.cdim, self.cperiod, self.cpower, self.coffset))
    
    def __reduce__(self):
        return (PoweredCosineSum, (self.cdim, self.cperiod, self.cpower, self.coffset))
    
    
class PoweredCosineSum(_Cdef_PoweredCosineSum):
    """Python Interface to the c++ PoweredCosineSum potential implementation
    """