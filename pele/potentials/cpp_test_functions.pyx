"""
# distutils: language = C++
"""
import numpy as np

from ctypes import c_size_t as size_t
from libcpp cimport bool as cbool
cimport numpy as np

cimport pele.potentials._pele as _pele
from pandas.core.arrays import period_array
from pele.potentials._pele cimport shared_ptr
from pele.potentials._pele cimport array_wrap_np, array_wrap_np_size_t, pele_array_to_np


cdef extern from "pele/rosenbrock.hpp" namespace "pele":
    cdef cppclass cPoweredCosineSum "pele::PoweredCosineSum":
        cPoweredCosineSum(size_t dim, _pele.Array[double] periods, _pele.Array[double] prefactors,
                          double cpower, double coffset) except +
        
        
        
cdef class _Cdef_PoweredCosineSum(_pele.BasePotential):
    """Define the python interface to the c++ PoweredCosineSum potential implementation
    """
    cdef int cdim
    cdef _pele.Array[double] cperiods
    cdef _pele.Array[double] cprefactors
    cdef double cpower
    cdef double coffset

    def __cinit__(self, int dim, period, double power=0.5, double offset = 1.0, prefactors=None):
        self.cdim = dim
        if isinstance(period, float):
            period_array = np.full(dim, period)
        else:
            period_array = period
        if prefactors is None:
            prefactors_array = np.ones(dim)
        else:
            prefactors_array = prefactors
        self.cperiods = array_wrap_np(period_array)
        self.cprefactors = array_wrap_np(prefactors_array)
        self.cpower = power
        self.coffset = offset
        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*>new cPoweredCosineSum(
            self.cdim, self.cperiods, self.cprefactors, self.cpower, self.coffset))
    
    def __reduce__(self):
        return (PoweredCosineSum, (self.cdim, pele_array_to_np(self.cperiods), pele_array_to_np(self.cprefactors),
                                   self.cpower, self.coffset))
    
    
class PoweredCosineSum(_Cdef_PoweredCosineSum):
    """Python Interface to the c++ PoweredCosineSum potential implementation
    """