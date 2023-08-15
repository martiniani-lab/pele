import numpy as np

from ctypes import c_size_t as size_t
from libcpp cimport bool as cbool
cimport numpy as np

cimport pele.potentials._pele as _pele
from pele.potentials._pele cimport shared_ptr
from pele.potentials._pele cimport array_wrap_np, array_wrap_np_size_t


cdef extern from "pele/rosenbrock.hpp" namespace "pele":
    cdef cppclass cNegativeCosProduct "pele::NegativeCosProduct":
        cNegativeCosProduct(size_t dim, double period) except +
        
        
        
cdef class _Cdef_NegativeCosProduct(_pele.BasePotential):
    """Define the python interface to the c++ NegativeCosProduct potential implementation
    """
    cdef int cdim
    cdef double cperiod
    
    
    def __cinit__(self, int dim, double period):
        self.cdim = dim
        self.cperiod = period
        self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new cNegativeCosProduct(cdim, period))
    
    def __reduce__(self):
        return (NegativeCosProduct, (self.cdim, self.cperiod))
    
    
class NegativeCosProduct(_Cdef_NegativeCosProduct):
    """Python Interface to the c++ NegativeCosProduct potential implementation
    """