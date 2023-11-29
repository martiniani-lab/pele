# distutils: language = c++
import numpy as np

from ctypes import c_size_t as size_t
from libcpp cimport bool as cbool
cimport numpy as np

cimport pele.potentials._pele as _pele
from pele.potentials._pele cimport shared_ptr

#===============================================================================
# THIS POTENTIAL NEEDS TO BE CLEANED UP
#===============================================================================

# use external c++ class
cdef extern from "pele/radial_gaussian.hpp" namespace "pele":
    cdef cppclass cBaseRadialGaussian "pele::BaseRadialGaussian":
        cBaseRadialGaussian(_pele.Array[double] coords, double k, double l0, size_t ndim) except +
        void set_k(double) except +
        double get_k() except +
        void set_l0(double) except +
        double get_l0() except +
    cdef cppclass cRadialGaussian "pele::RadialGaussian":
        cRadialGaussian(_pele.Array[double] coords, double k, double l0, size_t ndim) except +
    cdef cppclass cRadialGaussianCOM "pele::RadialGaussianCOM":
        cRadialGaussianCOM(_pele.Array[double] coords, double k, double l0, size_t ndim) except +

cdef class RadialGaussian(_pele.BasePotential):
    """define the python interface to the c++ RadialGaussian potential implementation
    """
    cdef cBaseRadialGaussian* newptr
    cdef origin
    cdef double k
    cdef double l0
    cdef cbool com
    cdef int bdim
    
    def __cinit__(self, coords, k, l0, bdim=3, com=False):
        
        cdef np.ndarray[double, ndim=1] corigin = coords
        self.k = k
        self.l0 = l0
        self.com = com
        self.bdim = bdim
        if self.com is True:
            self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new cRadialGaussianCOM(_pele.Array[double](<double*> corigin.data, corigin.size), 
                                                                   self.k, self.l0, self.bdim) )
        else:
            self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new cRadialGaussian(_pele.Array[double](<double*> corigin.data, corigin.size), 
                                                                self.k, self.l0, self.bdim) )
            
        self.origin = corigin
        self.newptr = <cBaseRadialGaussian*> self.thisptr.get()
        
    def set_k(self, newk):
        self.k = newk
        self.newptr.set_k(newk)
    
    def set_l0(self, newl0):
        self.l0 = newl0
        self.newptr.set_l0(newl0)
        
    def get_k(self):
        k = self.newptr.get_k()
        return k
        
    def get_l0(self):
        l0 = self.newptr.get_l0()
        return l0
    
    def __reduce__(self):
        return (RadialGaussian,(self.origin, self.k, self.l0, self.bdim, self.com))
        