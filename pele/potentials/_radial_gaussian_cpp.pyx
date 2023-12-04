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
        cBaseRadialGaussian(_pele.Array[double] coords, double k, double l0, double log_prefactor, double r_cutoff, size_t ndim) except +
        void set_k(double) except +
        double get_k() except +
        void set_l0(double) except +
        double get_l0() except +
        void set_log_prefactor(double) except +
        double get_log_prefactor() except +
        void set_r_cutoff(double) except +
        double get_r_cutoff() except +
    cdef cppclass cRadialGaussian "pele::RadialGaussian":
        cRadialGaussian(_pele.Array[double] coords, double k, double l0, double log_prefactor, double r_cutoff, size_t ndim) except +
    cdef cppclass cRadialGaussianCOM "pele::RadialGaussianCOM":
        cRadialGaussianCOM(_pele.Array[double] coords, double k, double l0, double log_prefactor, double r_cutoff, size_t ndim) except +

cdef class RadialGaussian(_pele.BasePotential):
    """define the python interface to the c++ RadialGaussian potential implementation
    """
    cdef cBaseRadialGaussian* newptr
    cdef origin
    cdef double k
    cdef double l0
    cdef double log_prefactor
    cdef double r_cutoff
    cdef cbool com
    cdef int bdim
    
    def __cinit__(self, coords, k, l0, log_prefactor = 1.0, r_cutoff = 1.0, bdim=3, com=False):
        
        cdef np.ndarray[double, ndim=1] corigin = coords
        self.k = k
        self.l0 = l0
        self.log_prefactor = log_prefactor
        self.r_cutoff = r_cutoff
        self.com = com
        self.bdim = bdim
        if self.com is True:
            self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new cRadialGaussianCOM(_pele.Array[double](<double*> corigin.data, corigin.size), 
                                                                   self.k, self.l0, self.log_prefactor, self.r_cutoff, self.bdim) )
        else:
            self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new cRadialGaussian(_pele.Array[double](<double*> corigin.data, corigin.size), 
                                                                self.k, self.l0, self.log_prefactor, self.r_cutoff, self.bdim) )
            
        self.origin = corigin
        self.newptr = <cBaseRadialGaussian*> self.thisptr.get()
        
    def set_k(self, newk):
        self.k = newk
        self.newptr.set_k(newk)
    
    def set_l0(self, newl0):
        self.l0 = newl0
        self.newptr.set_l0(newl0)
        
    def set_log_prefactor(self, newlog_prefactor):
        self.log_prefactor = newlog_prefactor
        self.newptr.set_log_prefactor(newlog_prefactor)
        
    def set_r_cutoff(self, newr_cutoff):
        self.r_cutoff = newr_cutoff
        self.newptr.set_r_cutoff(newr_cutoff)
        
    def get_k(self):
        k = self.newptr.get_k()
        return k
        
    def get_l0(self):
        l0 = self.newptr.get_l0()
        return l0
    
    def get_log_prefactor(self):
        log_prefactor = self.newptr.get_log_prefactor()
        return log_prefactor
    
    def get_r_cutoff(self):
        r_cutoff = self.newptr.get_r_cutoff()
        return r_cutoff
    
    def __reduce__(self):
        return (RadialGaussian,(self.origin, self.k, self.l0, self.bdim, self.com))
        