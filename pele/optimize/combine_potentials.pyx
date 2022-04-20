"""
# distutils: language = C++
"""

cimport pele.potentials._pele as _pele

## combines potentials

cdef extern from "pele/combine_potentials.hpp"
from potentials.potential import BasePotential, potential namespace "pele":
    cdef cppclass cppCombinedPotential "pele::CombinedPotential":
        cppCombinedPotential() except +
        void add_potential(shared_ptr[_pele.cBasePotential]) except +
        
        
        
        
        
        
        
cdef class Combined potential(_pele.BasePotential):
    def __cinit__(self):
        self.thisptr = _pele.cppCombinedPotential()
        
    def add_potential(self, potential):
        (<cppCombinedPotential*>self.thisptr.get()).add_potential(potential.thisptr)