# distutils: language = c++
# distutils: sources = modified_fire.cpp
import numpy as np
from yaml import DocumentStartEvent

from pele.potentials import _pele
from pele.potentials._pythonpotential import as_cpp_potential

#cimport numpy as np
#cimport cython
#from libcpp cimport bool as cbool

from libcpp cimport bool

cdef class _Cdef_MODIFIED_FIRE_CPP(_pele_opt.GradientOptimizer):
    """This class is the python interface for the c++ MODIFIED_FIRE implementation
    """  
    cdef _pele.BasePotential pot
    cdef double[:] x0
    cdef float dtstart
    cdef float dtmax
    cdef float maxstep
    cdef int Nmin
    cdef double finc
    cdef double fdec
    cdef double fa
    cdef double astart
    cdef double tol
    cdef bool stepback
    cdef int iprint
    cdef int nsteps
    cdef int verbosity
    
    
    
    def __cinit__(self, x0, potential, double dtstart = 0.1, double dtmax = 1, double maxstep=0.5, size_t Nmin=5, double finc=1.1, 
                   double fdec=0.5, double fa=0.99, double astart=0.1, double tol=1e-3, bool stepback = True, 
                   int iprint=-1, energy=None, gradient=None, int nsteps=10000, int verbosity=0, events = None):
        
        self.dtstart = dtstart
        self.dtmax = dtmax
        self.maxstep = maxstep
        self.Nmin = Nmin
        self.finc = finc
        self.fdec = fdec
        self.fa = fa
        self.astart = astart
        self.tol = tol
        self.stepback = stepback
        self.iprint = iprint
        self.nsteps = nsteps
        self.verbosity = verbosity
        potential = as_cpp_potential(potential, verbose=verbosity>0)
        
        cdef _pele.BasePotential pot = potential
        cdef np.ndarray[double, ndim=1] x0c = np.array(x0, dtype=float)
        self.x0 = x0c
        self.thisptr = shared_ptr[_pele_opt.cGradientOptimizer]( <_pele_opt.cGradientOptimizer*>
                        new cppMODIFIED_FIRE(pot.thisptr, _pele.Array[double](<double*> x0c.data, x0c.size),
                                             dtstart, dtmax, maxstep, Nmin, finc, fdec, fa, astart, tol, stepback) )
        
        self.thisptr.get().set_max_iter(nsteps)
        self.thisptr.get().set_verbosity(verbosity)
        self.thisptr.get().set_iprint(iprint)
        self.pot = pot
        cdef np.ndarray[double, ndim=1] g_  
        if energy is not None and gradient is not None:
            g_ = gradient
            self.thisptr.get().set_func_gradient(energy, _pele.Array[double](<double*> g_.data, g_.size))

        self.events = events
        if self.events is None: 
            self.events = []
            
    def __reduce__(self):
        # energy, gradient and events are set to None, change if necessary later
        return (_Cdef_MODIFIED_FIRE_CPP, (self.x0, self.pot, self.dtstart, self.dtmax, self.maxstep, self.Nmin, self.finc, self.fdec, self.fa, self.astart, self.tol, self.stepback, self.iprint, None, None, self.nsteps, self.verbosity, None))

class ModifiedFireCPP(_Cdef_MODIFIED_FIRE_CPP):
    """This class is the python interface for the c++ MODIFED_FIRE implementation.
    """
    
#     def reset(self, coords):
#         """do one iteration"""
#         _Cdef_MODIFIED_FIRE_CPP.reset(self, coords)
       
