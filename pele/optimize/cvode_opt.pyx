"""
# distutils: language = C++
"""
import sys

import numpy as np

from pele.potentials import _pele
from pele.potentials cimport _pele
from pele.optimize import Result
from pele.potentials._pythonpotential import as_cpp_potential


cimport numpy as np
cimport pele.optimize._pele_opt as _pele_opt
from pele.optimize._pele_opt cimport shared_ptr
cimport cython
from cpython cimport bool as cbool


cdef extern from "pele/cvode.hpp" namespace "pele":
    cdef cppclass cppCVODEBDFOptimizer "pele::CVODEBDFOptimizer":
        cppCVODEBDFOptimizer(shared_ptr[_pele.cBasePotential], _pele.Array[double],
                             double, double, double) except +
        double get_nhev() except +


cdef class _Cdef_CVODEBDFOptimizer_CPP(_pele_opt.GradientOptimizer):
    """This class is the python interface for the c++ CVODEBDFOptimizer implementation
       Not quite an optimizer in the usual sense. Solves for the trajectory taken by
       the steepest descent path using the sundials solver CVODE_BDF. the solver terminates
       when it gets close to a minimum, just like an optimizer
    """
    cdef _pele.BasePotential pot
    def __cinit__(self, potential, x0,  double tol=1e-4, double rtol = 1e-4, double atol = 1e-4, int nsteps=10000):
        potential = as_cpp_potential(potential, verbose=True)

        self.pot = potential
        cdef np.ndarray[double, ndim=1] x0c = np.array(x0, dtype=float)
        self.thisptr = shared_ptr[_pele_opt.cGradientOptimizer]( <_pele_opt.cGradientOptimizer*>
                new cppCVODEBDFOptimizer(self.pot.thisptr,
                             _pele.Array[double](<double*> x0c.data, x0c.size),
                                tol, rtol, atol))
        cdef cppCVODEBDFOptimizer* mxopt_ptr = <cppCVODEBDFOptimizer*> self.thisptr.get()
    
    def get_result(self):
        cdef cppCVODEBDFOptimizer* mxopt_ptr = <cppCVODEBDFOptimizer*> self.thisptr.get()
        res = super(_Cdef_CVODEBDFOptimizer_CPP, self).get_result()
        res["nhev"] = float(mxopt_ptr.get_nhev())
        return res

class CVODEBDFOptimizer_CPP(_Cdef_CVODEBDFOptimizer_CPP):
    """This class is the python interface for the c++ LBFGS implementation
    """