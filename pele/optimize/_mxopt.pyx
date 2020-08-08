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

cdef extern from "pele/mxopt.h" namespace "pele":
    cdef cppclass cppMixedOptimizer "pele::MixedOptimizer":
        cppMixedOptimizer(shared_ptr[_pele.cBasePotential], _pele.Array[double], double, int,
                 double, double, double) except +
        void set_H0(double) except +
        double get_H0() except +


cdef class _Cdef_MixedOptimizer_CPP(_pele_opt.GradientOptimizer):
    """This class is the python interface for the c++ MixedOptimizer implementation
    """
    cdef _pele.BasePotential pot
    def __cinit__(self, x0, potential, double tol=1e-5, int T=1, double step=1, double conv_tol = 1e-2,
                  double conv_factor=2, int nsteps=10000):
        potential = as_cpp_potential(potential, verbose=True)

        self.pot = potential
        cdef np.ndarray[double, ndim=1] x0c = np.array(x0, dtype=float)
        self.thisptr = shared_ptr[_pele_opt.cGradientOptimizer]( <_pele_opt.cGradientOptimizer*>
                new cppMixedOptimizer(self.pot.thisptr,
                             _pele.Array[double](<double*> x0c.data, x0c.size),
                                tol, T, step, conv_tol, conv_factor))
        cdef cppMixedOptimizer* mxopt_ptr = <cppMixedOptimizer*> self.thisptr.get()
    
    def set_H0(self, H0):
        cdef cppMixedOptimizer* mxopt_ptr = <cppMixedOptimizer*> self.thisptr.get()
        mxopt_ptr.set_H0(float(H0))
    
    def get_result(self):
        cdef cppMixedOptimizer* mxopt_ptr = <cppMixedOptimizer*> self.thisptr.get()
        res = super(_Cdef_MixedOptimizer_CPP, self).get_result()
        res["H0"] = float(mxopt_ptr.get_H0())
        return res

class MixedOptimizer_CPP(_Cdef_MixedOptimizer_CPP):
    """This class is the python interface for the c++ LBFGS implementation
    """