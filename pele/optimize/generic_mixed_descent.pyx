"""
# distutils: language = C++
"""


from multiprocessing.dummy import Array
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




cdef extern from "pele/generic_mixed_descent.hpp" namespace "pele":
    cdef cppclass cppGenericMixedDescent "pele::GenericMixedDescent":
        cppGenericMixedDescent(shared_ptr[_pele.cBasePotential], _pele.Array[double], 
                               shared_ptr[_pele.cGradientOptimizer],
                               shared_ptr[_pele.cGradientOptimizer],
                               double, int) except +
        double get_nhev() except +
        int get_n_phase_1_steps() except +
        int get_n_phase_2_steps() except +




cdef class _Cdef_MixedOptimizer_CPP(_pele_opt.GradientOptimizer):
    """This class is the python interface for the c++ MixedOptimizer implementation
    """
    cdef _pele.BasePotential pot
    cdef _pele.GradientOptimizer opt_conv
    cdef _pele.GradientOptimizer opt_non_conv
    def __cinit__(self, potential, x0,  optimizer_convex,
                  optimizer_non_convex, double translation_offset,
                  int steps_before_convex_check=1):
        
        potential = as_cpp_potential(potential, verbose=True)
        self.pot = potential
        opt_conv = optimizer_convex
        opt_non_conv = optimizer_non_convex
        cdef np.ndarray[double, ndim=1] x0c = np.array(x0, dtype=float)
        self.thisptr = shared_ptr[_pele_opt.cGradientOptimizer]( <_pele_opt.cGradientOptimizer*>
                new cppMixedOptimizer(self.pot.thisptr,
                             _pele.Array[double](<double*> x0c.data, x0c.size), 
                             opt_conv.thisptr,
                             opt_non_conv.thisptr,
                             offset,
                             steps_before_convex_check
                             ))
        cdef cppMixedOptimizer* mxopt_ptr = <cppMixedOptimizer*> self.thisptr.get()

    def get_result(self):
        cdef cppMixedOptimizer* mxopt_ptr = <cppMixedOptimizer*> self.thisptr.get()
        res = super(_Cdef_MixedOptimizer_CPP, self).get_result()
        res["nhev"] = float(mxopt_ptr.get_nhev())
        res["n_phase_1"] = int(mxopt_ptr.get_n_phase_1_steps())
        res["n_phase_2"] = int(mxopt_ptr.get_n_phase_2_steps())
        return res

class MixedOptimizer_CPP(_Cdef_MixedOptimizer_CPP):
    """This class is the python interface for the c++ LBFGS implementation
    """