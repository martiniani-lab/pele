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
from libcpp cimport bool
from libcpp.vector cimport vector

# import the externally defined gradient descent implementation
cdef extern from "pele/gradient_descent.hpp" namespace "pele":
    cdef cppclass cppGradientDescent "pele::GradientDescent":
        cppGradientDescent(shared_ptr[_pele.cBasePotential], _pele.Array[double], double, double, bool) except +
        void set_tol(double) except +
        void set_maxstep(double) except +
        void set_max_iter(int) except +
        void set_iprint(int) except +
        void set_verbosity(int) except +
        vector[double] get_time_trajectory() except +
        vector[double] get_gradient_norm_trajectory() except +


cdef class _Cdef_GradientDescent_CPP(_pele_opt.GradientOptimizer):
    """This class is the python interface for the c++ gradient descent implementation
    """
    cdef _pele.BasePotential pot
    cdef bool save_trajectory

    def __cinit__(self, potential, x0, double tol=1e-5, int iprint=-1,
                  energy=None, gradient=None,
                  int nsteps=10000, int verbosity=0, events=None, logger=None, double stepsize = 0.005, bool save_trajectory=False):
        potential = as_cpp_potential(potential, verbose=verbosity>0)
        self.pot = potential
        self.save_trajectory = save_trajectory
        if logger is not None:
            print "warning c++ gradient descent is ignoring logger"
        cdef np.ndarray[double, ndim=1] x0c = np.array(x0, dtype=float)
        self.thisptr = shared_ptr[_pele_opt.cGradientOptimizer]( <_pele_opt.cGradientOptimizer*>
                new cppGradientDescent(self.pot.thisptr,
                             _pele.Array[double](<double*> x0c.data, x0c.size),
                             tol, stepsize, save_trajectory))
        cdef cppGradientDescent* gd_ptr = <cppGradientDescent*> self.thisptr.get()
        gd_ptr.set_max_iter(nsteps)
        gd_ptr.set_verbosity(verbosity)
        gd_ptr.set_iprint(iprint)

        cdef np.ndarray[double, ndim=1] g_
        if energy is not None and gradient is not None:
            g_ = gradient
            self.thisptr.get().set_func_gradient(energy, _pele.Array[double](<double*> g_.data, g_.size))

        self.events = events
        if self.events is None:
            self.events = []

    def get_result(self):
        cdef cppGradientDescent* gd_ptr = <cppGradientDescent*> self.thisptr.get()
        result = super(_Cdef_GradientDescent_CPP, self).get_result()
        if self.save_trajectory:
            result.time_trajectory = np.array(gd_ptr.get_time_trajectory())
            result.gradient_norm_trajectory = np.array(gd_ptr.get_gradient_norm_trajectory())
        return result


class GradientDescent_CPP(_Cdef_GradientDescent_CPP):
    """This class is the python interface for the c++ gradient descent implementation
    """