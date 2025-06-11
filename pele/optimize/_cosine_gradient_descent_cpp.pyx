# distutils: language = C++
# cython: language_level=3str
"""
Cython wrapper for Cosine Gradient Descent
"""

# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
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

# Import the external cosine gradient descent implementation
cdef extern from "pele/cosine_gradient_descent.hpp" namespace "pele":
    cdef cppclass cppCosineGradientDescent "pele::CosineGradientDescent":
        cppCosineGradientDescent(shared_ptr[_pele.cBasePotential], _pele.Array[double], double, double, double, int, bool, int) except +
        void set_tol(double) except +
        void set_maxstep(double) except +
        void set_max_iter(int) except +
        void set_iprint(int) except +
        void set_verbosity(int) except +
        vector[double] get_time_trajectory() except +
        vector[double] get_gradient_norm_trajectory() except +
        vector[double] get_distance_trajectory() except +
        vector[double] get_energy_trajectory() except +
        vector[double] get_costly_time_trajectory() except +
        vector[vector[double]] get_coordinate_trajectory() except +
        vector[vector[double]] get_gradient_trajectory() except +


cdef class _Cdef_CosineGradientDescent_CPP(_pele_opt.GradientOptimizer):
    """
    Python interface for the C++ Cosine Gradient Descent implementation
    """
    cdef _pele.BasePotential pot
    cdef bool save_trajectory

    def __cinit__(self, potential, x0, double tol=1e-5, int iprint=-1,
                  energy=None, gradient=None, int nsteps=10000,
                  double cos_sim_tol=1e-2, double dt_initial=1e-5, int back_track_n=5,
                  int verbosity=0, events=None, logger=None, bool save_trajectory=False, int iterations_before_save=1):
        potential = as_cpp_potential(potential, verbose=verbosity > 0)
        self.pot = potential
        self.save_trajectory = save_trajectory
        if logger is not None:
            print("warning: C++ cosine gradient descent is ignoring logger")
        
        cdef np.ndarray[double, ndim=1] x0c = np.array(x0, dtype=float)
        self.thisptr = shared_ptr[_pele_opt.cGradientOptimizer](<_pele_opt.cGradientOptimizer*> 
            new cppCosineGradientDescent(self.pot.thisptr,
                                         _pele.Array[double](<double*> x0c.data, x0c.size),
                                         tol, cos_sim_tol, dt_initial, back_track_n, save_trajectory, iterations_before_save))
        
        cdef cppCosineGradientDescent* cos_gd_ptr = <cppCosineGradientDescent*> self.thisptr.get()
        cos_gd_ptr.set_max_iter(nsteps)
        cos_gd_ptr.set_verbosity(verbosity)
        cos_gd_ptr.set_iprint(iprint)

        cdef np.ndarray[double, ndim=1] g_
        if energy is not None and gradient is not None:
            g_ = gradient
            self.thisptr.get().set_func_gradient(energy, _pele.Array[double](<double*> g_.data, g_.size))

        self.events = events
        if self.events is None:
            self.events = []

    def get_result(self):
        cdef cppCosineGradientDescent* cos_gd_ptr = <cppCosineGradientDescent*> self.thisptr.get()
        result = super(_Cdef_CosineGradientDescent_CPP, self).get_result()
        if self.save_trajectory:
            result.time_trajectory = np.array(cos_gd_ptr.get_time_trajectory())
            result.gradient_norm_trajectory = np.array(cos_gd_ptr.get_gradient_norm_trajectory())
            result.distance_trajectory = np.array(cos_gd_ptr.get_distance_trajectory())
            result.energy_trajectory = np.array(cos_gd_ptr.get_energy_trajectory())
            result.costly_time_trajectory = np.array(cos_gd_ptr.get_costly_time_trajectory())
            result.coordinate_trajectory = np.array(cos_gd_ptr.get_coordinate_trajectory())
            result.gradient_trajectory = np.array(cos_gd_ptr.get_gradient_trajectory())
        return result


class CosineGradientDescent_CPP(_Cdef_CosineGradientDescent_CPP):
    """
    Python interface for the C++ Cosine Gradient Descent implementation
    """