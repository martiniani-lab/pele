"""a collections of cythonized routines for use in the various potentials"""
# cython: language_level=3str
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
import cython
import numpy as np
cimport numpy as np


cdef extern from "math.h":
    double sin(double x)
    double cos(double x)

@cython.boundscheck(False)
@cython.wraparound(False)
def xymodel_energy_gradient(np.ndarray[double, ndim=1] angles, 
                               np.ndarray[double, ndim=2] phases, 
                               np.ndarray[np.int64_t, ndim=2] neighbors):
    cdef int u, v
    cdef np.ndarray[double, ndim=1] grad = np.zeros(angles.size)
    cdef double E = 0.
    cdef double a, phase
    cdef int nneibs = neighbors.shape[0]
    for i in range(nneibs):
        u = neighbors[i,0]
        v = neighbors[i,1]
        phase = phases[u, v]
        a = -angles[u] + angles[v] + phase
        E += cos(a)
        g = -sin(a)
        grad[u] += g
        grad[v] += -g
    
    return -E, grad
