"""
# distutils: language = C++
"""
cimport pele.potentials._pele as _pele
from pele.potentials._pele cimport array_wrap_np
cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free
from distance_enum import Distance


# cython has no support for integer template argument.  This is a hack to get around it
# https://groups.google.com/forum/#!topic/cython-users/xAZxdCFw6Xs
# Basically you fool cython into thinking INT2 is the type integer,
# but in the generated c++ code you use 2 instead.
# The cython code MyClass[INT2] will create c++ code MyClass<2>.
cdef extern from *:
    ctypedef int INT2 "2"    # a fake type
    ctypedef int INT3 "3"    # a fake type


# use external c++ classes
cdef extern from "pele/distance.hpp" namespace "pele":
    cdef cppclass cppPeriodicDistance "pele::periodic_distance"[ndim]:
        cppPeriodicDistance(_pele.Array[double] box) except +
        void put_atom_in_box(double *)
        void put_in_box(_pele.Array[double]& double)
    cdef cppclass cppLeesEdwardsDistance "pele::leesedwards_distance"[ndim]:
        cppLeesEdwardsDistance(_pele.Array[double] box, double shear) except +
        void put_atom_in_box(double *)
        void put_in_box(_pele.Array[double]& double)


cpdef put_atom_in_box(np.ndarray[double] r, int ndim, method, np.ndarray[double] box, double shear=0.0):
    """
    Define the Python interface to the C++ distance implementation.

    Parameters
    ----------
    r : [float]
        Position of the particle
    ndim : int
        Number of dimensions
    method : Distance Enum
        Distance measurement method / boundary conditions.
    box : np.array(float)
        Box size
    shear : float, optional
        Amount of shear (for Lees-Edwards distance measure)
    """

    # Assert that the input parameters are right
    assert ndim == 2 or ndim == 3, "Dimension outside the required range."
    assert method is Distance.PERIODIC or method is Distance.LEES_EDWARDS, \
           "Distance measurement method should be PERIODIC or LEES_EDWARDS."

    # Define pointers for all distance measures
    # (otherwise this would be clumsily handled by Cython, which would call
    #  the empty constructor before constructing the objects properly.)
    cdef cppPeriodicDistance[INT2] *dist_per_2d
    cdef cppPeriodicDistance[INT3] *dist_per_3d
    cdef cppLeesEdwardsDistance[INT2] *dist_leesedwards_2d
    cdef cppLeesEdwardsDistance[INT3] *dist_leesedwards_3d

    # Initialize data arrays in C
    cdef double *c_r = <double *>malloc(ndim * sizeof(double))
    for i in xrange(ndim):
        c_r[i] = r[i]

    # Get box size from the input parameters
    cdef _pele.Array[double] c_box = array_wrap_np(box)

    # Calculate the distance
    if method is Distance.PERIODIC:
        if ndim == 2:
            dist_per_2d = new cppPeriodicDistance[INT2](c_box)
            dist_per_2d.put_atom_in_box(c_r)
        else:
            dist_per_3d = new cppPeriodicDistance[INT3](c_box)
            dist_per_3d.put_atom_in_box(c_r)
    else:
        if ndim == 2:
            dist_leesedwards_2d = new cppLeesEdwardsDistance[INT2](c_box, shear)
            dist_leesedwards_2d.put_atom_in_box(c_r)
        else:
            dist_leesedwards_3d = new cppLeesEdwardsDistance[INT3](c_box, shear)
            dist_leesedwards_3d.put_atom_in_box(c_r)

    # Copy results into Python object
    r_boxed = np.empty(ndim)
    for i in xrange(ndim):
        r_boxed[i] = c_r[i]

    # Free memory
    free(c_r)

    return r_boxed


cpdef put_in_box(np.ndarray[double] rs, int ndim, method, np.ndarray[double] box, double shear=0.0):
    """
    Define the Python interface to the C++ distance implementation.

    Parameters
    ----------
    rs : np.array[float]
        Positions of all particles
    ndim : int
        Number of dimensions
    method : Distance Enum
        Distance measurement method / boundary conditions.
    box : np.array[float]
        Box size
    shear : float, optional
        Amount of shear (for Lees-Edwards distance measure)
    """

    # Assert that the input parameters are right
    assert ndim == 2 or ndim == 3, "Dimension outside the required range."
    assert method is Distance.PERIODIC or method is Distance.LEES_EDWARDS, \
           "Distance measurement method should be PERIODIC or LEES_EDWARDS."

    # Define pointers for all distance measures
    # (otherwise this would be clumsily handled by Cython, which would call
    #  the empty constructor before constructing the objects properly.)
    cdef cppPeriodicDistance[INT2] *dist_per_2d
    cdef cppPeriodicDistance[INT3] *dist_per_3d
    cdef cppLeesEdwardsDistance[INT2] *dist_leesedwards_2d
    cdef cppLeesEdwardsDistance[INT3] *dist_leesedwards_3d

    # Get box size from the input parameters
    cdef _pele.Array[double] c_box = array_wrap_np(box)
    cdef _pele.Array[double] c_rs = array_wrap_np(rs)

    # Calculate the distance
    if method is Distance.PERIODIC:
        if ndim == 2:
            dist_per_2d = new cppPeriodicDistance[INT2](c_box)
            dist_per_2d.put_in_box(c_rs)
        else:
            dist_per_3d = new cppPeriodicDistance[INT3](c_box)
            dist_per_3d.put_in_box(c_rs)
    else:
        if ndim == 2:
            dist_leesedwards_2d = new cppLeesEdwardsDistance[INT2](c_box, shear)
            dist_leesedwards_2d.put_in_box(c_rs)
        else:
            dist_leesedwards_3d = new cppLeesEdwardsDistance[INT3](c_box, shear)
            dist_leesedwards_3d.put_in_box(c_rs)

    # Copy results into Python object
    rs_boxed = np.empty(c_rs.size())
    for i in xrange(c_rs.size()):
        rs_boxed[i] = c_rs[i]

    return rs_boxed
