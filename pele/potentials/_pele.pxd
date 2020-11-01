cimport numpy as np
import numpy as np
from ctypes import c_size_t as size_t

#===============================================================================
# shared pointer
#===============================================================================
cdef extern from "<memory>" namespace "std":
    cdef cppclass shared_ptr[T]:
        shared_ptr() except+
        shared_ptr(T*) except+
        T* get() except+
        # T& operator*() # doesn't do anything
        # Note: operator->, operator= are not supported

#===============================================================================
# std::vector
#===============================================================================
cdef extern from "<vector>" namespace "std":
    cdef cppclass vector[T]:
        cppclass iterator:
            T operator*()
            iterator operator++()
            bint operator==(iterator)
            bint operator!=(iterator)
        vector()
        void push_back(T&)
        T& operator[](int)
        T& at(int)
        iterator begin()
        iterator end()

cdef vector[int].iterator iter  #iter is declared as being of type vector<int>::iterator

#===============================================================================
# pele::Array
#===============================================================================
cdef extern from "pele/array.hpp" namespace "pele":
    cdef cppclass Array[dtype] :
        Array() except +
        Array(size_t) except +
        Array(dtype*, size_t n) except +
        size_t size() except +
        Array[dtype] copy() except +
        dtype *data() except +
        dtype & operator[](size_t) except +

#===============================================================================
# pele::BasePotential
#===============================================================================
cdef extern from "pele/base_potential.hpp" namespace "pele":
    cdef cppclass  cBasePotential "pele::BasePotential":
        cBasePotential() except +
        double get_energy(Array[double] &x) except +
        double get_energy_gradient(Array[double] &x, Array[double] &grad) except +
        double get_energy_gradient_hessian(Array[double] &x, Array[double] &g, Array[double] &hess) except +
        void get_hessian(Array[double] &x, Array[double] &hess) except +
        void numerical_gradient(Array[double] &x, Array[double] &grad, double eps) except +
        void numerical_hessian(Array[double] &x, Array[double] &hess, double eps) except +

#cdef extern from "potentialfunction.hpp" namespace "pele":
#    cdef cppclass  cPotentialFunction "pele::PotentialFunction":
#        cPotentialFunction(
#            double (*energy)(Array[double] x, void *userdata) except *,
#            double (*energy_gradient)(Array[double] x, Array[double] grad, void *userdata) except *,
#            void *userdata) except +

#===============================================================================
# pele::PairwisePotentialInterface
#===============================================================================
cdef extern from "pele/pairwise_potential_interface.hpp" namespace "pele":
    cdef cppclass  cPairwisePotentialInterface "pele::PairwisePotentialInterface":
        cPairwisePotentialInterface() except +
        double get_energy(Array[double] &x) except +
        double get_energy_gradient(Array[double] &x, Array[double] &grad) except +
        double get_energy_gradient_hessian(Array[double] &x, Array[double] &g, Array[double] &hess) except +
        void get_hessian(Array[double] &x, Array[double] &hess) except +
        void numerical_gradient(Array[double] &x, Array[double] &grad, double eps) except +
        void numerical_hessian(Array[double] &x, Array[double] &hess, double eps) except +
        double get_interaction_energy_gradient(double r2, double *gij, size_t atom_i, size_t atom_j) except +
        double get_interaction_energy_gradient_hessian(double r2, double *gij, double *hij, size_t atom_i, size_t atom_j) except +
        void get_neighbors_picky(Array[double] & coords, Array[vector[size_t]] & neighbor_indss, Array[vector[vector[double]]] & neighbor_distss, Array[short] & include_atoms, double cutoff_factor) except +
        void get_neighbors(Array[double] & coords, Array[vector[size_t]] & neighbor_indss, Array[vector[vector[double]]] & neighbor_distss, double cutoff_factor) except +
        vector[size_t] get_overlaps(Array[double] & coords) except +
        Array[size_t] get_atom_order(Array[double] & coords) except +
        size_t get_ndim() except +
        Array[double] get_radii() except +

#===============================================================================
# cython BasePotential
#===============================================================================
cdef class BasePotential:
    cdef shared_ptr[cBasePotential] thisptr      # hold a C++ instance which we're wrapping

#===============================================================================
# cython PairwisePotentialInterface
#===============================================================================
cdef class PairwisePotentialInterface(BasePotential):
    pass

#===============================================================================
# pele::CombinedPotential
#===============================================================================
cdef extern from "pele/combine_potentials.hpp" namespace "pele":
    cdef cppclass  cCombinedPotential "pele::CombinedPotential":
        cCombinedPotential() except +
        double get_energy(Array[double] &x) except +
        double get_energy_gradient(Array[double] &x, Array[double] &grad) except +
        void add_potential(shared_ptr[cBasePotential] potential) except +


#===============================================================================
# conversion routines between numpy arrays and pele::Array
#===============================================================================
cdef inline Array[double] array_wrap_np(np.ndarray[double] v) except *:
    """return a pele Array which wraps the data in a numpy array

    Notes
    -----
    We must be careful we only wrap the existing data.
    """
    if not v.flags["FORC"]:
        raise ValueError("the numpy array is not c-contiguous.  copy it into a contiguous format before wrapping with pele::Array")
    return Array[double](<double *> v.data, v.size)

cdef inline np.ndarray[double, ndim=1] pele_array_to_np(Array[double] v):
    """copy the data in a pele::Array into a new numpy array
    """
    cdef int i
    cdef int N = v.size()
    cdef np.ndarray[double, ndim=1] vnew = np.zeros(N)
    for i in xrange(N):
        vnew[i] = v[i]
    return vnew

cdef inline Array[size_t] array_wrap_np_size_t(np.ndarray[size_t] v) except *:
    """return a pele Array which wraps the data in a numpy array

    Notes
    -----
    we must be careful we only wrap the existing data
    """
    if not v.flags["FORC"]:
        raise ValueError("the numpy array is not c-contiguous.  copy it into a contiguous format before wrapping with pele::Array")
    return Array[size_t](<size_t *> v.data, v.size)

cdef inline Array[size_t] array_size_t_from_np(vin) except *:
    """return a pele Array which contains a copy of the data in a numpy array
    """
    cdef int i
    cdef np.ndarray[long, ndim=1] v = np.asarray(vin, dtype=long)
    cdef int N = v.size
    cdef Array[size_t] vnew = Array[size_t](N)
    for i in xrange(N):
        vnew[i] = v[i]
    return vnew

cdef inline np.ndarray[size_t, ndim=1] pele_array_to_np_size_t(Array[size_t] v):
    """copy the data in a pele::Array into a new numpy array
    """
    cdef int i
    cdef int N = v.size()
    cdef np.ndarray[size_t, ndim=1] vnew = np.zeros(N, dtype=size_t)
    for i in xrange(N):
        vnew[i] = v[i]
    return vnew

cdef inline Array[long] array_wrap_np_long(np.ndarray[long] v) except *:
    """return a pele Array which wraps the data in a numpy array

    Notes
    -----
    we must be careful that we only wrap the existing data
    """
    if not v.flags["FORC"]:
        raise ValueError("the numpy array is not c-contiguous.  copy it into a contiguous format before wrapping with pele::Array")
    return Array[long](<long*> v.data, v.size)

cdef inline np.ndarray[long, ndim=1] pele_array_to_np_long(Array[long] v):
    """copy the data in a pele::Array into a new numpy array
    """
    cdef int i
    cdef int N = v.size()
    cdef np.ndarray[long, ndim=1] vnew = np.zeros(N, dtype=long)
    for i in xrange(N):
        vnew[i] = v[i]
    return vnew
