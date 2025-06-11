"""
# distutils: language = C++
"""
import numpy as np
from numpy.core.fromnumeric import _compress_dispatcher, ndim

cimport numpy as np

cimport pele.potentials._pele as _pele
from pele.potentials._pele cimport shared_ptr

from libcpp cimport bool

# cython has no support for integer template argument.  This is a hack to get around it
# https://groups.google.com/forum/#!topic/cython-users/xAZxdCFw6Xs
# Basically you fool cython into thinking INT2 is the type integer,
# but in the generated c++ code you use 2 instead.
# The cython code MyClass[INT2] will create c++ code MyClass<2>.
cdef extern from *:
    ctypedef int INT2 "2"    # a fake type
    ctypedef int INT3 "3"    # a fake type
    ctypedef int INT5 "5"


# use external c++ class
cdef extern from "pele/inversepower_hs.hpp" namespace "pele":
    cdef cppclass  cInversePowerHS "pele::InversePowerHS"[ndim]:
        cInversePowerHS(double pow, double eps, double sigma, _pele.Array[double] radii, bool exact_sum, double non_additivity) except +
    cdef cppclass  cInversePowerHSPeriodic "pele::InversePowerHSPeriodic"[ndim]:
        cInversePowerHSPeriodic(double pow, double eps, double sigma, _pele.Array[double] radii, _pele.Array[double] boxvec,bool exact_sum, double non_additivity) except +
    cdef cppclass cInverseIntPowerHS "pele::InverseIntPowerHS"[ndim, pow]:
        cInverseIntPowerHS(double eps, double sigma, _pele.Array[double] radii,bool exact_sum, double non_additivity) except +
    cdef cppclass cInverseIntPowerHSPeriodic "pele::InverseIntPowerHSPeriodic"[ndim, pow]:
        cInverseIntPowerHSPeriodic(double eps, double sigma, _pele.Array[double] radii, _pele.Array[double] boxvec,bool exact_sum, double non_additivity) except +
    cdef cppclass cInverseHalfIntPowerHS "pele::InverseHalfIntPowerHS"[ndim, pow2]:
        cInverseHalfIntPowerHS(double eps, double sigma, _pele.Array[double] radii,bool exact_sum, double non_additivity) except +
    cdef cppclass cInverseHalfIntPowerHSPeriodic "pele::InverseHalfIntPowerHSPeriodic"[ndim, pow2]:
        cInverseHalfIntPowerHSPeriodic(double eps, double sigma, _pele.Array[double] radii, _pele.Array[double] boxvec,bool exact_sum, double non_additivity) except +
    cdef cppclass  cInversePowerHSPeriodicCellLists "pele::InversePowerHSPeriodicCellLists"[ndim]:
        cInversePowerHSPeriodicCellLists(double pow, double eps, double sigma, _pele.Array[double] radii, _pele.Array[double] boxvec, double ncellx_scale,bool exact_sum, double non_additivity) except +
    cdef cppclass  cInverseIntPowerHSPeriodicCellLists "pele::InverseIntPowerHSPeriodicCellLists"[ndim, pow]:
        cInverseIntPowerHSPeriodicCellLists(double eps, double sigma, _pele.Array[double] radii, _pele.Array[double] boxvec, double ncellx_scale,bool exact_sum, double non_additivity) except +
    cdef cppclass  cInverseHalfIntPowerHSPeriodicCellLists "pele::InverseHalfIntPowerHSPeriodicCellLists"[ndim, pow2]:
        cInverseHalfIntPowerHSPeriodicCellLists(double eps, double sigma, _pele.Array[double] radii, _pele.Array[double] boxvec, double ncellx_scale, bool exact_sum, double non_additivity) except +

cdef class InversePowerHS(_pele.PairwisePotentialInterface):
    """define the python interface to the c++ InversePowerHS implementation
    """
    cdef bool periodic
    cdef float pow
    cdef float eps
    cdef float sigma
    cdef int ndim
    cdef double[:] radii
    cdef double[:] boxvec
    cdef bool use_cell_lists
    cdef float ncellx_scale
    cdef double non_additivity
    def __cinit__(self, pow, eps, sigma, radii, ndim=3, boxvec=None, boxl=None,
                  bool use_cell_lists=False, ncellx_scale=1.0, bool exact_sum=False, double non_additivity=0.0):
        # stored for pickling
        self.pow = pow
        self.eps = eps
        self.sigma = sigma
        self.ndim = ndim
        max_radii = radii.max()
        self.use_cell_lists = use_cell_lists
        self.ncellx_scale = ncellx_scale
        self.non_additivity = non_additivity

        assert(ndim == 2 or ndim == 3)
        assert not (boxvec is not None and boxl is not None)
        if boxl is not None:
            boxvec = [boxl] * ndim
        
        cdef np.ndarray[double, ndim=1] bv
        cdef np.ndarray[double, ndim=1] radiic = np.array(radii, dtype=float)

        self.radii = radiic
        
        if boxvec is not None:
            boxvec = np.array(boxvec)
            min_box_vec = boxvec.min()
            self.boxvec = np.array(boxvec)
            if (min_box_vec < 2 * max_radii * (1. + sigma)):
                raise Exception("boxvec is too small for radii and sigma")
        else:
            self.boxvec = np.array([])


        if use_cell_lists:
            if boxvec is None:
                self.periodic = False
                raise NotImplementedError("This is not implemented yet.")
            else:
                self.periodic = True
                assert(len(boxvec) == ndim)
                bv = np.array(boxvec, dtype=float)
                if ndim == 2:
                    if self.close_enough(pow, 2):
                        # periodic, 2D, Hook
                        self.thisptr = shared_ptr[_pele.cBasePotential](
                            <_pele.cBasePotential*>new cInverseIntPowerHSPeriodicCellLists[INT2, INT2](
                                eps, sigma, _pele.Array[double](<double*> radiic.data, radiic.size),
                                _pele.Array[double](<double*> bv.data, bv.size), ncellx_scale, exact_sum, non_additivity
                                )
                            )
                    elif self.close_enough(pow, 2.5):
                        # periodic, 2D, Hertz
                        self.thisptr = shared_ptr[_pele.cBasePotential](
                            <_pele.cBasePotential*>new cInverseHalfIntPowerHSPeriodicCellLists[INT2, INT5](
                                eps, sigma, _pele.Array[double](<double*> radiic.data, radiic.size),
                                _pele.Array[double](<double*> bv.data, bv.size), ncellx_scale, exact_sum, non_additivity
                                )
                            )
                    else:
                        # periodic, 2D, any
                        self.thisptr = shared_ptr[_pele.cBasePotential](
                            <_pele.cBasePotential*>new cInversePowerHSPeriodicCellLists[INT2](
                                pow, eps, sigma, _pele.Array[double](<double*> radiic.data, radiic.size),
                                _pele.Array[double](<double*> bv.data, bv.size), ncellx_scale, exact_sum, non_additivity
                                )
                            )
                else:
                    if self.close_enough(pow, 2):
                        # periodic, 3D, Hook
                        self.thisptr = shared_ptr[_pele.cBasePotential](
                            <_pele.cBasePotential*>new cInverseIntPowerHSPeriodicCellLists[INT3, INT2](
                                eps, sigma, _pele.Array[double](<double*> radiic.data, radiic.size),
                                _pele.Array[double](<double*> bv.data, bv.size), ncellx_scale, exact_sum, non_additivity
                                )
                            )
                    elif self.close_enough(pow, 2.5):
                        # periodic, 3D, Hertz
                        self.thisptr = shared_ptr[_pele.cBasePotential](
                            <_pele.cBasePotential*>new cInverseHalfIntPowerHSPeriodicCellLists[INT3, INT5](
                                eps, sigma, _pele.Array[double](<double*> radiic.data, radiic.size),
                                _pele.Array[double](<double*> bv.data, bv.size), ncellx_scale, exact_sum, non_additivity
                                )
                            )
                    else:
                        # periodic, 3D, any
                        self.thisptr = shared_ptr[_pele.cBasePotential](
                            <_pele.cBasePotential*>new cInversePowerHSPeriodicCellLists[INT3](
                                pow, eps, sigma, _pele.Array[double](<double*> radiic.data, radiic.size),
                                _pele.Array[double](<double*> bv.data, bv.size), ncellx_scale, exact_sum, non_additivity
                                )
                            )
        else:
            if boxvec is None:
                self.periodic = False
                if ndim == 2:
                    if self.close_enough(pow, 2):
                        # non-periodic, 2D, Hook
                        self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new
                                                                     cInverseIntPowerHS[INT2, INT2](eps, sigma, _pele.Array[double](<double*> radiic.data, radiic.size), exact_sum, non_additivity))
                    elif self.close_enough(pow, 2.5):
                        # non-periodic, 2D, Hertz
                        self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new
                                                                     cInverseHalfIntPowerHS[INT2, INT5](eps, sigma, _pele.Array[double](<double*> radiic.data, radiic.size), exact_sum, non_additivity))
                    else:
                        # non-periodic, 2D, any
                        self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new
                                                                     cInversePowerHS[INT2](pow, eps, sigma, _pele.Array[double](<double*> radiic.data, radiic.size), exact_sum, non_additivity))
                else:
                    if self.close_enough(pow, 2):
                        # non-periodic, 3D, Hook
                        self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new
                                                                     cInverseIntPowerHS[INT3, INT2](eps, sigma, _pele.Array[double](<double*> radiic.data, radiic.size), exact_sum, non_additivity))
                    elif self.close_enough(pow, 2.5):
                        # non-periodic, 3D, Hertz
                        self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new
                                                                     cInverseHalfIntPowerHS[INT3, INT5](eps, sigma, _pele.Array[double](<double*> radiic.data, radiic.size), exact_sum, non_additivity))
                    else:
                        # non-periodic, 3D, any
                        self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new
                                                                     cInversePowerHS[INT3](pow, eps, sigma, _pele.Array[double](<double*> radiic.data, radiic.size), exact_sum, non_additivity))

            else:
                self.periodic = True
                assert(len(boxvec)==ndim)
                bv = np.array(boxvec, dtype=float)
                if ndim == 2:
                    if self.close_enough(pow, 2):
                        # periodic, 2D, Hook
                        self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new
                                                                     cInverseIntPowerHSPeriodic[INT2, INT2](eps, sigma, _pele.Array[double](<double*> radiic.data, radiic.size),
                                                                                                 _pele.Array[double](<double*> bv.data, bv.size), exact_sum, non_additivity))
                    elif self.close_enough(pow, 2.5):
                        # periodic, 2D, Hertz
                        self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new
                                                                     cInverseHalfIntPowerHSPeriodic[INT2, INT5](eps, sigma, _pele.Array[double](<double*> radiic.data, radiic.size),
                                                                                                 _pele.Array[double](<double*> bv.data, bv.size), exact_sum, non_additivity) )
                    else:
                        # periodic, 2D, any
                        self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new
                                                                     cInversePowerHSPeriodic[INT2](pow, eps, sigma, _pele.Array[double](<double*> radiic.data, radiic.size),
                                                                                                 _pele.Array[double](<double*> bv.data, bv.size), exact_sum, non_additivity) )
                else:
                    if self.close_enough(pow, 2):
                        # periodic, 3D, Hook
                        self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new
                                                                     cInverseIntPowerHSPeriodic[INT3, INT2](eps, sigma, _pele.Array[double](<double*> radiic.data, radiic.size),
                                                                                                 _pele.Array[double](<double*> bv.data, bv.size), exact_sum, non_additivity) )
                    elif self.close_enough(pow, 2.5):
                        # periodic, 3D, Hertz
                        self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new
                                                                     cInverseHalfIntPowerHSPeriodic[INT3, INT5](eps, sigma, _pele.Array[double](<double*> radiic.data, radiic.size),
                                                                                                 _pele.Array[double](<double*> bv.data, bv.size), exact_sum, non_additivity) )
                    else:
                        # periodic, 3D, any
                        self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new
                                                                     cInversePowerHSPeriodic[INT3](pow, eps, sigma, _pele.Array[double](<double*> radiic.data, radiic.size),
                                                                                                 _pele.Array[double](<double*> bv.data, bv.size), exact_sum, non_additivity) )


    def close_enough(self, pow_in, pow_true):
        in_ratio = float.as_integer_ratio(float(pow_in))
        true_ratio = float.as_integer_ratio(float(pow_true))
        if in_ratio[0] != true_ratio[0]:
            return False
        elif in_ratio[1] != true_ratio[1]:
            return False
        else:
            return True

    def __reduce__(self):
        d = {}
        return (self.__class__, (self, self.pow, self.eps, self.sigma, self.radii, self.ndim, self.boxvec, None,
                  self.use_cell_lists, self.ncellx_scale), d)

    def __setstate__(self, d):
        pass 