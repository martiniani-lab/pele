"""
# distutils: language = C++
"""
import numpy as np
from numpy.core.fromnumeric import _compress_dispatcher, ndim

cimport numpy as np
from cpython cimport bool

cimport pele.potentials._pele as _pele
from pele.potentials._pele cimport shared_ptr

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
cdef extern from "pele/inversepower.hpp" namespace "pele":
    cdef cppclass  cInversePower "pele::InversePower"[ndim]:
        cInversePower(double pow, double eps, _pele.Array[double] radii, exact_sum) except +
    cdef cppclass  cInversePowerPeriodic "pele::InversePowerPeriodic"[ndim]:
        cInversePowerPeriodic(double pow, double eps, _pele.Array[double] radii, _pele.Array[double] boxvec, exact_sum) except +
    cdef cppclass cInverseIntPower "pele::InverseIntPower"[ndim, pow]:
        cInverseIntPower(double eps, _pele.Array[double] radii, exact_sum) except +
    cdef cppclass cInverseIntPowerPeriodic "pele::InverseIntPowerPeriodic"[ndim, pow]:
        cInverseIntPowerPeriodic(double eps, _pele.Array[double] radii, _pele.Array[double] boxvec, exact_sum) except +
    cdef cppclass cInverseHalfIntPower "pele::InverseHalfIntPower"[ndim, pow2]:
        cInverseHalfIntPower(double eps, _pele.Array[double] radii, exact_sum) except +
    cdef cppclass cInverseHalfIntPowerPeriodic "pele::InverseHalfIntPowerPeriodic"[ndim, pow2]:
        cInverseHalfIntPowerPeriodic(double eps, _pele.Array[double] radii, _pele.Array[double] boxvec, exact_sum) except +
    cdef cppclass  cInversePowerPeriodicCellLists "pele::InversePowerPeriodicCellLists"[ndim]:
        cInversePowerPeriodicCellLists(double pow, double eps, _pele.Array[double] radii, _pele.Array[double] boxvec, double ncellx_scale, exact_sum) except +
    cdef cppclass  cInverseIntPowerPeriodicCellLists "pele::InverseIntPowerPeriodicCellLists"[ndim, pow]:
        cInverseIntPowerPeriodicCellLists(double eps, _pele.Array[double] radii, _pele.Array[double] boxvec, double ncellx_scale, exact_sum) except +
    cdef cppclass  cInverseHalfIntPowerPeriodicCellLists "pele::InverseHalfIntPowerPeriodicCellLists"[ndim, pow2]:
        cInverseHalfIntPowerPeriodicCellLists(double eps, _pele.Array[double] radii, _pele.Array[double] boxvec, double ncellx_scale, exact_sum) except +

cdef class InversePower(_pele.PairwisePotentialInterface):
    """define the python interface to the c++ InversePower implementation
    """
    cdef bool periodic
    cdef float pow
    cdef float eps
    cdef int ndim
    cdef double[:] radii
    cdef double[:] boxvec
    cdef bool use_cell_lists
    cdef float ncellx_scale
    def __cinit__(self, pow, eps, radii, ndim=3, boxvec=None, boxl=None,
                  use_cell_lists=False, ncellx_scale=1.0, exact_sum=False):
        
        # stored for pickling
        self.pow = pow
        self.eps = eps
        self.radii = radii
        self.ndim = ndim

        self.use_cell_lists = bool(use_cell_lists)
        self.ncellx_scale = ncellx_scale

        assert(ndim == 2 or ndim == 3)
        assert not (boxvec is not None and boxl is not None)
        if boxl is not None:
            boxvec = [boxl] * ndim
        cdef np.ndarray[double, ndim=1] bv
        cdef np.ndarray[double, ndim=1] radiic = np.array(radii, dtype=float)

        self.radii = radiic
        self.boxvec = np.array(boxvec)



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
                            <_pele.cBasePotential*>new cInverseIntPowerPeriodicCellLists[INT2, INT2](
                                eps, _pele.Array[double](<double*> radiic.data, radiic.size),
                                _pele.Array[double](<double*> bv.data, bv.size), ncellx_scale, exact_sum
                                )
                            )
                    elif self.close_enough(pow, 2.5):
                        # periodic, 2D, Hertz
                        self.thisptr = shared_ptr[_pele.cBasePotential](
                            <_pele.cBasePotential*>new cInverseHalfIntPowerPeriodicCellLists[INT2, INT5](
                                eps, _pele.Array[double](<double*> radiic.data, radiic.size),
                                _pele.Array[double](<double*> bv.data, bv.size), ncellx_scale, exact_sum
                                )
                            )
                    else:
                        # periodic, 2D, any
                        self.thisptr = shared_ptr[_pele.cBasePotential](
                            <_pele.cBasePotential*>new cInversePowerPeriodicCellLists[INT2](
                                pow, eps, _pele.Array[double](<double*> radiic.data, radiic.size),
                                _pele.Array[double](<double*> bv.data, bv.size), ncellx_scale, exact_sum
                                )
                            )
                else:
                    if self.close_enough(pow, 2):
                        # periodic, 3D, Hook
                        self.thisptr = shared_ptr[_pele.cBasePotential](
                            <_pele.cBasePotential*>new cInverseIntPowerPeriodicCellLists[INT3, INT2](
                                eps, _pele.Array[double](<double*> radiic.data, radiic.size),
                                _pele.Array[double](<double*> bv.data, bv.size), ncellx_scale, exact_sum
                                )
                            )
                    elif self.close_enough(pow, 2.5):
                        # periodic, 3D, Hertz
                        self.thisptr = shared_ptr[_pele.cBasePotential](
                            <_pele.cBasePotential*>new cInverseHalfIntPowerPeriodicCellLists[INT3, INT5](
                                eps, _pele.Array[double](<double*> radiic.data, radiic.size),
                                _pele.Array[double](<double*> bv.data, bv.size), ncellx_scale, exact_sum
                                )
                            )
                    else:
                        # periodic, 3D, any
                        self.thisptr = shared_ptr[_pele.cBasePotential](
                            <_pele.cBasePotential*>new cInversePowerPeriodicCellLists[INT3](
                                pow, eps, _pele.Array[double](<double*> radiic.data, radiic.size),
                                _pele.Array[double](<double*> bv.data, bv.size), ncellx_scale, exact_sum
                                )
                            )
        else:
            if boxvec is None:
                self.periodic = False
                if ndim == 2:
                    if self.close_enough(pow, 2):
                        # non-periodic, 2D, Hook
                        self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new
                                                                     cInverseIntPower[INT2, INT2](eps, _pele.Array[double](<double*> radiic.data, radiic.size), exact_sum))
                    elif self.close_enough(pow, 2.5):
                        # non-periodic, 2D, Hertz
                        self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new
                                                                     cInverseHalfIntPower[INT2, INT5](eps, _pele.Array[double](<double*> radiic.data, radiic.size), exact_sum))
                    else:
                        # non-periodic, 2D, any
                        self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new
                                                                     cInversePower[INT2](pow, eps, _pele.Array[double](<double*> radiic.data, radiic.size), exact_sum))
                else:
                    if self.close_enough(pow, 2):
                        # non-periodic, 3D, Hook
                        self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new
                                                                     cInverseIntPower[INT3, INT2](eps, _pele.Array[double](<double*> radiic.data, radiic.size), exact_sum))
                    elif self.close_enough(pow, 2.5):
                        # non-periodic, 3D, Hertz
                        self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new
                                                                     cInverseHalfIntPower[INT3, INT5](eps, _pele.Array[double](<double*> radiic.data, radiic.size), exact_sum))
                    else:
                        # non-periodic, 3D, any
                        self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new
                                                                     cInversePower[INT3](pow, eps, _pele.Array[double](<double*> radiic.data, radiic.size), exact_sum))

            else:
                self.periodic = True
                assert(len(boxvec)==ndim)
                bv = np.array(boxvec, dtype=float)
                if ndim == 2:
                    if self.close_enough(pow, 2):
                        # periodic, 2D, Hook
                        self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new
                                                                     cInverseIntPowerPeriodic[INT2, INT2](eps, _pele.Array[double](<double*> radiic.data, radiic.size),
                                                                                                 _pele.Array[double](<double*> bv.data, bv.size), exact_sum))
                    elif self.close_enough(pow, 2.5):
                        # periodic, 2D, Hertz
                        self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new
                                                                     cInverseHalfIntPowerPeriodic[INT2, INT5](eps, _pele.Array[double](<double*> radiic.data, radiic.size),
                                                                                                 _pele.Array[double](<double*> bv.data, bv.size), exact_sum) )
                    else:
                        # periodic, 2D, any
                        self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new
                                                                     cInversePowerPeriodic[INT2](pow, eps, _pele.Array[double](<double*> radiic.data, radiic.size),
                                                                                                 _pele.Array[double](<double*> bv.data, bv.size), exact_sum) )
                else:
                    if self.close_enough(pow, 2):
                        # periodic, 3D, Hook
                        self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new
                                                                     cInverseIntPowerPeriodic[INT3, INT2](eps, _pele.Array[double](<double*> radiic.data, radiic.size),
                                                                                                 _pele.Array[double](<double*> bv.data, bv.size), exact_sum) )
                    elif self.close_enough(pow, 2.5):
                        # periodic, 3D, Hertz
                        self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new
                                                                     cInverseHalfIntPowerPeriodic[INT3, INT5](eps, _pele.Array[double](<double*> radiic.data, radiic.size),
                                                                                                 _pele.Array[double](<double*> bv.data, bv.size), exact_sum) )
                    else:
                        # periodic, 3D, any
                        self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new
                                                                     cInversePowerPeriodic[INT3](pow, eps, _pele.Array[double](<double*> radiic.data, radiic.size),
                                                                                                 _pele.Array[double](<double*> bv.data, bv.size), exact_sum) )


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
        return (self.__class__, (self, self.pow, self.eps, self.radii, self.ndim, self.boxvec, None,
                  self.use_cell_lists, self.ncellx_scale), d)

    def __setstate__(self, d):
        pass
