"""
# distutils: language = C++
# cython: language_level=3str
"""
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
import numpy as np

cimport numpy as np
from cpython cimport bool

cimport pele.potentials._pele as _pele
from pele.potentials._pele cimport shared_ptr
from pele.potentials._pele cimport array_wrap_np

# https://groups.google.com/forum/#!topic/cython-users/xAZxdCFw6Xs
cdef extern from *:
    ctypedef int INT2 "2"    # a fake type
    ctypedef int INT3 "3"    # a fake type

cdef extern from "pele/inversepower_stillinger_cut.hpp" namespace "pele":
    cdef cppclass cInversePowerStillingerCut "pele::InversePowerStillingerCut"[ndim]:
        cInversePowerStillingerCut(size_t pow, _pele.Array[double] radii, double rcut) except +
    cdef cppclass cInversePowerStillingerCutPeriodic "pele::InversePowerStillingerCutPeriodic"[ndim]:
        cInversePowerStillingerCutPeriodic(size_t pow, _pele.Array[double] radii, double rcut, _pele.Array[double] boxvec) except +
    cdef cppclass cInversePowerStillingerCutPeriodicCellLists "pele::InversePowerStillingerCutPeriodicCellLists"[ndim]:
        cInversePowerStillingerCutPeriodicCellLists(size_t pow, _pele.Array[double] radii, double rcut, _pele.Array[double] boxvec, double ncellx_scale) except +


cdef class InversePowerStillingerCut(_pele.BasePotential):
    """
    Python interface to C++ implementation of InversePowerStillingerCut, a smooth potential with cutoff distance.

    Parameters
    ----------
    pow : integer
        Exponent value

    radii : np.array
        List of particles radii

    ndim : integer
        Euclidean dimension of simulation box

    rcut : float
        cutoff distance

    boxvec : array
        Box vector

    boxl : float
        In case the box is a cube, the cube length can be given as boxl
        instead of providing boxvec
    """
    cdef public bool periodic
    cdef _pele.Array[double] bv_, radii_
    def __cinit__(self, pow, radii, ndim=3, boxvec=None, boxl=None, rcut=1.5,
                  ncellx_scale=1., use_cell_lists=False):
        assert(ndim == 2 or ndim == 3)
        assert not (boxvec is not None and boxl is not None)
        if boxl is not None:
            boxvec = [boxl] * ndim
        radii_ = array_wrap_np(radii)
        if boxvec is not None:
            if len(boxvec) != ndim:
                raise Exception("InversePowerStillinger: illegal input, illegal boxvec")
            bv_ = array_wrap_np(boxvec)
            if use_cell_lists:
                if ndim == 2:
                    # no cell lists, periodic, 2d
                    self.thisptr = shared_ptr[_pele.cBasePotential](
                        <_pele.cBasePotential*> new cInversePowerStillingerCutPeriodicCellLists[INT2](
                            pow, radii_, rcut, bv_, ncellx_scale
                            )
                        )
                else:
                    # no cell lists, periodic, 3d
                    self.thisptr = shared_ptr[_pele.cBasePotential](
                        <_pele.cBasePotential*> new cInversePowerStillingerCutPeriodicCellLists[INT3](
                            pow, radii_, rcut, bv_, ncellx_scale
                            )
                        )
            else:
                if ndim == 2:
                    # no cell lists, periodic, 2d
                    self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                                                                                        cInversePowerStillingerCutPeriodic[INT2](pow, radii_, rcut, bv_))
                else:
                    # no cell lists, periodic, 3d
                    self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                                                                                        cInversePowerStillingerCutPeriodic[INT3](pow, radii_, rcut, bv_))
        else:
            assert use_cell_lists is False, "InversePowerStillingerCut, not implemented for cartesian distance"
            if ndim == 2:
                # no cell lists, non-periodic, 2d
                self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                                                                                    cInversePowerStillingerCut[INT2](pow, radii_, rcut))
            else:
                # no cell lists, non-periodic, 3d
                self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                                                                                    cInversePowerStillingerCut[INT3](pow, radii_, rcut))
