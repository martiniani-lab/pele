"""
# distutils: language = C++
"""
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

cdef extern from "pele/inversepower_stillinger_cut_quad.hpp" namespace "pele":
    cdef cppclass cInversePowerStillingerCutQuad "pele::InversePowerStillingerCutQuad"[ndim]:
        cInversePowerStillingerCutQuad(int power, double v0, double cutoff_factor, _pele.Array[double] radii, double non_additivity) except +
    cdef cppclass cInversePowerStillingerCutQuadPeriodic "pele::InversePowerStillingerCutQuadPeriodic"[ndim]:
        cInversePowerStillingerCutQuadPeriodic(int power, double v0, double cutoff_factor, _pele.Array[double] radii, _pele.Array[double] boxvec, double non_additivity) except +
    cdef cppclass cInversePowerStillingerCutQuadPeriodicCellLists "pele::InversePowerStillingerCutQuadPeriodicCellLists"[ndim]:
        cInversePowerStillingerCutQuadPeriodicCellLists(int power, double v0, double cutoff_factor, _pele.Array[double] radii, _pele.Array[double] boxvec, double ncellx_scale, double non_additivity) except +


cdef class InversePowerStillingerCutQuad(_pele.BasePotential):
    """
    Python interface to C++ implementation of InversePowerStillingerCutQuad, a smooth repulsive potential with a cutoff distance
    This corresponds to the soft sphere potential in https://journals.aps.org/prx/abstract/10.1103/PhysRevX.12.021001.

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
    cpdef bool periodic
    cdef _pele.Array[double] bv_, radii_
    def __cinit__(self, power, v0, cutoff_factor, radii, ndim=3, boxvec=None, boxl=None,
                  ncellx_scale=1., use_cell_lists=False, non_additivity=0.):
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
                        <_pele.cBasePotential*> new cInversePowerStillingerCutQuadPeriodicCellLists[INT2](
                            power, v0, cutoff_factor, radii_, bv_, ncellx_scale, non_additivity
                            )
                        )
                else:
                    # no cell lists, periodic, 3d
                    self.thisptr = shared_ptr[_pele.cBasePotential](
                        <_pele.cBasePotential*> new cInversePowerStillingerCutQuadPeriodicCellLists[INT3](
                            power, v0, cutoff_factor, radii_, bv_, ncellx_scale, non_additivity
                            )
                        )
            else:
                if ndim == 2:
                    # no cell lists, periodic, 2d
                    self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                                                                                        cInversePowerStillingerCutQuadPeriodic[INT2](power, v0, cutoff_factor, radii_, bv_, non_additivity))
                else:
                    # no cell lists, periodic, 3d
                    self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                                                                                        cInversePowerStillingerCutQuadPeriodic[INT3](power, v0, cutoff_factor, radii_, bv_, non_additivity))
        else:
            assert use_cell_lists is False, "InversePowerStillingerCutQuad, not implemented for cartesian distance"
            if ndim == 2:
                # no cell lists, non-periodic, 2d
                self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                                                                                    cInversePowerStillingerCutQuad[INT2](power, v0, cutoff_factor, radii_, non_additivity))
            else:
                # no cell lists, non-periodic, 3d
                self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                                                                                    cInversePowerStillingerCutQuad[INT3](power, v0, cutoff_factor, radii_, non_additivity))