"""
# distutils: language = C++
"""
import numpy as np
cimport pele.potentials._pele as _pele
from pele.potentials._pele cimport shared_ptr
from pele.potentials._pele cimport array_wrap_np
from pele.potentials._pele cimport array_wrap_np_long, array_wrap_np_size_t
cimport numpy as np
from libcpp cimport bool
from libcpp cimport bool as cbool
from ctypes import c_size_t as size_t
from pele.distance import Distance

# cython has no support for integer template argument.  This is a hack to get around it
# https://groups.google.com/forum/#!topic/cython-users/xAZxdCFw6Xs
# Basically you fool cython into thinking INT2 is the type integer,
# but in the generated c++ code you use 2 instead.
# The cython code MyClass[INT2] will create c++ code MyClass<2>.
cdef extern from *:
    ctypedef int INT1 "1"    # a fake type
    ctypedef int INT2 "2"    # a fake type
    ctypedef int INT3 "3"    # a fake type
    ctypedef int INT6 "6"    # a fake type

# use external c++ class
cdef extern from "pele/hs_wca.hpp" namespace "pele":
    cdef cppclass  cHS_WCA "pele::HS_WCA"[ndim, exp]:
        cHS_WCA(double eps, double sca, _pele.Array[double] radii) except +
    cdef cppclass  cHS_WCACellLists "pele::HS_WCACellLists"[ndim, exp]:
        cHS_WCACellLists(double eps, double sca, _pele.Array[double] radii,
                         _pele.Array[double] boxvec,
                         double ncellsx_scale, cbool balance_omp) except +
    cdef cppclass  cHS_WCAPeriodic "pele::HS_WCAPeriodic"[ndim, exp]:
        cHS_WCAPeriodic(double eps, double sca, _pele.Array[double] radii,
                        _pele.Array[double] boxvec) except +
    cdef cppclass  cHS_WCALeesEdwards "pele::HS_WCALeesEdwards"[ndim, exp]:
        cHS_WCALeesEdwards(double eps, double sca, _pele.Array[double] radii,
                        _pele.Array[double] boxvec, double shear) except +
    cdef cppclass  cHS_WCAPeriodicCellLists "pele::HS_WCAPeriodicCellLists"[ndim, exp]:
        cHS_WCAPeriodicCellLists(double eps, double sca,
                                 _pele.Array[double] radii, _pele.Array[double] boxvec,
                                 double ncellx_scale, cbool balance_omp) except +
    cdef cppclass  cHS_WCALeesEdwardsCellLists "pele::HS_WCALeesEdwardsCellLists"[ndim, exp]:
        cHS_WCALeesEdwardsCellLists(double eps, double sca,
                                 _pele.Array[double] radii, _pele.Array[double] boxvec,
                                 double shear, double ncellx_scale, cbool balance_omp) except +
    cdef cppclass  cHS_WCANeighborList "pele::HS_WCANeighborList"[exp]:
        cHS_WCANeighborList(_pele.Array[size_t] & ilist, double eps, double sca,
                            _pele.Array[double] radii) except +

cdef extern from "pele/frozen_atoms.hpp" namespace "pele":
    cdef cppclass  cppFrozenPotentialWrapper "pele::FrozenPotentialWrapper":
        cppFrozenPotentialWrapper(shared_ptr[_pele.cBasePotential] potential,
            _pele.Array[double] reference_coords,
            _pele.Array[size_t] frozen_dof) except +

cdef class HS_WCA(_pele.PairwisePotentialInterface):
    """
    Define the python interface to the c++ HS_WCA implementation.

    Parameters
    ----------
    eps : float
        wca parameter
    sca : float
        the thickness of the wca shell is sca * R where R is the hard core
        radius of the sphere
    radii : array
        list of radii of the particles
    ndim : integer
        Euclidean dimension of simulation box
    boxvec : array
        Box vector
    boxl : float
        In case the box is a cube, the cube length can be given as boxl instead
        of providing boxvec
    distance_method : Distance Enum
        Distance measurement method / boundary conditions. Default: Cartesian
    pot_kwargs : dictionary, optional
        Dictionary containing information used by the potential
        (e.g. shear for Lees-Edwards boundary conditions, exponent of the
        potential, flag determining if OpenMP subdomain sizes should be balanced
        by increasing the cell size)
    use_frozen : bool
        Flag to switch on freezing of selected particles
    use_cell_lists : bool
        Flag to switch on usage of cell lists
    reference_coords : array
        Initial values of all coordinates, consisting of frozen and non-frozen
        dof
    frozen_atoms : array
        List of labels of frozen (immobile) particles. Note: This is not the
        list of frozen degrees of freedom
    ncellx_scale : float
        Parameter controlling the cell list grid spacing: values larger than
        unity lead to finer cell meshing
    """
    cpdef bool periodic
    cpdef bool leesedwards
    def __cinit__(self, double eps=1.0, double sca=0.12,
                  np.ndarray[double, ndim=1] radii=None, int ndim=3, boxvec=None,
                  boxl=None, distance_method=Distance.CARTESIAN, pot_kwargs={},
                  bool use_frozen=False, bool use_cell_lists=False,
                  np.ndarray[double, ndim=1] reference_coords=None,
                  frozen_atoms=None, double ncellx_scale=1.0):
        assert boxvec is None or boxl is None
        cdef np.ndarray[size_t, ndim=1] frozen_dof
        if boxl is not None:
            boxvec = np.array([boxl] * ndim)
        cdef np.ndarray[double, ndim=1] bv
        cdef size_t i
        if use_frozen:
            if frozen_atoms is None:
                print "HS_WCA: warning: initialising frozen particle potential without frozen particles"
            if reference_coords is None:
                raise Exception("missing input: can not initialise frozen particle potential without reference coordinates")
            else:
                frozen_dof = np.array([range(ndim * i, ndim * i + ndim) for i in frozen_atoms], dtype=size_t).reshape(-1)
        if use_cell_lists and boxvec is None:
            raise Exception("HS_WCA: illegal input")
        bv = None
        self.periodic = distance_method is Distance.PERIODIC
        self.leesedwards = distance_method is Distance.LEES_EDWARDS
        cdef cbool balance_omp
        if 'balance_omp' in pot_kwargs:
            balance_omp = pot_kwargs['balance_omp']
        else:
            balance_omp = True
        if 'exp' in pot_kwargs:
            exp = pot_kwargs['exp']
        else:
            exp = 6
        if self.leesedwards or self.periodic:
            if boxvec is None:
                raise Exception("boxvec is not specified")
            bv = np.array(boxvec, dtype=float)
            assert bv.size == ndim
        cdef _pele.Array[double] rd_
        cdef _pele.Array[double] bv_
        cdef _pele.Array[double] rc_
        cdef _pele.Array[size_t] fd_
        if radii is not None:
            rd_ = array_wrap_np(radii)
        if bv is not None:
            bv_ = array_wrap_np(bv)
        if reference_coords is not None:
            rc_ = array_wrap_np(reference_coords)
        if frozen_atoms is not None:
            fd_ = array_wrap_np_size_t(frozen_dof)

        """
        non-frozen
        """
        if exp == 1:
            if self.leesedwards:
                assert 'shear' in pot_kwargs, "Required argument 'shear' not defined in pot_kwargs."
                if ndim == 2:
                    if not use_cell_lists:
                        # non-frozen, 2d, periodic, no cell lists
                        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                             cHS_WCALeesEdwards[INT2, INT1](eps, sca, rd_, bv_, pot_kwargs['shear']))
                    else:
                        # non-frozen, 2d, periodic, use cell lists
                        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                             cHS_WCALeesEdwardsCellLists[INT2, INT1](eps, sca, rd_, bv_,
                                                                     pot_kwargs['shear'],
                                                                     ncellx_scale, balance_omp))
                elif ndim == 3:
                    if not use_cell_lists:
                        # non-frozen, 3d, periodic, no cell lists
                        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                             cHS_WCALeesEdwards[INT3, INT1](eps, sca, rd_, bv_, pot_kwargs['shear']))
                    else:
                        # non-frozen, 3d, periodic, use cell lists
                        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                             cHS_WCALeesEdwardsCellLists[INT3, INT1](eps, sca, rd_, bv_,
                                                                     pot_kwargs['shear'],
                                                                     ncellx_scale, balance_omp))
                else:
                    raise Exception("HS_WCA: illegal ndim")
            elif self.periodic:
                if ndim == 2:
                    if not use_cell_lists:
                        # non-frozen, 2d, periodic, no cell lists
                        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                             cHS_WCAPeriodic[INT2, INT1](eps, sca, rd_, bv_))
                    else:
                        # non-frozen, 2d, periodic, use cell lists
                        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                             cHS_WCAPeriodicCellLists[INT2, INT1](eps, sca, rd_, bv_,
                                                                  ncellx_scale, balance_omp))
                elif ndim == 3:
                    if not use_cell_lists:
                        # non-frozen, 3d, periodic, no cell lists
                        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                             cHS_WCAPeriodic[INT3, INT1](eps, sca, rd_, bv_))
                    else:
                        # non-frozen, 3d, periodic, use cell lists
                        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                             cHS_WCAPeriodicCellLists[INT3, INT1](eps, sca, rd_, bv_,
                                                                  ncellx_scale, balance_omp))
                else:
                    raise Exception("HS_WCA: illegal ndim")
            else:
                if ndim == 2:
                    if not use_cell_lists:
                        # non-frozen, 2d, cartesian, no cell lists
                        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                             cHS_WCA[INT2, INT1](eps, sca, rd_))
                    else:
                        # non-frozen, 2d, cartesian, use cell lists
                        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                             cHS_WCACellLists[INT2, INT1](eps, sca, rd_, bv_,
                                                          ncellx_scale, balance_omp))
                elif ndim == 3:
                    if not use_cell_lists:
                        # non-frozen, 3d, cartesian, no cell lists
                        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                             cHS_WCA[INT3, INT1](eps, sca, rd_))
                    else:
                        # non-frozen, 3d, cartesian, use cell lists
                        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                             cHS_WCACellLists[INT3, INT1](eps, sca, rd_, bv_,
                                                          ncellx_scale, balance_omp))
                else:
                    raise Exception("HS_WCA: illegal ndim")
        elif exp == 2:
            if self.leesedwards:
                assert 'shear' in pot_kwargs, "Required argument 'shear' not defined in pot_kwargs."
                if ndim == 2:
                    if not use_cell_lists:
                        # non-frozen, 2d, periodic, no cell lists
                        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                             cHS_WCALeesEdwards[INT2, INT2](eps, sca, rd_, bv_, pot_kwargs['shear']))
                    else:
                        # non-frozen, 2d, periodic, use cell lists
                        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                             cHS_WCALeesEdwardsCellLists[INT2, INT2](eps, sca, rd_, bv_,
                                                                     pot_kwargs['shear'],
                                                                     ncellx_scale, balance_omp))
                elif ndim == 3:
                    if not use_cell_lists:
                        # non-frozen, 3d, periodic, no cell lists
                        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                             cHS_WCALeesEdwards[INT3, INT2](eps, sca, rd_, bv_, pot_kwargs['shear']))
                    else:
                        # non-frozen, 3d, periodic, use cell lists
                        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                             cHS_WCALeesEdwardsCellLists[INT3, INT2](eps, sca, rd_, bv_,
                                                                     pot_kwargs['shear'],
                                                                     ncellx_scale, balance_omp))
                else:
                    raise Exception("HS_WCA: illegal ndim")
            elif self.periodic:
                if ndim == 2:
                    if not use_cell_lists:
                        # non-frozen, 2d, periodic, no cell lists
                        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                             cHS_WCAPeriodic[INT2, INT2](eps, sca, rd_, bv_))
                    else:
                        # non-frozen, 2d, periodic, use cell lists
                        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                             cHS_WCAPeriodicCellLists[INT2, INT2](eps, sca, rd_, bv_,
                                                                  ncellx_scale, balance_omp))
                elif ndim == 3:
                    if not use_cell_lists:
                        # non-frozen, 3d, periodic, no cell lists
                        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                             cHS_WCAPeriodic[INT3, INT2](eps, sca, rd_, bv_))
                    else:
                        # non-frozen, 3d, periodic, use cell lists
                        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                             cHS_WCAPeriodicCellLists[INT3, INT2](eps, sca, rd_, bv_,
                                                                  ncellx_scale, balance_omp))
                else:
                    raise Exception("HS_WCA: illegal ndim")
            else:
                if ndim == 2:
                    if not use_cell_lists:
                        # non-frozen, 2d, cartesian, no cell lists
                        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                             cHS_WCA[INT2, INT2](eps, sca, rd_))
                    else:
                        # non-frozen, 2d, cartesian, use cell lists
                        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                             cHS_WCACellLists[INT2, INT2](eps, sca, rd_, bv_,
                                                          ncellx_scale, balance_omp))
                elif ndim == 3:
                    if not use_cell_lists:
                        # non-frozen, 3d, cartesian, no cell lists
                        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                             cHS_WCA[INT3, INT2](eps, sca, rd_))
                    else:
                        # non-frozen, 3d, cartesian, use cell lists
                        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                             cHS_WCACellLists[INT3, INT2](eps, sca, rd_, bv_,
                                                          ncellx_scale, balance_omp))
                else:
                    raise Exception("HS_WCA: illegal ndim")
        elif exp == 6:
            if self.leesedwards:
                assert 'shear' in pot_kwargs, "Required argument 'shear' not defined in pot_kwargs."
                if ndim == 2:
                    if not use_cell_lists:
                        # non-frozen, 2d, periodic, no cell lists
                        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                             cHS_WCALeesEdwards[INT2, INT6](eps, sca, rd_, bv_, pot_kwargs['shear']))
                    else:
                        # non-frozen, 2d, periodic, use cell lists
                        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                             cHS_WCALeesEdwardsCellLists[INT2, INT6](eps, sca, rd_, bv_,
                                                                     pot_kwargs['shear'],
                                                                     ncellx_scale, balance_omp))
                elif ndim == 3:
                    if not use_cell_lists:
                        # non-frozen, 3d, periodic, no cell lists
                        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                             cHS_WCALeesEdwards[INT3, INT6](eps, sca, rd_, bv_, pot_kwargs['shear']))
                    else:
                        # non-frozen, 3d, periodic, use cell lists
                        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                             cHS_WCALeesEdwardsCellLists[INT3, INT6](eps, sca, rd_, bv_,
                                                                     pot_kwargs['shear'],
                                                                     ncellx_scale, balance_omp))
                else:
                    raise Exception("HS_WCA: illegal ndim")
            elif self.periodic:
                if ndim == 2:
                    if not use_cell_lists:
                        # non-frozen, 2d, periodic, no cell lists
                        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                             cHS_WCAPeriodic[INT2, INT6](eps, sca, rd_, bv_))
                    else:
                        # non-frozen, 2d, periodic, use cell lists
                        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                             cHS_WCAPeriodicCellLists[INT2, INT6](eps, sca, rd_, bv_,
                                                                  ncellx_scale, balance_omp))
                elif ndim == 3:
                    if not use_cell_lists:
                        # non-frozen, 3d, periodic, no cell lists
                        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                             cHS_WCAPeriodic[INT3, INT6](eps, sca, rd_, bv_))
                    else:
                        # non-frozen, 3d, periodic, use cell lists
                        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                             cHS_WCAPeriodicCellLists[INT3, INT6](eps, sca, rd_, bv_,
                                                                  ncellx_scale, balance_omp))
                else:
                    raise Exception("HS_WCA: illegal ndim")
            else:
                if ndim == 2:
                    if not use_cell_lists:
                        # non-frozen, 2d, cartesian, no cell lists
                        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                             cHS_WCA[INT2, INT6](eps, sca, rd_))
                    else:
                        # non-frozen, 2d, cartesian, use cell lists
                        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                             cHS_WCACellLists[INT2, INT6](eps, sca, rd_, bv_,
                                                          ncellx_scale, balance_omp))
                elif ndim == 3:
                    if not use_cell_lists:
                        # non-frozen, 3d, cartesian, no cell lists
                        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                             cHS_WCA[INT3, INT6](eps, sca, rd_))
                    else:
                        # non-frozen, 3d, cartesian, use cell lists
                        self.thisptr = shared_ptr[_pele.cBasePotential](<_pele.cBasePotential*> new
                             cHS_WCACellLists[INT3, INT6](eps, sca, rd_, bv_,
                                                          ncellx_scale, balance_omp))
                else:
                    raise Exception("HS_WCA: illegal ndim")
        else:
            raise Exception("HS_WCA: illegal exp (currently only 2 & 6 are supported)")


        cdef cppFrozenPotentialWrapper *frozen_pot
        if use_frozen:
            # replace self.thisptr a wrapped version of the existing potential
            frozen_pot = new cppFrozenPotentialWrapper(self.thisptr, rc_, fd_)
            # now the shared_ptr takes ownership of it
            self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*> frozen_pot )
