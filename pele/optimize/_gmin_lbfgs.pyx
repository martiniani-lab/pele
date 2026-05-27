"""
# distutils: language = C++
# cython: language_level=3str

Bridge to GMIN's native MYLBFGS optimizer.

Built only when pele is configured with -DWITH_GMIN=ON. The compiled .so
links statically against a copy of GMIN's libgminlib.a from which the
real POTENTIAL has been removed via archive surgery; the Fortran shim in
source/gmin/pele_gmin_shim.f90 supplies the replacement, which calls
back into the extern "C" symbol pele_gmin_callback defined in
source/gmin/pele_gmin_bridge.cpp. On module import we register the
Cython callback (`_cython_energy_grad_cb`) via pele_gmin_set_callback.

The intended use is paper-revision data for arxiv 2409.12113: running
GMIN's L-BFGS quench on the same soft-sphere configurations the paper
uses, with pele's existing basin-geometry analysis on the output.

GMIN holds optimizer state in a global Fortran module (COMMONS); this
wrapper is NOT reentrant and not thread-safe — serialize calls.
"""

# cython: language_level=3

import numpy as np
cimport numpy as np
cimport cython


cdef object _current_potential = None


cdef extern from *:
    """
    /* C-linkage entry points provided by source/gmin/pele_gmin_bridge.cpp
       and source/gmin/pele_gmin_wrapper.f90 respectively. */
    extern "C" {
        typedef void (*pele_gmin_cb_t)(int n, const double* x, double* grad,
                                       double* energy, int gradt);
        void pele_gmin_set_callback(pele_gmin_cb_t cb);
        void pele_gmin_mylbfgs(int natoms, int n, int m, double* xcoords,
                               double eps, int itmax,
                               int* mflag, double* energy, int* itdone);
        void pele_gmin_cgmin(int natoms, int n, double* xcoords,
                             double eps, int itmax,
                             int* mflag, double* energy, int* itdone);
    }
    """
    ctypedef void (*pele_gmin_cb_t)(int n, const double* x, double* grad,
                                     double* energy, int gradt) noexcept
    void pele_gmin_set_callback(pele_gmin_cb_t cb)
    void pele_gmin_mylbfgs(int natoms, int n, int m, double* xcoords,
                           double eps, int itmax,
                           int* mflag, double* energy, int* itdone)
    void pele_gmin_cgmin(int natoms, int n, double* xcoords,
                         double eps, int itmax,
                         int* mflag, double* energy, int* itdone)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _cython_energy_grad_cb(int n, const double* x, double* grad,
                                  double* energy, int gradt) noexcept with gil:
    """Forwards a Fortran-side energy/gradient request to the Python
    potential currently registered in _current_potential.

    Energy is always computed. Gradient is computed when gradt != 0.
    """
    global _current_potential
    cdef int i
    if _current_potential is None:
        energy[0] = 0.0
        if gradt:
            for i in range(n):
                grad[i] = 0.0
        return

    cdef np.ndarray[double, ndim=1, mode='c'] x_np = np.empty(n, dtype=np.float64)
    for i in range(n):
        x_np[i] = x[i]

    cdef np.ndarray[double, ndim=1, mode='c'] g_np
    cdef double e
    if gradt:
        e, g_np = _current_potential.getEnergyGradient(x_np)
        energy[0] = e
        for i in range(n):
            grad[i] = g_np[i]
    else:
        energy[0] = _current_potential.getEnergy(x_np)


# Register the Cython callback with the C++ bridge on module import.
pele_gmin_set_callback(_cython_energy_grad_cb)


def gmin_mylbfgs(potential, x0, int M=4, double eps=1e-7, int itmax=10000):
    """Minimize `potential` from `x0` using GMIN's native MYLBFGS.

    Parameters
    ----------
    potential : pele BasePotential
        Must implement getEnergyGradient(x) -> (energy, grad).
    x0 : array-like of float, shape (3*N,)
        Initial coordinates (flattened). Length must be a multiple of 3.
    M : int
        L-BFGS history size (default 4, matching GMIN's default).
    eps : float
        Convergence threshold on RMS gradient force.
    itmax : int
        Maximum iterations.

    Returns
    -------
    x : ndarray, shape (3*N,)
        Minimized coordinates.
    energy : float
        Final energy.
    converged : bool
        True if MYLBFGS reported convergence.
    niter : int
        Iterations performed.
    """
    global _current_potential
    if _current_potential is not None:
        raise RuntimeError(
            "gmin_mylbfgs is not reentrant — a previous call did not "
            "clear the potential registration"
        )

    cdef np.ndarray[double, ndim=1, mode='c'] x = np.ascontiguousarray(
        np.asarray(x0, dtype=np.float64).ravel()
    )
    cdef int n = x.shape[0]
    if n % 3 != 0:
        raise ValueError(
            "x0 length must be a multiple of 3 (GMIN expects 3*NATOMS coordinates), "
            "got length %d" % n
        )
    cdef int natoms = n // 3
    cdef double energy = 0.0
    cdef int mflag = 0
    cdef int itdone = 0

    _current_potential = potential
    try:
        pele_gmin_mylbfgs(natoms, n, M, &x[0], eps, itmax,
                          &mflag, &energy, &itdone)
    finally:
        _current_potential = None

    return x, energy, bool(mflag), itdone


def gmin_cgmin(potential, x0, double eps=1e-7, int itmax=10000):
    """Minimize `potential` from `x0` using GMIN's native CGMIN (conjugate gradient).

    Parameters
    ----------
    potential : pele BasePotential
        Must implement getEnergyGradient(x) -> (energy, grad).
    x0 : array-like of float, shape (3*N,)
        Initial coordinates (flattened). Length must be a multiple of 3.
    eps : float
        Convergence threshold on RMS gradient force (maps to COMMONS::GMAX).
    itmax : int
        Maximum iterations.

    Returns
    -------
    x : ndarray, shape (3*N,)
        Minimized coordinates.
    energy : float
        Final energy.
    converged : bool
        True if CGMIN reported convergence (RMS < eps).
    niter : int
        Iterations performed.
    """
    global _current_potential
    if _current_potential is not None:
        raise RuntimeError(
            "gmin_cgmin is not reentrant — a previous call did not "
            "clear the potential registration"
        )

    cdef np.ndarray[double, ndim=1, mode='c'] x = np.ascontiguousarray(
        np.asarray(x0, dtype=np.float64).ravel()
    )
    cdef int n = x.shape[0]
    if n % 3 != 0:
        raise ValueError(
            "x0 length must be a multiple of 3 (GMIN expects 3*NATOMS coordinates), "
            "got length %d" % n
        )
    cdef int natoms = n // 3
    cdef double energy = 0.0
    cdef int mflag = 0
    cdef int itdone = 0

    _current_potential = potential
    try:
        pele_gmin_cgmin(natoms, n, &x[0], eps, itmax,
                        &mflag, &energy, &itdone)
    finally:
        _current_potential = None

    return x, energy, bool(mflag), itdone
