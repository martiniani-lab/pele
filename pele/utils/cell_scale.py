"""
A set of utilities functions that determines the appropriate arguments for putting particles in a box.
"""
import os

import numpy as np


def _omp_thread_count():
    """what omp_get_max_threads() returns: OMP_NUM_THREADS, else the usable cores"""
    if os.environ.get("OMP_NUM_THREADS"):
        return int(os.environ["OMP_NUM_THREADS"].split(",")[0])
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count()


def get_ncellsx_scale(radii, boxv, omp_threads=None):
    """gets the cell scale for given radii and boxv"""
    if omp_threads is None:
        omp_threads = _omp_thread_count()
    ndim = len(boxv)
    ncellsx_max = max(omp_threads, int(np.power(radii.size, 1.0 / ndim)))
    rcut = np.amax(radii) * 2
    ncellsx = 1 * boxv[0] / rcut
    if ncellsx <= ncellsx_max:
        ncellsx_scale = (
            1 if ncellsx >= omp_threads else np.ceil(omp_threads / ncellsx)
        )
    else:
        ncellsx_scale = ncellsx_max / ncellsx
    print("ncellsx: {}, ncellsx_scale: {}".format(ncellsx, ncellsx_scale))
    return ncellsx_scale


def get_box_length(radii, dim: int, phi: float) -> float:
    """gets the box length for a given number
    of particles and given packing fraction
    """

    if dim == 3:
        vol_spheres = np.sum(4.0 / 3.0 * np.pi * radii**3)
        box_length = (vol_spheres / phi) ** (1 / 3.0)
        return box_length
    elif dim == 2:
        vol_discs = np.sum(np.pi * radii**2)
        box_length = (vol_discs / phi) ** (1 / 2.0)
        return box_length
    else:
        raise NotImplementedError(
            " dimensions other than 2/3 have not been implemented"
        )
