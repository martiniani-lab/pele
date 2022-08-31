"""
A set of utilities functions that determines the appropriate arguments for putting particles in a box.
"""
import omp_thread_count
import numpy as np


def get_ncellsx_scale(radii, boxv, omp_threads=None):
    """ gets the cell scale for given radii and boxv
    """
    if omp_threads is None:
        omp_threads = omp_thread_count.get_thread_count()
    ndim = len(boxv)
    ncellsx_max = max(omp_threads, int(np.power(radii.size, 1. / ndim)))
    rcut = np.amax(radii) * 2
    ncellsx = 1 * boxv[0] / rcut
    if ncellsx <= ncellsx_max:
        ncellsx_scale = 1 if ncellsx >= omp_threads else np.ceil(omp_threads / ncellsx)
    else:
        ncellsx_scale = ncellsx_max / ncellsx
    print("ncellsx: {}, ncellsx_scale: {}".format(ncellsx, ncellsx_scale))    
    return ncellsx_scale


def get_box_length(radii, dim : int, phi : float) -> float:
    """ gets the box length for a given number
        of particles and given packing fraction
    """
    
    if dim == 3:
        vol_spheres = np.sum(4./3. * np.pi*radii**3)
        box_length = (vol_spheres/phi)**(1/3.)
        return box_length
    elif dim == 2:
        vol_discs = np.sum(np.pi*radii**2)
        box_length = (vol_discs/phi)**(1/2.)
        return box_length
    else:
        raise NotImplementedError(" dimensions other than 2/3 have not been implemented")
