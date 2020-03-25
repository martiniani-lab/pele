import omp_thread_count
import numpy as np


def get_ncellsx_scale(radii, boxv, omp_threads=None):
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
    print "ncellsx: {}, ncellsx_scale: {}".format(ncellsx, ncellsx_scale)    
    return ncellsx_scale