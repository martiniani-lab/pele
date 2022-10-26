"""
Benchmarks for the CVODE solver.
Helps profile the performance of the CVODE solver.
"""

import argparse

import numpy as np
from pele.optimize import CVODEBDFOptimizer, HessianType
from pele.potentials import InversePower
from pele.utils.cell_scale import get_box_length, get_ncellsx_scale


def generate_initial_conditions_and_potential(n_part, seed=0):
    """
    Generates a periodic inverse power potential with n particles
    and initial conditions for the minimizer for benchmarking.
    """
    rng = np.random.default_rng(seed)
    n_part_by_2 = n_part // 2
    
    r1, r2 = 1.0, 1.4
    rstd1, rstd2 = 0.05*r1, 0.05*r2
    
    phi = 0.9
    n_dim = 2
    
    
    radii =  np.array(
        list(r1 + rstd1 * rng.normal(size=n_part_by_2))
        + list(r2 + rstd2 * rng.normal(size=n_part - n_part_by_2))
    )
    box_length = get_box_length(radii, 2, 0.9)
    ncellx_scale = get_ncellsx_scale(radii, [box_length, box_length], omp_threads=None)
    
    initial_coords = rng.uniform(size=n_part*n_dim) * box_length
    
    power = 2.5
    eps = 1.0
    use_cell_lists = True if n_part > 64 else False
    
    potential = InversePower(power, eps, use_cell_lists=use_cell_lists, ndim=n_dim, radii=radii, boxvec=[box_length, box_length])
    
    return initial_coords, potential


        
        
        

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Benchmark CVODE solver")
    
    parser.add_argument("--n_particles", type=int, default=16, help="Number of particles")
    parser.add_argument("--n_ensemble", type=int, default=10, help="Number of ensembles")
    
    cvode_args = {
        "rtol" : 1e-10,
        "atol" : 1e-10,
        "tol" : 1e-9,
        "hessian_type" : HessianType.ITERATIVE,
    }
    args = parser.parse_args()
    
    
    

