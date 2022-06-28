"""
Tests whether the CVODE results through the python wrapper match with the C++ results.

TODO: The results are hardcoded for the current test_cvode.py.
      We need to later ensure that we automate the comparison.
"""
from inverse_power_system import setup_test_potential, STARTING_COORDINATES
from pele.optimize import CVODEBDFOptimizer
import numpy as np


def test_cvode_compare_with_cpp():
    """
    Test the CVODE results through the python wrapper match with the C++ results.
    """
    # Setup the potential
    potential = setup_test_potential()
    coords = STARTING_COORDINATES
    cvode_optimizer = CVODEBDFOptimizer(potential, coords, rtol=1e-5, atol=1e-5, tol=1e-9)
    res = cvode_optimizer.run(10000)
    print(res)
    final_coords = res.coords
    
    test_final = np.array([ 1.57060565,  3.91255814,  1.62235538,  7.45540734,  2.80790972,
        1.56646306,  3.46353796,  7.15479982, -0.32415793,  4.31534937,
        0.98863622,  2.12570204, -0.12728098,  8.04595801,  6.50148221,
        2.39787121,  6.72015756,  6.24428002,  5.21017732,  3.83258729,
        4.83143628,  5.71769049,  1.06446307,  5.69991786,  4.63170217,
        2.07204592,  5.36197688,  0.28591197,  2.89994155,  5.31564514,
        3.37229792,  3.43821332])
    
    # check whether right minima is reached
    assert(np.allclose(final_coords, test_final))
    
    
    # Expect steps to match with C++ results from test_cvode.cpp
    assert(res.nsteps == 363)
    
    
    
    
    
    
    
    
    
test_cvode_compare_with_cpp()