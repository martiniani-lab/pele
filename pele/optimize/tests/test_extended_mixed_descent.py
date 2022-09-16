"""
Tests for Extended Mixed Descent.
#TODO: Extended Mixed Descent needs to be folded into Generic Mixed Descent.
"""

from .potential_fixture import potential_initial_and_final_conditions
from pele.optimize import ExtendedMixedOptimizer
import numpy as np
from pele.potentials import InversePower




def test_extended_mixed_descent_compare_with_cpp(potential_initial_and_final_conditions):
    """
    Test the CVODE results through the python wrapper match with the C++ results.
    """
    
    potential, initial_coordinates, expected_final_coordinates, pot_dict = potential_initial_and_final_conditions
    
    potential_extension = InversePower(**pot_dict)
    
    
    emd = ExtendedMixedOptimizer(
        potential,
        initial_coordinates,
        potential_extension,
        rtol=1e-4,
        atol=1e-4,
        tol=1e-9,
        T=1,
        iterative=False,
    )
    res = emd.run(200)
    final_coords = res.coords
    print(res.nsteps)
    # check whether right minima is reached
    assert np.allclose(final_coords, expected_final_coordinates)
    # Expect steps to match with C++ results from test_cvode.cpp
    return True


def test_extended_mixed_descent_compare_with_cpp_iterative(potential_initial_and_final_conditions):
    """
    Test the CVODE results through the python wrapper match with the C++ results.
    """
    
    potential, initial_coordinates, expected_final_coordinates, pot_dict = potential_initial_and_final_conditions
    
    potential_extension = InversePower(**pot_dict)
    emd = ExtendedMixedOptimizer(
        potential,
        initial_coordinates,
        potential_extension,
        rtol=1e-4,
        atol=1e-4,
        tol=1e-9,
        T=60,
        iterative=True,
    )
    res = emd.run(10000)
    print(res.nsteps)
    final_coords = res.coords
    # check whether right minima is reached
    assert np.allclose(final_coords, expected_final_coordinates)
    # Expect steps to match with C++ results from test_cvode.cpp
    return True
