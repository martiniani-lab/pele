"""
Tests whether the CVODE results through the python wrapper match with the C++ results.

TODO: The results are hardcoded for the current test_cvode.py.
      We need to later ensure that we automate the comparison.
"""

from pele.optimize import CVODEBDFOptimizer
import numpy as np

from .potential_fixture import potential_initial_and_final_conditions


def test_cvode_compare_with_cpp(potential_initial_and_final_conditions):
    """
    Test the CVODE results through the python wrapper match with the C++ results.
    """
    
    potential, initial_coordinates, expected_final_coordinates = potential_initial_and_final_conditions
    cvode_optimizer = CVODEBDFOptimizer(
        potential,
        initial_coordinates,
        rtol=1e-5,
        atol=1e-5,
        tol=1e-9,
        iterative=False,
        use_newton_stop_criterion=False,
    )
    res = cvode_optimizer.run(10000)
    print(res)
    final_coords = res.coords

    # check whether right minima is reached
    assert np.allclose(final_coords, expected_final_coordinates)
    # Expect steps to match with C++ results from test_cvode.cpp
    assert res.nsteps == 363
    return True


def test_iterative_works_cvode(potential_initial_and_final_conditions):
    """
    Test the CVODE results through the python wrapper match with the C++ results.
    """
    potential, initial_coordinates, expected_final_coordinates = potential_initial_and_final_conditions
    cvode_optimizer = CVODEBDFOptimizer(
        potential,
        initial_coordinates,
        rtol=1e-10,
        atol=1e-10,
        tol=1e-9,
        iterative=True,
        use_newton_stop_criterion=True,
    )
    res = cvode_optimizer.run(10000)
    print(res)
    final_coords = res.coords

    # check whether right minima is reached
    assert np.allclose(final_coords, expected_final_coordinates)
    # Expect steps to match with C++ results from test_cvode.cpp
    return True
