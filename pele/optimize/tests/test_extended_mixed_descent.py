"""
Tests for Extended Mixed Descent.
#TODO: Extended Mixed Descent needs to be folded into Generic Mixed Descent.
"""

import numpy as np

from pele.optimize import ExtendedMixedOptimizer

from .potential_fixture import (potential_extension,
                                potential_initial_and_final_conditions)


def test_extended_mixed_descent_compare_with_cpp(
    potential_initial_and_final_conditions, potential_extension
):
    """
    Test the CVODE results through the python wrapper match with the C++ results.
    """

    (
        potential,
        initial_coordinates,
        expected_final_coordinates,
    ) = potential_initial_and_final_conditions
    emd = ExtendedMixedOptimizer(
        potential,
        initial_coordinates,
        potential_extension,
        rtol=1e-5,
        atol=1e-5,
        tol=1e-9,
        iterative=False,
    )
    res = emd.run(10000)
    final_coords = res.coords
    # check whether right minima is reached
    assert np.allclose(final_coords, expected_final_coordinates)
    # Expect steps to match with C++ results from test_cvode.cpp
    return True


def test_extended_mixed_descent_compare_with_cpp(
    potential_initial_and_final_conditions, potential_extension
):
    """
    Test the CVODE results through the python wrapper match with the C++ results.
    """

    (
        potential,
        initial_coordinates,
        expected_final_coordinates,
    ) = potential_initial_and_final_conditions
    emd = ExtendedMixedOptimizer(
        potential,
        initial_coordinates,
        potential_extension,
        rtol=1e-5,
        atol=1e-5,
        tol=1e-9,
        iterative=True,
    )
    res = emd.run(10000)
    final_coords = res.coords
    # check whether right minima is reached
    assert np.allclose(final_coords, expected_final_coordinates)
    # Expect steps to match with C++ results from test_cvode.cpp
    return True
