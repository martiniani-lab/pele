"""Test for CVODE.

Tests whether the CVODE results through the python wrapper match with the C++
results.

TODO: The results are hardcoded for the current test_cvode.py.
      We need to later ensure that we automate the comparison.
"""

from pele.optimize import CVODEBDFOptimizer, HessianType
import numpy as np

from .potential_fixture import potential_initial_and_final_conditions


def test_cvode_compare_with_cpp(potential_initial_and_final_conditions):
    """
    Test the CVODE results through the python wrapper match with the C++ results.
    """

    (
        potential,
        initial_coordinates,
        expected_final_coordinates,
        _,
    ) = potential_initial_and_final_conditions
    cvode_optimizer = CVODEBDFOptimizer(
        potential,
        initial_coordinates,
        rtol=1e-5,
        atol=1e-5,
        tol=1e-9,
        hessian_type=HessianType.DENSE,
        use_newton_stop_criterion=False,
        save_trajectory=True,
    )
    res = cvode_optimizer.run(10000)
    final_coords = res.coords

    # check whether right minima is reached
    assert np.allclose(final_coords, expected_final_coordinates)
    # Expect steps to match with C++ results from test_cvode.cpp
    N_CVODE_CPP_STEPS = 387
    assert res.nsteps == N_CVODE_CPP_STEPS
    assert len(res.time_trajectory) == res.nsteps
    assert len(res.gradient_norm_trajectory) == res.nsteps
    assert res.coordinate_trajectory.shape == (N_CVODE_CPP_STEPS, 32)
    assert res.gradient_trajectory.shape == (N_CVODE_CPP_STEPS, 32)
    return True


def test_iterative_works_cvode(potential_initial_and_final_conditions):
    """
    Test the CVODE results through the python wrapper match with the C++ results.
    """
    (
        potential,
        initial_coordinates,
        expected_final_coordinates,
        _,
    ) = potential_initial_and_final_conditions
    cvode_optimizer = CVODEBDFOptimizer(
        potential,
        initial_coordinates,
        rtol=1e-10,
        atol=1e-10,
        tol=1e-9,
        hessian_type=HessianType.ITERATIVE,
        use_newton_stop_criterion=True,
        save_trajectory=True,
    )
    res = cvode_optimizer.run(10000)
    print(res)
    final_coords = res.coords

    # check whether right minima is reached
    assert np.allclose(final_coords, expected_final_coordinates)
    # Expect steps to match with C++ results from test_cvode.cpp
    assert len(res.time_trajectory) == res.nsteps
    assert len(res.gradient_norm_trajectory) == res.nsteps
    return True
