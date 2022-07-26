"""
Tests whether the CVODE results through the python wrapper match with the C++ results.

TODO: The results are hardcoded for the current test_cvode.py.
      We need to later ensure that we automate the comparison.
"""
import pytest
from inverse_power_system import setup_test_potential, STARTING_COORDINATES
from pele.optimize import CVODEBDFOptimizer
import numpy as np

from pele.potentials import InversePower


# define potential and coordinates
@pytest.fixture(scope="module")
def potential_initial_and_final_conditions():
    radii = np.array(
        [
            0.982267,
            0.959526,
            1.00257,
            0.967356,
            1.04893,
            0.97781,
            0.954191,
            0.988939,
            0.980737,
            0.964811,
            1.04198,
            0.926199,
            0.969865,
            1.08593,
            1.01491,
            0.968892,
        ]
    )
    BOX_LENGTH = 7.40204
    box_vec = np.array([BOX_LENGTH, BOX_LENGTH])
    starting_coordinates = [
        1.8777,
        3.61102,
        1.70726,
        6.93457,
        2.14539,
        2.55779,
        4.1191,
        7.02707,
        0.357781,
        4.92849,
        1.28547,
        2.83375,
        0.775204,
        6.78136,
        6.27529,
        1.81749,
        6.02049,
        6.70693,
        5.36309,
        4.6089,
        4.9476,
        5.54674,
        0.677836,
        6.04457,
        4.9083,
        1.24044,
        5.09315,
        0.108931,
        2.18619,
        6.52932,
        2.85539,
        2.30303,
    ]
    dim = 2
    power = 2.5
    eps = 1.0
    use_cell_lists = False
    # Setup the potential
    potential = InversePower(
        power,
        eps,
        radii,
        use_cell_lists=use_cell_lists,
        ndim=dim,
        boxvec=box_vec,
    )
    # Minimum of the basin of attraction the initial coordinates give with the basin of attraction
    expected_corresponding_minimum = np.array(
        [
            1.57060565,
            3.91255814,
            1.62235538,
            7.45540734,
            2.80790972,
            1.56646306,
            3.46353796,
            7.15479982,
            -0.32415793,
            4.31534937,
            0.98863622,
            2.12570204,
            -0.12728098,
            8.04595801,
            6.50148221,
            2.39787121,
            6.72015756,
            6.24428002,
            5.21017732,
            3.83258729,
            4.83143628,
            5.71769049,
            1.06446307,
            5.69991786,
            4.63170217,
            2.07204592,
            5.36197688,
            0.28591197,
            2.89994155,
            5.31564514,
            3.37229792,
            3.43821332,
        ]
    )
    return (potential, starting_coordinates, expected_corresponding_minimum)


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
