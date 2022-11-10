""" Test for generating mixed descent run in C++
"""
from matplotlib import use
import numpy as np
from pele.optimize import GenericMixedDescent, ModifiedFireCPP, CVODEBDFOptimizer
from pele.potentials import InversePower

from .potential_fixture import potential_initial_and_final_conditions


def test_mixed_descent(potential_initial_and_final_conditions):
    # use radii, box_length and coordinates from the C++ test test_mixed_descent.cpp

    (
        potential,
        initial_coordinates,
        expected_final_coordinates,
        _,
    ) = potential_initial_and_final_conditions

    optimizer_convex = ModifiedFireCPP(
        initial_coordinates, potential, fdec=0.5, tol=1e-9
    )

    optimizer_non_convex = CVODEBDFOptimizer(
        potential, initial_coordinates, rtol=1e-5, atol=1e-5, tol=1e-9
    )

    # optimizer_non_convex = ModifiedFireCPP(x, potential, tol=1e-9)

    opt_mixed_descent = GenericMixedDescent(
        potential,
        initial_coordinates,
        optimizer_non_convex,
        optimizer_convex,
        tol=1e-9,
        translation_offset=1,
        steps_before_convex_check=50,
    )
    res = opt_mixed_descent.run(2000)
    # res = optimizer_non_convex.run(10000)
    print(res)
    # check that the number of steps is the same as the C++ version
    assert res.nsteps == 494
    assert np.allclose(res.coords, expected_final_coordinates)


# TODO: This test needs to be written for a larger number of particles
# in a sparse system where we expect to use this
def test_mixed_descent_with_extension(potential_initial_and_final_conditions):
    (
        potential,
        initial_coordinates,
        expected_final_coordinates,
        pot_dict,
    ) = potential_initial_and_final_conditions

    pot_dict["radii"] *= 1.2
    pot_dict["eps"] *= 1e-6
    print(pot_dict)

    potential_extension = InversePower(**pot_dict)

    optimizer_convex = ModifiedFireCPP(
        initial_coordinates, potential, fdec=0.5, tol=1e-9
    )

    optimizer_non_convex = CVODEBDFOptimizer(
        potential, initial_coordinates, rtol=1e-5, atol=1e-5, tol=1e-9
    )

    # optimizer_non_convex = ModifiedFireCPP(x, potential, tol=1e-9)

    opt_mixed_descent = GenericMixedDescent(
        potential,
        initial_coordinates,
        optimizer_non_convex,
        optimizer_convex,
        tol=1e-9,
        translation_offset=1,
        steps_before_convex_check=50,
    )
    res = opt_mixed_descent.run(2000)
    # res = optimizer_non_convex.run(10000)
    print(res)
    # check that the number of steps is the same as the C++ version
    assert res.nsteps == 494
    assert np.allclose(res.coords, expected_final_coordinates)
