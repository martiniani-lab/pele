"""Test saving the trajectory data.

Tests for saving the trajectory data with three different optimizers
for which this is currently implemented,
1. Modified FIRE
2. CVODE
3. Gradient Descent

All three optimizers derive from the ODEBasedOptimizer class.
"""

from pele.optimize import (
    CVODEBDFOptimizer,
    GradientDescent_CPP,
    ModifiedFireCPP,
)

from .potential_fixture import potential_initial_and_final_conditions


def test_trajectory_saving_works_every_step(
    potential_initial_and_final_conditions,
):
    """Test that the trajectory is saved every step."""
    (
        potential,
        initial_coordinates,
        _,
        _,
    ) = potential_initial_and_final_conditions

    cvode = CVODEBDFOptimizer(
        potential,
        initial_coordinates,
        save_trajectory=True,
        iterations_before_save=1,
    )
    fire = ModifiedFireCPP(
        initial_coordinates,
        potential,
        save_trajectory=True,
        iterations_before_save=1,
    )
    gradient_descent = GradientDescent_CPP(
        potential,
        initial_coordinates,
        save_trajectory=True,
        iterations_before_save=1,
    )

    optimizer_dict = {
        "cvode": cvode,
        "fire": fire,
        "gradient_descent": gradient_descent,
    }

    for key, optimizer in optimizer_dict.items():
        print("Testing optimizer: ", key)
        res = optimizer.run(100)
        assert len(res.time_trajectory) == res.nsteps
        assert len(res.gradient_norm_trajectory) == res.nsteps


def test_skipping_trajectory_saving(potential_initial_and_final_conditions):
    """Test that the trajectory saving when saving is skipped."""
    (
        potential,
        initial_coordinates,
        _,
        _,
    ) = potential_initial_and_final_conditions

    skip = 10
    cvode = CVODEBDFOptimizer(
        potential,
        initial_coordinates,
        save_trajectory=True,
        iterations_before_save=skip,
    )
    fire = ModifiedFireCPP(
        initial_coordinates,
        potential,
        save_trajectory=True,
        iterations_before_save=skip,
    )
    gradient_descent = GradientDescent_CPP(
        potential,
        initial_coordinates,
        save_trajectory=True,
        iterations_before_save=skip,
    )

    optimizer_dict = {
        "cvode": cvode,
        "fire": fire,
        "gradient_descent": gradient_descent,
    }

    for key, optimizer in optimizer_dict.items():
        print("Testing optimizer: ", key)
        res = optimizer.run(100)
        assert len(res.time_trajectory) == res.nsteps // skip
        assert len(res.gradient_norm_trajectory) == res.nsteps // skip
