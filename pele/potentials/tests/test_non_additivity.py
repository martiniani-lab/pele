from pytest import param
from pele.potentials import InversePower
from pele.potentials import InversePowerStillingerCutQuad
import numpy as np
import os
from pele.optimize import lbfgs_cpp

DATA_DIR = os.path.join(os.path.dirname(__file__), "non_additive_test_data")


def test_check_cell_lists_works_non_additive():
    """Test that the cell lists work for non-additive potentials."""

    def check_vector_almost_equal(v1, v2, tol=1e-10):
        """Check that two vectors are almost equal."""
        for i in range(len(v1)):
            if abs(v1[i] - v2[i]) > tol:
                print(v1[i] - v2[i])
                assert False

    def check_matrix_almost_equal(m1, m2, tol=1e-10):
        """Check that two matrices are almost equal."""
        assert m1.shape == m2.shape

        for i in range(len(m1)):
            for j in range(len(m1[0])):
                if abs(m1[i][j] - m2[i][j]) > tol:
                    print(m1[i][j] - m2[i][j])
                    assert False

    parameters = {
        "n_part": 32,
        "ndim": 2,
        "phi": 1.2,
        "seed": 0,
        "dmin_by_dmax": 0.449,
        "d_mean": 1.0,
        "use_cell_lists": 0,
        "pow": 2,
        "eps": 1.0,
        "non_additivity": 0.2,
    }
    radii = np.loadtxt(os.path.join(DATA_DIR, "radii.txt"))
    box_length = np.loadtxt(os.path.join(DATA_DIR, "box_length.txt"))
    initial_coords = np.loadtxt(os.path.join(DATA_DIR, "initial_coords.txt"))

    parameters.pop("n_part")
    parameters.pop("phi")
    parameters.pop("seed")
    parameters.pop("dmin_by_dmax")
    parameters.pop("d_mean")

    parameters["radii"] = radii
    parameters["boxvec"] = np.array([box_length] * parameters["ndim"])

    print(parameters)
    potential_without_cell_lists = InversePower(**parameters)
    parameters["use_cell_lists"] = 1
    potential_with_cell_lists = InversePower(**parameters)

    final_coords = lbfgs_cpp(initial_coords, potential_with_cell_lists)["coords"]

    energy_with_cell_lists = potential_with_cell_lists.getEnergy(final_coords)
    energy_without_cell_lists = potential_without_cell_lists.getEnergy(final_coords)

    assert (
        energy_with_cell_lists - energy_without_cell_lists
        < energy_with_cell_lists * 1e-10
    )

    (
        energy_with_cell_lists,
        grad_with_cell_lists,
    ) = potential_with_cell_lists.getEnergyGradient(final_coords)

    (
        energy_without_cell_lists,
        grad_without_cell_lists,
    ) = potential_without_cell_lists.getEnergyGradient(final_coords)

    assert (
        energy_with_cell_lists - energy_without_cell_lists
        < energy_with_cell_lists * 1e-10
    )

    check_vector_almost_equal(grad_with_cell_lists, grad_without_cell_lists)

    (
        energy_with_cell_lists,
        grad_with_cell_lists,
        hess_with_cell_lists,
    ) = potential_with_cell_lists.getEnergyGradientHessian(final_coords)

    (
        energy_without_cell_lists,
        grad_without_cell_lists,
        hess_without_cell_lists,
    ) = potential_without_cell_lists.getEnergyGradientHessian(final_coords)

    assert (
        energy_with_cell_lists - energy_without_cell_lists
        < energy_with_cell_lists * 1e-10
    )

    check_vector_almost_equal(grad_with_cell_lists, grad_without_cell_lists)

    check_matrix_almost_equal(hess_with_cell_lists, hess_without_cell_lists)

    return
