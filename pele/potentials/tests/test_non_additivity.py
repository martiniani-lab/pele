from pytest import param
from pele.potentials import InversePower
from pele.potentials import InversePowerStillingerCutQuad
import numpy as np
import os
from pele.optimize import lbfgs_cpp

DATA_DIR = os.path.join(os.path.dirname(__file__), "non_additive_test_data")


def test_check_cell_lists_works_non_additive():
    parameters = {
        "n_part": 4096,
        "ndim": 2,
        "phi": 1.2,
        "seed": 0,
        "dmin_by_dmax": 0.449,
        "d_mean": 1.0,
        "use_cell_lists": 0,
        "power": 12.0,  # goes repulsively as r^-12
        "v0": 1.0,
        "non_additivity": 0.2,
        "cutoff_factor": 1.25,
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
    parameters["boxl"] = box_length

    potential_without_cell_lists = InversePowerStillingerCutQuad(**parameters)
    parameters["use_cell_lists"] = 1
    potential_with_cell_lists = InversePowerStillingerCutQuad(**parameters)

    final_coords = lbfgs_cpp(initial_coords, potential_with_cell_lists)["coords"]

    energy_with_cell_lists = potential_with_cell_lists.getEnergy(final_coords)
    energy_without_cell_lists = potential_without_cell_lists.getEnergy(final_coords)

    assert energy_with_cell_lists == energy_without_cell_lists
    return
