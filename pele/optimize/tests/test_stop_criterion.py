from .potential_fixture import potential_initial_and_final_conditions

import numpy as np
from pele.optimize import LBFGS_CPP, StopCriterionType


def test_stop_criterion(potential_initial_and_final_conditions):
    """Test that different stop criteria work."""
    (
        potential,
        initial_conditions,
        _,
        _,
    ) = potential_initial_and_final_conditions
    res_list = []
    energy_list = []

    for criterion in StopCriterionType:
        print("Testing criterion: {}".format(criterion))
        print("Initial conditions: {}".format(initial_conditions))
        print("Potential: {}".format(potential))
        lbgs = LBFGS_CPP(
            initial_conditions,
            potential,
            stop_criterion=criterion,
        )
        print("done")
        lbgs.run()
        print("run done")
        result = lbgs.get_result()
        print("get result")
        res_list.append(result)
        energy_list.append(result["energy"])
    energy_list = np.array(energy_list)

    assert np.all(energy_list - energy_list[0] < 1e-5)
