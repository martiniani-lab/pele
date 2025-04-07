from pele.optimize import CosineGradientDescent
from pele.potentials import Harmonic
import numpy as np


def test_cos_gd():
    potential = Harmonic(np.array([0.0, 0.0]), 1, bdim=2)

    opt = CosineGradientDescent(potential, np.array([1.0, 1.0]))
    res = opt.run(1000)
    assert np.allclose(res.coords, [0.0, 0.0], atol=1e-3)
