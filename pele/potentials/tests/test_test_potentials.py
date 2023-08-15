from pele.potentials import NegativeCosProduct
import numpy as np

np.random.seed(1234)


def test_negative_cos_product():
    """
    Test the negative cos product potential.
    the corresponding potential is V(x) = - cos(x[0]) cos(x[1])...cos(x[n])
    """
    dim=5
    period = 2*np.pi
    potential = NegativeCosProduct(dim=dim, period=period)
    x = np.random.rand(dim)
    cos_x = np.cos(x)
    sin_x = np.sin(x)
    assert np.isclose(potential.getEnergy(x), -np.prod(np.cos(x))), "The energy is not correct"
    e, g = potential.getEnergyGradient(x)
    assert np.isclose(e, -np.prod(np.cos(x))), "The energy is not correct"
    assert np.allclose(g, -e*sin_x/cos_x), "The gradient is not correct"

