from pele.potentials import RadialGaussian
import numpy as np


def test_radial_gaussian():
    origin = np.array([0.0, 0.0, 0.0])
    bdim = 3
    k = 1.0
    l0 = 1.0
    check_coords = 0.5 * np.array([1.0, 0.3, 4.0])
    potential = RadialGaussian(origin, k, l0, bdim=bdim)
    # calculate the numerical gradient
    n_grad = np.zeros(bdim)
    delta = 1e-6
    for i in range(bdim):
        check_coords[i] += delta
        e_plus = potential.getEnergy(check_coords)
        check_coords[i] -= 2 * delta
        e_minus = potential.getEnergy(check_coords)
        check_coords[i] += delta
        n_grad[i] = (e_plus - e_minus) / (2 * delta)

    e, grad = potential.getEnergyGradient(check_coords)
    print(n_grad)
    print(grad)
    for i in range(bdim):
        assert np.isclose(
            n_grad[i], grad[i], rtol=1e-5, atol=1e-5
        ), "The gradient is not correct for dimension {}".format(i)
