from re import A
from pele.potentials import PoweredCosineSum
from pele.optimize import CVODEBDFOptimizer, ExtendedMixedOptimizer
import numpy as np

np.random.seed(1234)


def test_negative_cos_product():
    """
    Test the negative cos product potential.
    the corresponding potential is V(x) = - cos(x[0]) cos(x[1])...cos(x[n])
    """
    dim = 5
    period = 2 * np.pi
    potential = PoweredCosineSum(dim=dim, period=period)

    # check thet the numerical gradients and hessians match
    x = np.random.uniform(-period / 2, period / 2, dim)
    eps = 1e-6
    gradient = np.zeros(dim)
    hessian = np.zeros((dim, dim))

    # compute the gradient numerically
    for i in range(dim):
        x[i] += eps
        plus = potential.getEnergy(x)
        x[i] -= 2 * eps
        minus = potential.getEnergy(x)
        x[i] += eps
        gradient[i] = (plus - minus) / (2 * eps)

    # compute the hessian numerically
    for i in range(dim):
        x[i] += eps
        _, plus = potential.getEnergyGradient(x)
        x[i] -= 2 * eps
        _, minus = potential.getEnergyGradient(x)
        x[i] += eps
        hessian[i] = (plus - minus) / (2 * eps)

    # check that the numerical and analytical gradients match
    e, g1, h = potential.getEnergyGradientHessian(x)
    e, g2 = potential.getEnergyGradient(x)
    assert np.allclose(g1, g2)
    assert np.allclose(gradient, g1)
    assert np.allclose(hessian, h)


def test_minimum_convergence():
    dims = range(6, 7)
    for dim in dims:
        start_coords = 3 * np.ones(dim) / 8
        period = 1.0
        potential = PoweredCosineSum(dim=dim, period=period)
        opt = CVODEBDFOptimizer(potential, start_coords)
        res = opt.run(1000)
        assert np.allclose(res.coords, 0.0, atol=1e-3)


def test_mxd_works_with_neg_cos():
    """TODO: test doesn't hit the Newton step"""
    dims = [1024]
    for dim in dims:
        start_coords = 3 * np.ones(dim) / 8
        period = 1.0
        potential = PoweredCosineSum(dim=dim, period=period)
        opt = ExtendedMixedOptimizer(
            potential,
            start_coords,
            tol=1e-10,
            T=10,
            conv_tol=1e-8,
            rtol=1e-10,
            atol=1e-10,
        )
        res = opt.run(1000)
        assert np.allclose(res.coords, 0.0, atol=1e-3)
