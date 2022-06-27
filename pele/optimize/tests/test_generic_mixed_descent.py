""" Test for generating mixed descent run in C++
"""
from matplotlib import use
import numpy as np
from pele.optimize import GenericMixedDescent, ModifiedFireCPP, CVODEBDFOptimizer
from pele.potentials import InversePower


def test_mixed_descent():
    # use radii, box_length and coordinates from the C++ test test_mixed_descent.cpp

    radii = [
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
    box_length = 7.40204
    x = [
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
    box_vec = [box_length, box_length]

    radii = np.array(radii)
    box_vec = np.array(box_vec)
    x = np.array(x)
    print(radii)

    ### potential parameters ###
    ndim = 2
    use_cell_lists = False
    power = 2.5
    eps = 1.0  # epsilon for the main potential

    potential = InversePower(
        power, eps, radii, use_cell_lists=use_cell_lists, ndim=ndim, boxvec=box_vec,
    )

    optimizer_convex = ModifiedFireCPP(x, potential, fdec=0.5, tol=1e-9)

    optimizer_non_convex = CVODEBDFOptimizer(potential, x, rtol=1e-5, atol=1e-5, tol=1e-9)

    # optimizer_non_convex = ModifiedFireCPP(x, potential, tol=1e-9)

    opt_mixed_descent = GenericMixedDescent(
        potential,
        x,
        optimizer_convex,
        optimizer_non_convex,
        tol=1e-9,
        translation_offset=1,
        steps_before_convex_check=1,
    )
    # res = opt_mixed_descent.run(2000)
    
    res = optimizer_non_convex.run(10000)
    print(res)


test_mixed_descent()
