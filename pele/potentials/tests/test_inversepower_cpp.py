from __future__ import absolute_import, division, print_function

import logging
import os
import unittest
from builtins import range

import _base_test
import numpy as np

from pele.optimize._quench import lbfgs_cpp
from pele.potentials import _inversepower_cpp


def read_xyzdr(fname, bdim=3):
    coords = []
    radii = []
    rattlers = []
    f = open(fname, "r")
    while True:
        xyzdr = f.readline()
        if not xyzdr:
            break
        x, y, z, d, r = xyzdr.split()
        coords.extend([float(x), float(y), float(z)])
        radii.extend([float(d) / 2])
        for _ in range(bdim):
            rattlers.extend([float(r)])
    return np.array(coords), np.array(radii), np.array(rattlers)


def minimize(coords, pot):
    result = lbfgs_cpp(coords, pot)
    # result = modifiedfire_cpp(coords, pot)
    return result.coords, result.energy, result.grad, result.rms


class TestInversePower_CPP(_base_test._BaseTest):
    def setUp(self):
        current_dir = os.path.dirname(__file__)
        xyz, hs_radii, rattlers = read_xyzdr(current_dir + "/_hswca20_min2.xyzdr")
        sca = 0.205071132088
        radii = hs_radii * (1.0 + sca)
        boxv = np.array([6.26533756282, 6.26533756282, 6.26533756282])
        pow = 4
        eps = 1
        self.pot = _inversepower_cpp.InversePower(pow, eps, radii, boxvec=boxv)
        self.natoms = 20
        result = minimize(xyz, self.pot)
        self.xmin = result[0]  # xyz
        self.Emin = result[1]  # self.pot.getEnergy(self.xmin)
        print(self.Emin)
        self.xrandom = np.random.uniform(-1, 1, len(xyz)) * 1e-2


if __name__ == "__main__":
    logging.basicConfig(filename="hs_wca_cpp.log", level=logging.DEBUG)
    unittest.main()
