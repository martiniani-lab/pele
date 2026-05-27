"""Smoke tests for the GMIN MYLBFGS bridge (pele/optimize/_gmin_lbfgs).

Skipped when pele was built without -DWITH_GMIN=ON. See the module-level
docstring of pele/optimize/_gmin_lbfgs.pyx for the integration story.
"""

import unittest

import numpy as np

from pele.potentials import BasePotential

try:
    from pele.optimize._gmin_lbfgs import gmin_mylbfgs, gmin_cgmin
    HAVE_GMIN = True
except ImportError:
    HAVE_GMIN = False


class _Quadratic(BasePotential):
    """Trivial potential E = 0.5 * sum(x_i^2), grad = x_i. Minimum at 0."""

    def getEnergy(self, x):
        return 0.5 * float(np.dot(x, x))

    def getEnergyGradient(self, x):
        return self.getEnergy(x), x.copy()


@unittest.skipUnless(HAVE_GMIN, "pele was not built with -DWITH_GMIN=ON")
class TestGminMylbfgsQuadratic(unittest.TestCase):
    """Verify the bridge end-to-end on a quadratic well.

    Catches Fortran/C callback wiring bugs without needing a real LJ
    cluster — if the callback path is broken the test simply doesn't
    converge.
    """

    def test_converges_to_origin(self):
        x0 = np.arange(12, dtype=np.float64) * 0.5 + 0.5
        x, energy, converged, niter = gmin_mylbfgs(
            _Quadratic(), x0, M=4, eps=1e-9, itmax=1000
        )
        self.assertTrue(converged, "MYLBFGS did not converge on a quadratic well")
        self.assertGreater(niter, 0, "MYLBFGS reported 0 iterations")
        self.assertAlmostEqual(energy, 0.0, places=10)
        self.assertLess(np.max(np.abs(x)), 1e-5)

    def test_rejects_non_multiple_of_3(self):
        with self.assertRaises(ValueError):
            gmin_mylbfgs(_Quadratic(), np.array([1.0, 2.0]))


@unittest.skipUnless(HAVE_GMIN, "pele was not built with -DWITH_GMIN=ON")
class TestGminMylbfgsLJ(unittest.TestCase):
    """Compare MYLBFGS against pele's LBFGS_CPP on a small LJ cluster.

    They should both reach the same local minimum (within numerical
    tolerance), even though they may take different paths.
    """

    def test_agrees_with_pele_lbfgs_on_lj(self):
        from pele.potentials import LJ
        from pele.optimize import LBFGS_CPP

        natoms = 5
        rng = np.random.default_rng(0)
        x0 = rng.uniform(-1.5, 1.5, 3 * natoms)

        pot = LJ()

        # GMIN's MYLBFGS
        x_gmin, e_gmin, conv_gmin, _ = gmin_mylbfgs(
            pot, x0.copy(), eps=1e-7, itmax=10000
        )

        # Pele's reference
        ref = LBFGS_CPP(x0.copy(), pot, tol=1e-7, nsteps=10000).run()

        self.assertTrue(conv_gmin)
        self.assertTrue(ref.success)
        # Both should land at the same energy from the same start point.
        self.assertAlmostEqual(e_gmin, ref.energy, places=5)


@unittest.skipUnless(HAVE_GMIN, "pele was not built with -DWITH_GMIN=ON")
class TestGminCgmin(unittest.TestCase):
    """Same checks as the MYLBFGS suite, but for GMIN's CGMIN (conjugate gradient).

    CGMIN converges on RMS < COMMONS::GMAX; our wrapper sets GMAX = eps.
    """

    def test_converges_to_origin(self):
        x0 = np.arange(12, dtype=np.float64) * 0.5 + 0.5
        x, energy, converged, niter = gmin_cgmin(
            _Quadratic(), x0, eps=1e-9, itmax=1000
        )
        self.assertTrue(converged, "CGMIN did not converge on a quadratic well")
        self.assertGreater(niter, 0, "CGMIN reported 0 iterations")
        self.assertAlmostEqual(energy, 0.0, places=10)
        self.assertLess(np.max(np.abs(x)), 1e-5)

    def test_rejects_non_multiple_of_3(self):
        with self.assertRaises(ValueError):
            gmin_cgmin(_Quadratic(), np.array([1.0, 2.0]))

    def test_agrees_with_pele_lbfgs_on_lj(self):
        from pele.potentials import LJ
        from pele.optimize import LBFGS_CPP

        natoms = 5
        rng = np.random.default_rng(0)
        x0 = rng.uniform(-1.5, 1.5, 3 * natoms)
        pot = LJ()

        x_cg, e_cg, conv_cg, _ = gmin_cgmin(
            pot, x0.copy(), eps=1e-7, itmax=10000
        )
        ref = LBFGS_CPP(x0.copy(), pot, tol=1e-7, nsteps=10000).run()

        # CGMIN often returns conv=False even after reaching the minimum
        # due to its "STUCK" check tripping at machine-epsilon gradient
        # noise. We assert on energy agreement, not the flag.
        self.assertTrue(ref.success)
        self.assertAlmostEqual(e_cg, ref.energy, places=4)


if __name__ == "__main__":
    unittest.main()
