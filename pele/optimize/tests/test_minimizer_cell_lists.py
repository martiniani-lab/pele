"""
Test whether minimizer gives exactly the same answer with/without cell lists

"""
import numpy as np
from pele.optimize import CVODEBDFOptimizer
from pele.potentials import InversePower, HS_WCA, Frenkel
import unittest
from pele.utils.cell_scale import get_ncellsx_scale
from pele.optimize._quench import lbfgs_cpp, modifiedfire_cpp
from pele.distance import Distance

np.random.seed(0)


class TestSameMinimaInversePower(unittest.TestCase):
    """Tests whether energy minimization results in the same minima, (done with OMP_Threads=1)"""

    # def setUp(self):
    #     nparticles = 2
    #     print("hello")

    def test_same_minima_InversePower(self):
        nparticles = 32
        hs_radii = 0.05 * np.random.randn(nparticles) + 1
        volpart = np.sum(4.0 / 3.0 * np.pi * hs_radii**3)
        phi = 0.7
        boxl = (volpart / phi) ** (1 / 3.0)
        boxv = [boxl, boxl, boxl]
        coords = np.random.rand(nparticles * 3) * boxl
        ncellx_scale = get_ncellsx_scale(np.ones(nparticles), boxv)
        bdim = 3
        pot_cellists = InversePower(
            2.5,
            1,
            hs_radii,
            ndim=3,
            boxvec=boxv,
            use_cell_lists=True,
            ncellx_scale=ncellx_scale,
        )
        pot_no_cellists = InversePower(
            2.5,
            1,
            hs_radii,
            ndim=3,
            boxvec=boxv,
            use_cell_lists=False,
            ncellx_scale=ncellx_scale,
        )

        nsteps = 1000
        tol = 1e-10

        res_cell_lists = modifiedfire_cpp(
            coords, pot_cellists, nsteps=nsteps, tol=tol
        )
        res_no_cell_lists = modifiedfire_cpp(
            coords, pot_no_cellists, nsteps=nsteps, tol=tol
        )

        fcoords_cell_lists = res_cell_lists.coords
        fcoords_no_cell_lists = res_no_cell_lists.coords
        # self.assertEqual(fcoords_no_cell_lists,fcoords_cell_lists)
        print(np.max(fcoords_no_cell_lists - fcoords_cell_lists))
        self.assertTrue(
            np.all(fcoords_no_cell_lists - fcoords_cell_lists < 1e-5)
        )

    def test_same_minima_HS_WCA(self):
        nparticles = 32
        radius_sca = 0.9085602964160698
        pot_sca = 0.1
        eps = 1.0
        hs_radii = 0.05 * np.random.randn(nparticles) + 1
        volpart = np.sum(4.0 / 3.0 * np.pi * hs_radii**3)
        phi = 0.7
        boxl = (volpart / phi) ** (1 / 3.0)
        boxv = [boxl, boxl, boxl]
        coords = np.random.rand(nparticles * 3) * boxl
        ncellx_scale = get_ncellsx_scale(np.ones(nparticles), boxv)
        bdim = 3
        distance_method = Distance.PERIODIC
        pot_cellists = HS_WCA(
            use_cell_lists=True,
            eps=eps,
            sca=pot_sca,
            radii=hs_radii * radius_sca,
            boxvec=boxv,
            ndim=bdim,
            ncellx_scale=ncellx_scale,
            distance_method=distance_method,
        )
        pot_no_cellists = HS_WCA(
            use_cell_lists=False,
            eps=eps,
            sca=pot_sca,
            radii=hs_radii * radius_sca,
            boxvec=boxv,
            ndim=bdim,
            ncellx_scale=ncellx_scale,
            distance_method=distance_method,
        )
        nsteps = 1000
        tol = 1e-10
        res_cell_lists = modifiedfire_cpp(
            coords, pot_cellists, nsteps=nsteps, tol=tol
        )
        res_no_cell_lists = modifiedfire_cpp(
            coords, pot_no_cellists, nsteps=nsteps, tol=tol
        )

        fcoords_cell_lists = res_cell_lists.coords
        fcoords_no_cell_lists = res_no_cell_lists.coords
        # self.assertEqual(fcoords_no_cell_lists,fcoords_cell_lists)
        print(np.max(fcoords_no_cell_lists - fcoords_cell_lists))
        self.assertTrue(
            np.all(fcoords_no_cell_lists - fcoords_cell_lists < 1e-5)
        )

    def test_same_minima_Frenkel(self):
        nparticles = 32
        hs_radii = 0.05 * np.random.randn(nparticles) + 1
        volpart = np.sum(4.0 / 3.0 * np.pi * hs_radii**3)
        phi = 0.7
        boxl = (volpart / phi) ** (1 / 3.0)
        boxv = [boxl, boxl, boxl]
        coords = np.random.rand(nparticles * 3) * boxl
        ncellx_scale = get_ncellsx_scale(np.ones(nparticles), boxv)
        pot_cellists = Frenkel(
            nparticles, boxvec=boxv, celllists=True, ncellx_scale=ncellx_scale
        )
        pot_no_cellists = Frenkel(
            nparticles, boxvec=boxv, celllists=False, ncellx_scale=ncellx_scale
        )
        nsteps = 1000
        tol = 1e-10
        res_cell_lists = modifiedfire_cpp(
            coords, pot_cellists, nsteps=nsteps, tol=tol
        )
        res_no_cell_lists = modifiedfire_cpp(
            coords, pot_no_cellists, nsteps=nsteps, tol=tol
        )

        fcoords_cell_lists = res_cell_lists.coords
        fcoords_no_cell_lists = res_no_cell_lists.coords
        # self.assertEqual(fcoords_no_cell_lists,fcoords_cell_lists)
        print(np.max(fcoords_no_cell_lists - fcoords_cell_lists))
        self.assertTrue(
            np.all(fcoords_no_cell_lists - fcoords_cell_lists < 1e-5)
        )


if __name__ == "__main__":
    unittest.main()
