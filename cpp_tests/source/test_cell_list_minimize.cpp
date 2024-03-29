/**
 * This test should check whether minimization should give the same result with
 * or without cell lists (particularly intended for what seems like an edge case
 * with 2~3 cells which seems unstable for particular number of iterations for
 * lbfgs)]
 *
 */
#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "pele/hs_wca.hpp"
#include "pele/lbfgs.hpp"
#include "pele/modified_fire.hpp"

// Class for making sure the minimizer gives the same answer with and without
// cell lists
//
class CheckInitializerTest : public ::testing::Test {
 public:
  static const size_t _ndim = 3;
  size_t nr_particles;
  size_t nr_dof;
  double eps;
  double sca;
  double rsca;
  double energy;
  pele::Array<double> x;
  pele::Array<double> x_overlap;
  pele::Array<double> hs_radii;
  pele::Array<double> radii;
  pele::Array<double> xfinal;
  pele::Array<double> boxvec;

  // pele::Array<double> boxvec;
  double k;
  // pele::Array<double> hs_radii;
  // std::shared_ptr<pele::BasePotential> potcell;
  virtual void SetUp() {
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    nr_particles = 12;

    nr_dof = nr_particles * _ndim;
    sca = 0.2;
    rsca = 0.9085602964160698;
    eps = 1.0;
    x = pele::Array<double>(nr_dof, 0);
    x_overlap = pele::Array<double>(nr_dof, 0);
    // hs_radii = pele::Array<double>(nr_particles, 1);
    boxvec = {15.69339383, 15.69339383, 15.69339383};
    // for (size_t i = 0; i < nr_particles; ++i) {
    //   x[i * _ndim] = 2.1 * i - 0.5*(2.1*nr_particles);
    //   x_overlap[i * _ndim] = 1.9 * i - 0.5*(2.1*nr_particles);
    // }
    x = {
        2.5378779,  3.8130062,  0.46970527, 5.10753552, 1.69665129, 1.49340466,
        0.02920454, 3.09266585, 2.70775742, 3.62312597, 5.56930138, 5.17337494,
        5.18185991, 2.9904874,  0.59222085, 1.03001804, 5.42603331, 2.34541057,
        4.92488314, 3.82696998, 3.57992517, 1.56886464, 5.10553537, 1.17791406,
        2.30243832, 5.6567995,  4.18878403, 2.5339099,  3.19207623, 2.34143946,
        4.13902895, 2.27278509, 3.81540035, 4.01222349, 3.47046441, 3.07462889,
        1.17330514, 1.13390472, 4.53045474, 1.65297885, 3.73465472, 1.7058363,
        0.82257231, 2.2998844,  1.76648661, 1.38576076, 3.34830793, 1.39683576};

    hs_radii = {1.05147999, 0.81830371, 0.92429938, 0.89301688,
                1.17161467, 0.917398,   1.09963772, 1.40203985,
                1.25257231, 0.91215703, 0.93071242, 1.09106393,
                0.66626746, 0.8275829,  1.09858217, 0.97513732};
    radii = hs_radii * rsca;

    size_t ndim = _ndim;
    // pot = std::make_shared<pele::Harmonic>(x, k, ndim);
    std::cout << eps << "\n";
    std::cout << sca << "\n";
    std::cout << x << "\n";

    // potcell = std::make_shared<pele::HS_WCAPeriodic<_ndim, 6>>(eps, sca,
    // radii, boxvec);
    // energy = potcell->get_energy(x);
    // std::cout << energy << "\n";
  }
};

TEST_F(CheckInitializerTest, FullRunFire) {
  double dtstart =
      0.1;  // Parameters match the defaults for the python interface for fire
  double dtmax = 1;
  double maxstep = 0.5;
  size_t Nmin = 5;
  double finc = 1.1;
  double fdec = 0.5;
  double fa = 0.99;
  double astart = 0.1;
  double tol = 1e-3;
  bool stepback = 1;
  int nsteps = 1e5;
  int verbosity = 0;
  int iprint = -1;

  std::shared_ptr<pele::HS_WCAPeriodicCellLists<_ndim>> potcell =
      std::make_shared<pele::HS_WCAPeriodicCellLists<_ndim>>(eps, sca, hs_radii,
                                                             boxvec, 1.0);
  std::shared_ptr<pele::HS_WCAPeriodic<_ndim>> pot =
      std::make_shared<pele::HS_WCAPeriodic<_ndim>>(eps, sca, hs_radii, boxvec);

  std::vector<double> xinit;
  std::vector<double> xinit2;

  pele::MODIFIED_FIRE optimizer_modified_fire_cell_list(
      potcell, x, dtstart, dtmax, maxstep, Nmin, finc, fdec, fa, astart, tol,
      stepback);

  optimizer_modified_fire_cell_list.run(nsteps);

  xfinal = optimizer_modified_fire_cell_list.get_x();

  pele::MODIFIED_FIRE optimizer_modified_fire(pot, x, dtstart, dtmax, maxstep,
                                              Nmin, finc, fdec, fa, astart, tol,
                                              stepback);

  for (int i = 0; i < x.size(); ++i) {
    xinit.push_back(xfinal[i] - x[i]);
  }

  for (auto i = xinit.begin(); i != xinit.end(); ++i) {
    std::cout << *i << ", ";
  }

  std::cout << "\n"
            << "\n";

  // optimizer_modified_fire.one_iteration();
  optimizer_modified_fire.run(nsteps);

  xfinal = optimizer_modified_fire.get_x();

  for (int i = 0; i < x.size(); ++i) {
    xinit2.push_back(xfinal[i] - x[i]);
  }

  for (auto i = xinit2.begin(); i != xinit2.end(); ++i) {
    std::cout << *i << ", ";
  }
  std::cout << " optimizer output \n";
  // We do it for one value assuming it holds for all values
  EXPECT_NEAR(xinit[0], xinit2[0], 1e-10);
};
