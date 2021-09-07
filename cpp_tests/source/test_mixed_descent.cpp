// base imports
#include "pele/array.h"
#include "pele/cell_lists.h"
#include "pele/utils.hpp"
// optimizer imports
#include "pele/mxopt.h"

// potential imports
#include "pele/inversepower.h"
#include "pele/rosenbrock.h"

#include "pele/gradient_descent.h"
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

using pele::Array;
using std::cout;

// Test for whether the switch works for mixed descent
TEST(MXD, TEST_256_RUN) {
  static const size_t _ndim = 2;
  size_t n_particles;
  size_t n_dof;

  double power;
  double eps;

  double phi;

  pele::Array<double> x;
  pele::Array<double> radii;
  pele::Array<double> xfinal;
  pele::Array<double> boxvec;

  eps = 1.0;
  power = 2.5;

  n_particles = 256;
  n_dof = n_particles * _ndim;
  phi = 0.9;

  int n_1 = n_particles / 2;
  int n_2 = n_particles - n_1;

  double r_1 = 1.0;
  double r_2 = 1.0;
  double r_std1 = 0.05;
  double r_std2 = 0.05;

  radii = pele::generate_radii(n_1, n_2, r_1, r_2, r_std1, r_std2);

  double box_length = pele::get_box_length(radii, _ndim, phi);

  boxvec = {box_length, box_length};

  x = pele::generate_random_coordinates(box_length, n_particles, _ndim);

  //////////// parameters for inverse power potential at packing fraction 0.7 in
  ///3d
  //////////// as generated from code params.py in basinerror library

  double k;
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif

  double ncellsx_scale = get_ncellx_scale(radii, boxvec, 1);
  std::shared_ptr<pele::InversePowerPeriodicCellLists<_ndim>> potcell =
      std::make_shared<pele::InversePowerPeriodicCellLists<_ndim>>(
          power, eps, radii, boxvec, ncellsx_scale);
  // std::shared_ptr<pele::InversePowerPeriodic<_ndim> > pot =
  // std::make_shared<pele::InversePowerPeriodic<_ndim> >(power, eps, radii,
  // boxvec);

  pele::MixedOptimizer optimizer(potcell, x, 1e-4, 10);
  optimizer.run(500);
  std::cout << optimizer.get_nfev() << "nfev \n";
  std::cout << optimizer.get_nhev() << "nhev \n";
  std::cout << optimizer.get_rms() << "rms \n";
  std::cout << optimizer.get_niter() << "\n";
}

TEST(MXD, Rosebrock_works) {
  auto rosenbrock = std::make_shared<pele::RosenBrock>();
  Array<double> x0(2, 0);
  pele::MixedOptimizer optimizer(rosenbrock, x0);
  // pele::LBFGS lbfgs(rosenbrock, x0, 1e-4, 1, 1);
  // pele ::GradientDescent lbfgs(rosenbrock, x0);
  optimizer.run(100);
  Array<double> x = optimizer.get_x();
  std::cout << x << "\n";
  cout << optimizer.get_nfev() << " get_nfev() \n";
  cout << optimizer.get_niter() << " get_niter() \n";
  cout << optimizer.get_rms() << " get_rms() \n";
  cout << optimizer.get_rms() << " get_rms() \n";
  std::cout << x0 << "\n"
            << " \n";
  std::cout << x << "\n";
  std::cout << "this is okay"
            << "\n";
}