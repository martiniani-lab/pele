#include "pele/cell_lists.hpp"
#include "pele/cvode.hpp"
#include "pele/gradient_descent.hpp"
#include "pele/harmonic.hpp"
#include "pele/inversepower.hpp"
#include "pele/lbfgs.hpp"
#include "pele/mxd_end_only.hpp"
#include "pele/mxopt.hpp"
#include "pele/rosenbrock.hpp"
#include "pele/utils.hpp"

//#include "pele/mxd_end_only.hpp"

#include "pele/array.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

using pele::Array;
using std::cout;

// Test for whether the switch works for mixed descent
// TODO: seed the random number generator
TEST(CVODE, TEST_256_RUN) {
  static const size_t _ndim = 2;
  size_t n_particles;
  size_t n_dof;
  double power;
  double eps;
  double phi;
  double box_length;
  pele::Array<double> x;
  pele::Array<double> radii;
  pele::Array<double> xfinal;
  pele::Array<double> boxvec;

  eps = 1.0;
  power = 2.5;
  const int pow2 = 5;

  n_particles = 16;
  n_dof = n_particles * _ndim;
  phi = 0.9;

  int n_1 = n_particles / 2;
  int n_2 = n_particles - n_1;

  radii = {0.982267, 0.959526, 1.00257,  0.967356, 1.04893, 0.97781,
           0.954191, 0.988939, 0.980737, 0.964811, 1.04198, 0.926199,
           0.969865, 1.08593,  1.01491,  0.968892};
  box_length = 7.40204;
  x = {1.8777,  3.61102,  1.70726, 6.93457, 2.14539, 2.55779,  4.1191,
       7.02707, 0.357781, 4.92849, 1.28547, 2.83375, 0.775204, 6.78136,
       6.27529, 1.81749,  6.02049, 6.70693, 5.36309, 4.6089,   4.9476,
       5.54674, 0.677836, 6.04457, 4.9083,  1.24044, 5.09315,  0.108931,
       2.18619, 6.52932,  2.85539, 2.30303};
  boxvec = {box_length, box_length};

  //////////// parameters for inverse power potential at packing fraction 0.7 in
  /// 3d
  //////////// as generated from code params.py in basinerror library

  double k;
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif
  bool exact_sum = false;
  double ncellsx_scale = get_ncellx_scale(radii, boxvec, 1);
  // std::shared_ptr<pele::InversePowerPeriodicCellLists<_ndim> > potcell =
  // std::make_shared<pele::InversePowerPeriodicCellLists<_ndim>>(power, eps,
  // radii, boxvec, ncellsx_scale);
  // std::shared_ptr<pele::InversePowerPeriodic<_ndim> > pot =
  // std::make_shared<pele::InversePowerPeriodic<_ndim> >(power, eps, radii,
  // boxvec); std::shared_ptr<pele::InverseHalfIntPowerPeriodicCellLists<_ndim,
  // pow2>> potcell =
  // std::make_shared<pele::InverseHalfIntPowerPeriodicCellLists<_ndim,
  // pow2>>(eps, radii, boxvec, ncellsx_scale, exact_sum);
  std::shared_ptr<pele::InverseHalfIntPowerPeriodic<_ndim, pow2>> pot =
      std::make_shared<pele::InverseHalfIntPowerPeriodic<_ndim, pow2>>(
          eps, radii, boxvec, exact_sum);

  pele::CVODEBDFOptimizer optimizer(pot, x, 1e-9, 1e-5, 1e-5);
  optimizer.run();
  std::cout << optimizer.get_nfev() << "nfev \n";
  std::cout << optimizer.get_nhev() << "nhev \n";
  std::cout << optimizer.get_rms() << "rms \n";
  std::cout << optimizer.get_niter() << "the \n";
  std::cout << optimizer.get_x() << "x \n";
  ASSERT_LE(optimizer.get_niter(), 400);
}

// Test for whether CVODE works with RosenBrock. an easy check for a dense
// hessian system with ill conditioned minima
TEST(CVODE, RosenBrock) {
  auto rosenbrock = std::make_shared<pele::RosenBrock>();
  Array<double> x0(2, 0);
  pele::CVODEBDFOptimizer cvode(rosenbrock, x0);
  // pele::LBFGS lbfgs(rosenbrock, x0, 1e-4, 1, 1);
  // pele ::GradientDescent lbfgs(rosenbrock, x0);
  cvode.run(2000);
  Array<double> x = cvode.get_x();
  std::cout << x << "\n";
  cout << cvode.get_nfev() << " get_nfev() \n";
  cout << cvode.get_niter() << " get_niter() \n";
  cout << cvode.get_rms() << " get_rms() \n";
  cout << cvode.get_rms() << " get_rms() \n";
  std::cout << x0 << "\n"
            << " \n";
  std::cout << x << "\n";
}

// Test for whether CVODE works with a matrix with zeros.
// This checks for memory issues when using sparse CVODE
TEST(CVODE, Harmonic) {

  double k = 1.0;
  size_t dim = 2;
  Array<double> origin(2);
  origin[0] = 0;
  origin[1] = 0;

  // define potential
  auto harmonic = std::make_shared<pele::Harmonic>(origin, k, dim);
  // define the initial position
  Array<double> x0(dim);
  x0[0] = 1;
  x0[1] = 1;
  pele::CVODEBDFOptimizer cvode(harmonic, x0);
  // pele::LBFGS lbfgs(rosenbrock, x0, 1e-4, 1, 1);
  // pele ::GradientDescent lbfgs(rosenbrock, x0);
  cvode.run(2000);
  Array<double> x = cvode.get_x();
  std::cout << x0 << "\n";
  cout << cvode.get_nfev() << " get_nfev() \n";
  cout << cvode.get_niter() << " get_niter() \n";
  cout << cvode.get_rms() << " get_rms() \n";
  cout << cvode.get_rms() << " get_rms() \n";
  std::cout << x0 << "\n"
            << " \n";
  std::cout << x0 << "\n";
}