#include <gtest/gtest.h>

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

#include <cmath>
#include <iostream>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <vector>

#include "pele/array.hpp"

using pele::Array;
using std::cout;

TEST(CVODE, IterativeDenseCheck) {
  static const size_t _ndim = 2;
  size_t n_particles;
  size_t n_dof;
  double power;
  double eps;
  double phi;
  double box_length;
  pele::Array<double> x_sparse;
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
  x_sparse = {1.8777,  3.61102,  1.70726, 6.93457, 2.14539, 2.55779,  4.1191,
              7.02707, 0.357781, 4.92849, 1.28547, 2.83375, 0.775204, 6.78136,
              6.27529, 1.81749,  6.02049, 6.70693, 5.36309, 4.6089,   4.9476,
              5.54674, 0.677836, 6.04457, 4.9083,  1.24044, 5.09315,  0.108931,
              2.18619, 6.52932,  2.85539, 2.30303};
  Array<double> x_dense = x_sparse.copy();
  boxvec = {box_length, box_length};

  //////////// parameters for inverse power potential at packing fraction 0.7 in
  /// 3d
  //////////// as generated from code params.py in basinerror library

  double k;
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif
  bool exact_sum = false;
  double ncellsx_scale = pele::get_ncellx_scale(radii, boxvec, 1);
  // std::shared_ptr<pele::InversePowerPeriodicCellLists<_ndim> > potcell =
  // std::make_shared<pele::InversePowerPeriodicCellLists<_ndim>>(power, eps,o
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

  pele::CVODEBDFOptimizer cvode_iterative(pot, x_sparse, 1e-9, 1e-10, 1e-10,
                                          pele::ITERATIVE);
  pele::CVODEBDFOptimizer cvode_dense(pot, x_dense, 1e-9, 1e-10, 1e-10,
                                      pele::DENSE);

  Array<double> x_iterative_end;
  Array<double> x_dense_end;

  size_t nstep = 20000;

  cvode_iterative.run(nstep);
  cvode_dense.run(nstep);

  x_iterative_end = cvode_iterative.get_x();
  x_dense_end = cvode_dense.get_x();
  for (auto i = 0; i < x_sparse.size(); ++i) {
    ASSERT_NEAR(x_iterative_end[i], x_dense_end[i], 1e-6);
  }
}
