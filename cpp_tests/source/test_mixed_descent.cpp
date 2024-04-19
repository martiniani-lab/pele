#include <gtest/gtest.h>

#include "pele/array.hpp"
#include "pele/cell_lists.hpp"
#include "pele/extended_mixed_descent.hpp"
#include "pele/utils.hpp"
// optimizer imports

// potential imports
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include "pele/gradient_descent.hpp"
#include "pele/inversepower.hpp"
#include "pele/rosenbrock.hpp"

using pele::Array;

TEST(EMXD, Reset) {
  static const size_t _ndim = 2;
  size_t n_particles;
  size_t n_dof;
  double power;
  double eps;
  double phi;
  double box_length;
  pele::Array<double> x_start;
  pele::Array<double> radii;
  pele::Array<double> xfinal;
  pele::Array<double> boxvec;

  eps = 1.0;
  power = 2.5;
  const int pow2 = 5;

  n_particles = 32;
  n_dof = n_particles * _ndim;
  phi = 0.9;

  int n_1 = n_particles / 2;
  int n_2 = n_particles - n_1;

  radii =
      pele::generate_bidisperse_radii(n_1, n_2, 1.0, 1.4, 0.05, 0.07).copy();
  box_length = pele::get_box_length(radii, _ndim, 0.9);

  x_start =
      pele::generate_random_coordinates(box_length, n_particles, _ndim, 0);
  Array<double> x_reset_copy =
      pele::generate_random_coordinates(box_length, n_particles, _ndim, 42);

  // generate random coordinates in a box of size box_length
  // use std::uniform_real_distribution to generate random numbers
  // set rng
  std::mt19937 rng(0);
  for (auto i = 0; i < n_particles; i++) {
    x_reset_copy[i] = std::uniform_real_distribution<>(0, box_length)(rng);
  }

  std::mt19937 rng2(5);
  for (auto i = 0; i < n_particles; i++) {
    x_start[i] = std::uniform_real_distribution<>(0, box_length)(rng2);
  }
  Array<double> x_dense = x_start.copy();
  boxvec = {box_length, box_length};

  double k;
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif
  bool exact_sum = false;
  double ncellsx_scale = pele::get_ncellx_scale(radii, boxvec, 1);
  std::shared_ptr<pele::InverseHalfIntPowerPeriodic<_ndim, pow2>> pot =
      std::make_shared<pele::InverseHalfIntPowerPeriodic<_ndim, pow2>>(
          eps, radii, boxvec, exact_sum);

  pele::ExtendedMixedOptimizer mxd(pot, x_start, nullptr, 1e-10, 10, 1, 1e-4,
                                   1e-4, 1e-4, false, Array<double>(0), false,
                                   1, pele::StopCriterionType::GRADIENT);

  pele::ExtendedMixedOptimizer mxd_reference(
      pot, x_reset_copy, nullptr, 1e-10, 10, 1, 1e-4, 1e-4, 1e-4, false,
      Array<double>(0), false, 1, pele::StopCriterionType::GRADIENT);

  Array<double> reference_start_x = x_reset_copy.copy();
  // reference run
  mxd_reference.run(4000);
  Array<double> test_mxd_reference_x = mxd_reference.get_x().copy();
  int nfev_reference = mxd_reference.get_nfev();
  int nhev_reference = mxd_reference.get_nhev();

  // run with original
  Array<double> original_x = x_start.copy();
  mxd.run(4000);
  Array<double> x_before_reset = mxd.get_x().copy();
  int nfev = mxd.get_nfev();
  int nhev = mxd.get_nhev();

  mxd.reset(reference_start_x);
  ASSERT_FALSE(mxd.success());
  ASSERT_EQ(mxd.get_nfev(), 0);
  ASSERT_EQ(mxd.get_nhev(), 0);
  ASSERT_FALSE(mxd.stop_criterion_satisfied());
  mxd.run(4000);
  Array<double> x_after_reset = mxd.get_x();
  int nfev_after_reset = mxd.get_nfev();
  int nhev_after_reset = mxd.get_nhev();

  ASSERT_EQ(nfev_reference, nfev_after_reset);
  ASSERT_EQ(nhev_reference, nhev_after_reset);

  for (auto i = 0; i < x_start.size(); ++i) {
    ASSERT_EQ(test_mxd_reference_x[i], x_after_reset[i]);
  }
}