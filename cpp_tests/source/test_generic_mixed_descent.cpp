
// base imports
#include "pele/array.hpp"
#include "pele/cell_lists.hpp"
#include "pele/utils.hpp"
// optimizer imports
// #include "pele/extended_mixed_descent.hpp"
#include "pele/generic_mixed_descent.hpp"

#include "pele/cvode.hpp"
#include "pele/lbfgs.hpp"
#include "pele/modified_fire.hpp"

// potential imports
#include "pele/inversepower.hpp"
#include "pele/rosenbrock.hpp"

#include "pele/gradient_descent.hpp"
#include <cmath>
#include <complex>
#include <cstddef>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <pele/cvode.hpp>
#include <pele/lbfgs.hpp>
#include <pele/modified_fire.hpp>
#include <pele/mxopt.hpp>
#include <stdexcept>
#include <vector>

using pele::Array;
using std::cout;

// Test for whether the switch works for mixed descent
TEST(GENERIC_MXD, TEST_256_RUN) {
  static const size_t _ndim = 2;
  size_t n_particles;
  size_t n_dof;

  double power;
  double eps;
  cout.precision(17);
  double phi;

  pele::Array<double> x;
  pele::Array<double> radii;
  pele::Array<double> xfinal;
  pele::Array<double> boxvec;

  eps = 1.0;
  power = 2.5;

  n_particles = 16;
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

// required to plot trajectory
#if PRINT_TO_FILE == 1
  std::ofstream radii_file;
  std::ofstream box_length_file;

  radii_file.open("radii_n.txt");
  box_length_file.open("box_length_n.txt");
  radii_file << radii;
  box_length_file << box_length << std::endl;
  radii_file.close();
  box_length_file.close();
#endif
  x = pele::generate_random_coordinates(box_length, n_particles, _ndim);

  // x = {2.28061,
  // 0.96684, 3.16573, 4.31568, 5.68627, 3.70002, 4.17922, 4.82482, 4.9194, 4.06309,
  // 2.77059, 4.03329, 5.71843, 4.22168, 1.04098, 5.16762, 2.98504, 3.97102, 3.39741,
  // 7.20914, 1.29798, 6.73186,
  // 0.396831, 4.22964, 1.3706, 3.12105, 5.03942, 2.20184, 1.32305, 2.31878, 5.5583,
  // 1.16185};

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

  double ncellsx_scale = get_ncellx_scale(radii, boxvec, 1);
  // std::shared_ptr<pele::InversePowerPeriodicCellLists<_ndim>> potcell =
  //     std::make_shared<pele::InversePowerPeriodicCellLists<_ndim>>(
  //         power, eps, radii, boxvec, ncellsx_scale);

  std::shared_ptr<pele::InversePowerPeriodic<_ndim>> pot =
      std::make_shared<pele::InversePowerPeriodic<_ndim>>(power, eps, radii,
                                                          boxvec);

  std::shared_ptr<pele::InversePowerPeriodic<_ndim>> extension_potential =
      std::make_shared<pele::InversePowerPeriodic<_ndim>>(power, eps * 1e-6,
                                                          radii * 2, boxvec);

  // std::shared_ptr<pele::InversePowerPeriodic<_ndim>> extension_potential

  Array<double> x_new = x.copy();
  Array<double> x_new_extension = x.copy();
  Array<double> x_cvode = x.copy();

  std::cout << "x" << std::endl;
  std::cout << x << std::endl;
  std::cout << "radii" << std::endl;
  std::cout << radii << std::endl;

  double tol = 1e-9;
  // combine both optimizers

  std::shared_ptr<pele::MODIFIED_FIRE> shared_pointer_to_fire =
      std::make_shared<pele::MODIFIED_FIRE>(pot, x_new, 0.1, 1, 0.5, 5, 1.1,
                                            0.5, 0.99, 0.1, 1e-9);
  std::shared_ptr<pele::CVODEBDFOptimizer> shared_pointer_to_cvode =
      std::make_shared<pele::CVODEBDFOptimizer>(pot, x, 1e-9);
  size_t steps_before_convex_check = 50;
  pele::GenericMixedDescent optimizer_mixed_descent_fire(
      pot, x_new, 1e-9, shared_pointer_to_cvode, shared_pointer_to_fire, 1.0,
      steps_before_convex_check);

  optimizer_mixed_descent_fire.run(2000);

  double niter = optimizer_mixed_descent_fire.get_niter();

  std::cout << niter << endl;
}