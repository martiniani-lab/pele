#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "pele/array.hpp"
#include "pele/inversepower.hpp"

#include "pele/combine_potentials.hpp"
#include "pele/utils.hpp"

using pele::Array;
using pele::InversePower;

TEST(CombinedPotential, TestEnergyGradientHessianWithoutCellLists) {
  static const size_t dim = 2;
  size_t n_particles;
  size_t n_dof;
  double power;
  double eps_a;
  double eps_b;
  double phi;
  pele::Array<double> x;
  pele::Array<double> radii_a;
  pele::Array<double> radii_b;
  pele::Array<double> xfinal;
  pele::Array<double> boxvec;

  eps_a = 1.0;
  eps_b = 1e-3;

  power = 2.5;
  const int pow2 = 5;

  n_particles = 16;
  n_dof = n_particles * dim;
  phi = 0.9;

  int n_1 = n_particles / 2;
  int n_2 = n_particles - n_1;

  double r_1 = 1.0;
  double r_2 = 1.0;
  double r_std1 = 0.05;
  double r_std2 = 0.05;

  radii_a = pele::generate_radii(n_1, n_2, r_1, r_2, r_std1, r_std2);

  radii_b = 2 * radii_a;

  double box_length = pele::get_box_length(radii_a, dim, phi);

  boxvec = {box_length, box_length};
  bool exact_sum = false;
  double ncellsx_scale_a = pele::get_ncellx_scale(radii_a, boxvec, 1);
  double ncellsx_scale_b = pele::get_ncellx_scale(radii_a, boxvec, 1);

  x = pele::generate_random_coordinates(box_length, n_particles, dim);

  std::shared_ptr<pele::InverseHalfIntPowerPeriodic<dim, pow2>> pot_a =
      std::make_shared<pele::InverseHalfIntPowerPeriodic<dim, pow2>>(
          eps_a, radii_a, boxvec, exact_sum);
  std::shared_ptr<pele::InverseHalfIntPowerPeriodic<dim, pow2>> pot_b =
      std::make_shared<pele::InverseHalfIntPowerPeriodic<dim, pow2>>(
          eps_b, radii_b, boxvec, exact_sum);

  std::shared_ptr<pele::CombinedPotential> pot =
      std::make_shared<pele::CombinedPotential>();

  pot->add_potential(pot_a);
  pot->add_potential(pot_b);

  double energy = pot->get_energy(x);

  double energy_a = pot_a->get_energy(x);
  double energy_b = pot_b->get_energy(x);

  ASSERT_EQ(energy, energy_a + energy_b);

  pele::Array<double> grad(n_dof);
  pele::Array<double> hess(n_dof * n_dof);

  pele::Array<double> grad_a(n_dof);
  pele::Array<double> hess_a(n_dof * n_dof);

  pele::Array<double> grad_b(n_dof);
  pele::Array<double> hess_b(n_dof * n_dof);

  pot->get_energy_gradient(x, grad);
  pot_a->get_energy_gradient(x, grad_a);
  pot_b->get_energy_gradient(x, grad_b);

  for (size_t i = 0; i < n_dof; i++) {
    ASSERT_NEAR(grad[i], grad_a[i] + grad_b[i], 1e-9);
  }

  pot->get_energy_gradient_hessian(x, grad, hess);
  pot_a->get_energy_gradient_hessian(x, grad_a, hess_a);
  pot_b->get_energy_gradient_hessian(x, grad_b, hess_b);

  for (size_t i = 0; i < n_dof * n_dof; i++) {
    ASSERT_NEAR(hess[i], hess_a[i] + hess_b[i], 1e-9);
  }
}

TEST(EXTENDED_POTENTIAL, TEST_ALL_ENERGY_GRADIENT_HESSIAN_FUNCTIONS) {
  static const size_t dim = 2;
  size_t n_particles;
  size_t n_dof;
  double power;
  double eps_a;
  double eps_b;
  double phi;
  pele::Array<double> x;
  pele::Array<double> radii_a;
  pele::Array<double> radii_b;
  pele::Array<double> xfinal;
  pele::Array<double> boxvec;

  eps_a = 1.0;
  eps_b = 1e-3;

  power = 2.5;
  const int pow2 = 5;

  n_particles = 16;
  n_dof = n_particles * dim;
  phi = 0.9;

  int n_1 = n_particles / 2;
  int n_2 = n_particles - n_1;

  double r_1 = 1.0;
  double r_2 = 1.0;
  double r_std1 = 0.05;
  double r_std2 = 0.05;

  radii_a = pele::generate_radii(n_1, n_2, r_1, r_2, r_std1, r_std2);

  radii_b = 2 * radii_a;

  double box_length = pele::get_box_length(radii_a, dim, phi);

  boxvec = {box_length, box_length};
  bool exact_sum = false;
  double ncellsx_scale_a = pele::get_ncellx_scale(radii_a, boxvec, 1);
  double ncellsx_scale_b = pele::get_ncellx_scale(radii_a, boxvec, 1);

  x = pele::generate_random_coordinates(box_length, n_particles, dim);

  std::shared_ptr<pele::InverseHalfIntPowerPeriodic<dim, pow2>> pot_a =
      std::make_shared<pele::InverseHalfIntPowerPeriodic<dim, pow2>>(
          eps_a, radii_a, boxvec, exact_sum);
  std::shared_ptr<pele::InverseHalfIntPowerPeriodic<dim, pow2>> pot_b =
      std::make_shared<pele::InverseHalfIntPowerPeriodic<dim, pow2>>(
          eps_b, radii_b, boxvec, exact_sum);

  std::shared_ptr<pele::ExtendedPotential> pot =
      std::make_shared<pele::ExtendedPotential>(pot_a, pot_b);

  // check without potential extension switched on
  double energy = pot->get_energy(x);
  double energy_a = pot_a->get_energy(x);
  ASSERT_EQ(energy, energy_a);

  // check with potential extension switched on
  pot->switch_on_extended_potential();
  energy = pot->get_energy(x);
  double energy_b = pot_b->get_energy(x);
  ASSERT_EQ(energy, energy_a + energy_b);

  pele::Array<double> grad(n_dof);
  pele::Array<double> hess(n_dof * n_dof);

  pele::Array<double> grad_a(n_dof);
  pele::Array<double> hess_a(n_dof * n_dof);

  pele::Array<double> grad_b(n_dof);
  pele::Array<double> hess_b(n_dof * n_dof);

  // switch off extended potential
  pot->switch_off_extended_potential();
  pot->get_energy_gradient(x, grad);

  pot_a->get_energy_gradient(x, grad_a);

  for (size_t i = 0; i < n_dof; i++) {
    ASSERT_NEAR(grad[i], grad_a[i], 1e-9);
  }

  pot->switch_on_extended_potential();
  pot->get_energy_gradient(x, grad);
  pot_b->get_energy_gradient(x, grad_b);

  for (size_t i = 0; i < n_dof; i++) {
    ASSERT_NEAR(grad[i], grad_a[i] + grad_b[i], 1e-9);
  }

  // switch off extended potential
  pot->switch_off_extended_potential();
  pot->get_energy_gradient_hessian(x, grad, hess);

  pot_a->get_energy_gradient_hessian(x, grad_a, hess_a);

  for (size_t i = 0; i < n_dof * n_dof; i++) {
    ASSERT_NEAR(hess[i], hess_a[i], 1e-9);
  }

  // switch off extended potential
  pot->switch_on_extended_potential();
  pot->get_energy_gradient_hessian(x, grad, hess);

  pot_b->get_energy_gradient_hessian(x, grad_b, hess_b);

  for (size_t i = 0; i < n_dof * n_dof; i++) {
    ASSERT_NEAR(hess[i], hess_a[i] + hess_b[i], 1e-9);
  }

  // switch on extended potential
  pot->switch_off_extended_potential();
  pot->get_hessian(x, hess);

  pot_a->get_hessian(x, hess_a);

  for (size_t i = 0; i < n_dof * n_dof; i++) {
    ASSERT_NEAR(hess[i], hess_a[i], 1e-9);
  }

  pot->switch_on_extended_potential();
  pot->get_hessian(x, hess);
  pot_b->get_hessian(x, hess_b);

  for (size_t i = 0; i < n_dof * n_dof; i++) {
    ASSERT_NEAR(hess[i], hess_a[i] + hess_b[i], 1e-9);
  }
}

/**
 * @brief Check whether cell list based potentials work with combined potentials
 */
TEST(ExtendedPotential, WithCellLists) {
  static const size_t dim = 2;
  size_t n_particles;
  size_t n_dof;
  double power;
  double eps_a;
  double eps_b;
  double phi;
  pele::Array<double> x;
  pele::Array<double> radii_a;
  pele::Array<double> radii_b;
  pele::Array<double> xfinal;
  pele::Array<double> boxvec;

  eps_a = 1.0;
  eps_b = 1e-3;

  power = 2.5;
  const int pow2 = 5;

  n_particles = 16;
  n_dof = n_particles * dim;
  phi = 0.9;

  int n_1 = n_particles / 2;
  int n_2 = n_particles - n_1;

  double r_1 = 1.0;
  double r_2 = 1.0;
  double r_std1 = 0.05;
  double r_std2 = 0.05;

  radii_a = pele::generate_radii(n_1, n_2, r_1, r_2, r_std1, r_std2);

  radii_b = 2 * radii_a;

  double box_length = pele::get_box_length(radii_a, dim, phi);

  boxvec = {box_length, box_length};
  bool exact_sum = false;
  double ncellsx_scale_a = pele::get_ncellx_scale(radii_a, boxvec, 1);
  double ncellsx_scale_b = pele::get_ncellx_scale(radii_a, boxvec, 1);

  x = pele::generate_random_coordinates(box_length, n_particles, dim);

  std::shared_ptr<pele::InverseHalfIntPowerPeriodicCellLists<dim, pow2>> pot_a =
      std::make_shared<pele::InverseHalfIntPowerPeriodicCellLists<dim, pow2>>(
          eps_a, radii_a, boxvec, ncellsx_scale_a, exact_sum);
  std::shared_ptr<pele::InverseHalfIntPowerPeriodicCellLists<dim, pow2>> pot_b =
      std::make_shared<pele::InverseHalfIntPowerPeriodicCellLists<dim, pow2>>(
          eps_b, radii_b, boxvec, ncellsx_scale_b, exact_sum);

  std::shared_ptr<pele::ExtendedPotential> pot =
      std::make_shared<pele::ExtendedPotential>(pot_a, pot_b);

  // check without potential extension switched on
  double energy = pot->get_energy(x);
  double energy_a = pot_a->get_energy(x);
  ASSERT_EQ(energy, energy_a);

  // check with potential extension switched on
  pot->switch_on_extended_potential();
  energy = pot->get_energy(x);
  double energy_b = pot_b->get_energy(x);
  ASSERT_EQ(energy, energy_a + energy_b);

  pele::Array<double> grad(n_dof);
  pele::Array<double> hess(n_dof * n_dof);

  pele::Array<double> grad_a(n_dof);
  pele::Array<double> hess_a(n_dof * n_dof);

  pele::Array<double> grad_b(n_dof);
  pele::Array<double> hess_b(n_dof * n_dof);

  // switch off extended potential
  pot->switch_off_extended_potential();
  pot->get_energy_gradient(x, grad);

  pot_a->get_energy_gradient(x, grad_a);

  for (size_t i = 0; i < n_dof; i++) {
    ASSERT_NEAR(grad[i], grad_a[i], 1e-9);
  }

  pot->switch_on_extended_potential();
  pot->get_energy_gradient(x, grad);
  pot_b->get_energy_gradient(x, grad_b);

  for (size_t i = 0; i < n_dof; i++) {
    ASSERT_NEAR(grad[i], grad_a[i] + grad_b[i], 1e-9);
  }

  // switch off extended potential
  pot->switch_off_extended_potential();
  pot->get_energy_gradient_hessian(x, grad, hess);
  pot_a->get_energy_gradient_hessian(x, grad_a, hess_a);

  for (size_t i = 0; i < n_dof * n_dof; i++) {
    ASSERT_NEAR(hess[i], hess_a[i], 1e-9);
  }

  // switch off extended potential
  pot->switch_on_extended_potential();
  pot->get_energy_gradient_hessian(x, grad, hess);
  pot_b->get_energy_gradient_hessian(x, grad_b, hess_b);

  for (size_t i = 0; i < n_dof * n_dof; i++) {
    ASSERT_NEAR(hess[i], hess_a[i] + hess_b[i], 1e-9);
  }

  // switch on extended potential
  pot->switch_off_extended_potential();
  pot->get_hessian(x, hess);

  pot_a->get_hessian(x, hess_a);

  for (size_t i = 0; i < n_dof * n_dof; i++) {
    ASSERT_NEAR(hess[i], hess_a[i], 1e-9);
  }

  pot->switch_on_extended_potential();
  pot->get_hessian(x, hess);
  pot_b->get_hessian(x, hess_b);

  for (size_t i = 0; i < n_dof * n_dof; i++) {
    ASSERT_NEAR(hess[i], hess_a[i] + hess_b[i], 1e-9);
  }
}