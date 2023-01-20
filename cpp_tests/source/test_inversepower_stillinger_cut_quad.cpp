#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>

#include "pele/array.hpp"
#include "pele/base_potential.hpp"
#include "pele/inversepower_stillinger_cut_quad.hpp"
#include "test_utils.hpp"

#define EXPECT_NEAR_RELATIVE(A, B, T) \
  EXPECT_NEAR(A / (fabs(A) + fabs(B) + EPS), B / (fabs(A) + fabs(B) + EPS), T)

class BasicIPSCutQuadTest : public ::testing::Test {
 public:
  size_t ndim;
  size_t ndof;
  size_t npart;
  pele::Array<double> x, radii;
  int pow;
  std::shared_ptr<pele::InversePowerStillingerCutQuad<2>> pot;
  double a, eps;
  double etrue, e0_test;
  double expected_cutoff;
  double cutoff_factor;
  double v0;
  double non_additivity;

  virtual void SetUp() {
    ndim = 2;
    npart = 2;
    eps = 1e-15;
    ndof = ndim * npart;
    x = {1.1, 3, 1, 2};
    radii = {1.05, 1.15};
    a = radii[0] * 2.;
    non_additivity = 0.2;

    // Potential parameters
    // Set non obvious parameters to avoid running test for a trivial case
    pow = 4;
    v0 = 1.5;
    cutoff_factor = 1.25;
    double dij = (radii[0] + radii[1]) *
                 (1 - 2 * non_additivity * abs(radii[0] - radii[1]));
    expected_cutoff = cutoff_factor * dij;
    double dr = std::sqrt(std::pow(x[0] - x[2], 2) + std::pow(x[1] - x[3], 2));

    etrue = get_test_energy(dr, cutoff_factor, v0, pow, radii);

    e0_test =
        get_test_energy(expected_cutoff - eps, cutoff_factor, v0, pow, radii);
    pot = std::make_shared<pele::InversePowerStillingerCutQuad<2>>(
        pow, v0, cutoff_factor, radii, non_additivity);
  }

  // Expected energy for a pair of particles with distance dr
  double get_test_energy(double dr, double cutoff_factor, double v0, int pow,
                         Array<double> radii) {
    double dij = (radii[0] + radii[1]) *
                 (1 - 2 * non_additivity * abs(radii[0] - radii[1]));
    double dr_scaled = dr / dij;
    double c0 = (-1.0 / 8.0) * (8 + 6 * pow + pow * pow) *
                std::pow(cutoff_factor, -pow) * v0;
    double c2 = 1 / 4.0 * (pow * pow + 4 * pow) *
                std::pow(cutoff_factor, -pow - 2) * v0;
    double c4 = -1 / 8.0 * (pow * pow + 2 * pow) *
                std::pow(cutoff_factor, -pow - 4) * v0;
    double etest = v0 * std::pow(1 / dr_scaled, pow) + c0 +
                   c2 * dr_scaled * dr_scaled +
                   c4 * dr_scaled * dr_scaled * dr_scaled * dr_scaled;
    return etest;
  }
};

TEST_F(BasicIPSCutQuadTest, Pow_Works) {
  for (size_t power = 0; power < 129; ++power) {
    const double inp = 0.3;
    if (power % 2 == 0) {
      pele::InversePowerStillingerQuadCutInteraction in(power, 1.0, 1.0);
      const double e = pot->get_energy(x);
      ASSERT_NEAR(etrue, e, 1e-10);
    }
  }
}

TEST_F(BasicIPSCutQuadTest, Energy_WorksIntPower12) {
  // TODO: Figure out how to do a constexpr loop
  constexpr size_t power = 12;
  pele::InversePowerStillingerQuadCutInteractionInt<power> in(1.0, 1.0);
  const double e = pot->get_energy(x);
  ASSERT_NEAR(etrue, e, 1e-10);
}

// Checks that pairwise energy, gradient and hessian are zero at cutoff
TEST_F(BasicIPSCutQuadTest, zero_at_cutoff) {
  double dr0 = expected_cutoff - eps;
  pele::Array<double> x0 = {0, 0, dr0, 0};
  const double e = pot->get_energy(x0);
  Array<double> g(x0.size());
  Array<double> gh(x0.size());
  Array<double> h(x0.size() * x0.size());

  const double eg = pot->get_energy_gradient(x0, g);

  const double eh = pot->get_energy_gradient_hessian(x0, gh, h);

  ASSERT_NEAR(e, e0_test, 1e-14);
  ASSERT_NEAR(e, 0, 1e-14);
  ASSERT_NEAR(eg, e, 1e-14);
  ASSERT_NEAR(eh, e, 1e-14);
  for (size_t i = 0; i < g.size(); ++i) {
    ASSERT_NEAR(0, g[i], 1e-10);
  }
  for (size_t i = 0; i < gh.size(); ++i) {
    ASSERT_NEAR(0, gh[i], 1e-10);
    for (size_t i = 0; i < h.size(); ++i) {
      ASSERT_NEAR(h[i], 0, 1e-10);
    }
  }
}

/*
 * Base class for testing variants of the InversePowerStillingerCutQuad
 * potential. The potential should the same but different optimizations are
 * made based on information available at compile time
 */
class BaseTestInversePowerStillingerCutQuadAuto : public PotentialTest {
 protected:
  size_t ndim;
  size_t ndof;
  size_t npart;
  static constexpr size_t exponent = 12;
  double diameters, rcut;
  // do not declare etrue here
  pele::Array<double> radii;
  double non_additivity;

  double v0, cutoff_factor, expected_cutoff;

  // Expected energy for a pair of particles with distance dr
  // Expected energy for a pair of particles with distance dr
  double get_test_energy(double dr, double cutoff_factor, double v0, int pow,
                         Array<double> radii) {
    double dij = (radii[0] + radii[1]) *
                 (1 - 2 * non_additivity * abs(radii[0] - radii[1]));
    double dr_scaled = dr / dij;
    double c0 = (-1.0 / 8.0) * (8 + 6 * pow + pow * pow) *
                std::pow(cutoff_factor, -pow) * v0;
    double c2 = 1 / 4.0 * (pow * pow + 4 * pow) *
                std::pow(cutoff_factor, -pow - 2) * v0;
    double c4 = -1 / 8.0 * (pow * pow + 2 * pow) *
                std::pow(cutoff_factor, -pow - 4) * v0;
    double etest = v0 * std::pow(1 / dr_scaled, pow) + c0 +
                   c2 * dr_scaled * dr_scaled +
                   c4 * dr_scaled * dr_scaled * dr_scaled * dr_scaled;
    return etest;
  }

  void init_potential_parameters() {
    ndim = 2;
    npart = 2;
    ndof = ndim * npart;
    x = {1.1, 3, 1, 1.5};
    radii = {1.05, 1.15};
    diameters = radii[0] * 2;
    v0 = 1.5;
    cutoff_factor = 1.25;
    non_additivity = 0.0;

    // cuts off when distance is greater than two radii and a factor
    expected_cutoff = cutoff_factor * 2 * radii[0];

    double dr = std::sqrt(std::pow(x[0] - x[2], 2) + std::pow(x[1] - x[3], 2));
    etrue = get_test_energy(dr, cutoff_factor, v0, exponent, radii);
  }
};

class TestInversePowerStillingerCutQuadAuto
    : public BaseTestInversePowerStillingerCutQuadAuto {
  virtual void SetUp() {
    init_potential_parameters();
    pot = std::make_shared<pele::InversePowerStillingerCutQuad<2>>(
        exponent, v0, cutoff_factor, radii);
  }
};

TEST_F(TestInversePowerStillingerCutQuadAuto, Energy_Works) { test_energy(); }

TEST_F(TestInversePowerStillingerCutQuadAuto,
       EnergyGradient_AgreesWithNumerical) {
  test_energy_gradient();
}

TEST_F(TestInversePowerStillingerCutQuadAuto,
       EnergyGradientHessian_AgreesWithNumerical) {
  test_energy_gradient_hessian();
}

class TestInversePowerStillingerCutQuadAutoInt
    : public BaseTestInversePowerStillingerCutQuadAuto {
  virtual void SetUp() {
    init_potential_parameters();
    pot = std::make_shared<pele::InversePowerStillingerCutQuadInt<2, exponent>>(
        v0, cutoff_factor, radii);
  }
};

TEST_F(TestInversePowerStillingerCutQuadAutoInt, Energy_Works) {
  test_energy();
}
TEST_F(TestInversePowerStillingerCutQuadAutoInt,
       EnergyGradient_AgreesWithNumerical) {
  test_energy_gradient();
}
TEST_F(TestInversePowerStillingerCutQuadAutoInt,
       EnergyGradientHessian_AgreesWithNumerical) {
  test_energy_gradient_hessian();
}
