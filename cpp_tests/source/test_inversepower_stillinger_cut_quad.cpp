#include <cstddef>
#include <gtest/gtest.h>

#include "pele/array.hpp"
#include "pele/base_potential.hpp"
#include "pele/inversepower_stillinger_cut_quad.hpp"
#include "test_utils.hpp"

#define EXPECT_NEAR_RELATIVE(A, B, T)                                          \
  EXPECT_NEAR(A / (fabs(A) + fabs(B) + EPS), B / (fabs(A) + fabs(B) + EPS), T)

class BasicIPSCutQuadTest : public ::testing::Test {
public:
  size_t ndim;
  size_t ndof;
  size_t npart;
  pele::Array<double> x, radii;
  size_t pow;
  std::shared_ptr<pele::InversePowerStillingerCutQuad<2>> pot;
  double a, eps;
  double etrue, e0;
  double expected_cutoff;
  double cutoff_factor;
  double v0;

  virtual void SetUp() {
    ndim = 2;
    npart = 2;
    eps = 1e-15;
    ndof = ndim * npart;
    x = {1.1, 3, 1, 2};
    radii = {1.05, 1.05};
    a = radii[0] * 2.;

    // Potential parameters
    pow = 4;
    v0 = 1.0;
    cutoff_factor = 1.0;
    expected_cutoff = 2.0 * cutoff_factor * radii[0];
    double dr = std::sqrt(std::pow(x[0] - x[2], 2) + std::pow(x[1] - x[3], 2));

    etrue = get_test_energy(dr, cutoff_factor, v0, pow, radii);
    
    e0 = get_test_energy(expected_cutoff - eps, cutoff_factor, v0, pow, radii);
    pot = std::make_shared<pele::InversePowerStillingerCutQuad<2>>(
        pow, v0, cutoff_factor, radii);
  }

  // Expected energy for a pair of particles with distance dr
  double get_test_energy(double dr, double cutoff_factor, double v0, size_t pow,
                         Array<double> radii) {
    double dij = (radii[0] + radii[1]) *
                 (1 - 2 * cutoff_factor * abs(radii[0] - radii[1]));
    double dr_scaled = dr / dij;
    double c0 = (-1.0 / 8.0) * (8 + 6 * pow + pow * pow) *
                std::pow(cutoff_factor, -pow) * v0;
    double c2 = 1 / 4.0 * (pow * pow + 4 * pow) *
                std::pow(cutoff_factor, -pow - 2) * v0;
    double c4 = -1 / 8.0 * (pow * pow + 2 * pow) *
                std::pow(cutoff_factor, -pow - 4) * v0;
    double etest = std::pow(v0 / dr_scaled, pow) + c0 +
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

TEST_F(BasicIPSCutQuadTest, e0_works) {
  double dr0 = expected_cutoff - eps;
  pele::Array<double> x0 = {0, 0, dr0, 0};
  const double e = pot->get_energy(x0);
  ASSERT_NEAR(e, e0, 1e-14);
  ASSERT_NEAR(e, 0, 1e-14);
}

class TestInversePowerStillingerCutQuadAuto : public PotentialTest {
  size_t ndim;
  size_t ndof;
  size_t npart;
  size_t exponent;
  double diameters, rcut;
  // do not declare etrue here
  pele::Array<double> radii;

  double v0, cutoff_factor, expected_cutoff;

  virtual void SetUp() {
    ndim = 2;
    npart = 2;
    ndof = ndim * npart;
    x = {1.1, 3, 1, 2};
    radii = {1.05, 1.05};
    diameters = radii[0] * 2;
    exponent = 12;
    v0 = 1.0;
    cutoff_factor = 1.0;

    // cuts off when distance is greater than two radii and a factor
    expected_cutoff = cutoff_factor * 2 * radii[0];

    double dr = std::sqrt(std::pow(x[0] - x[2], 2) + std::pow(x[1] - x[3], 2));
    etrue = get_test_energy(dr, cutoff_factor, v0, exponent, radii);
    std::cout << "test energy " << etrue << std::endl;

    pot = std::make_shared<pele::InversePowerStillingerCutQuad<2>>(
        exponent, v0, cutoff_factor, radii);
  }
  // Expected energy for a pair of particles with distance dr
  double get_test_energy(double dr, double cutoff_factor, double v0, size_t pow,
                         Array<double> radii) {
    double dij = (radii[0] + radii[1]) *
                 (1 - 2 * cutoff_factor * abs(radii[0] - radii[1]));
    double dr_scaled = dr / dij;
    double c0 = (-1.0 / 8.0) * (8 + 6 * pow + pow * pow) *
                std::pow(cutoff_factor, -pow) * v0;
    double c2 = 1 / 4.0 * (pow * pow + 4 * pow) *
                std::pow(cutoff_factor, -pow - 2) * v0;
    double c4 = -1 / 8.0 * (pow * pow + 2 * pow) *
                std::pow(cutoff_factor, -pow - 4) * v0;
    double etest = std::pow(v0 / dr_scaled, pow) + c0 +
                   c2 * dr_scaled * dr_scaled +
                   c4 * dr_scaled * dr_scaled * dr_scaled * dr_scaled;
    return etest;
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
