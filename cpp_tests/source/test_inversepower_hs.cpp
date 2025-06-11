#include <gtest/gtest.h>
#include <iostream>
#include <stdexcept>
#include <cmath>

#include "pele/array.hpp"
#include "pele/inversepower_hs.hpp"
#include "pele/inversepower.hpp"

using pele::Array;

static double const EPS = std::numeric_limits<double>::min();
#define EXPECT_NEAR_RELATIVE(A, B, T) \
  EXPECT_NEAR(A / (fabs(A) + fabs(B) + EPS), B / (fabs(A) + fabs(B) + EPS), T)

/*
 * InversePowerHS tests
 */

class InversePowerHSTest : public ::testing::Test {
 public:
  double pow, eps, sigma;
  Array<double> x, g, gnum, radii;
  virtual void SetUp() {
    pow = 2.5;
    eps = 1.0;
    sigma = 0.2;
    // two particles
    x = Array<double>(6);
    x[0] = 0.0; x[1] = 0.0; x[2] = 0.0;
    x[3] = 0.0; x[4] = 0.0; x[5] = 0.0;
    radii = Array<double>(2);
    radii[0] = 0.5;
    radii[1] = 0.5;
    g = Array<double>(x.size());
    gnum = Array<double>(x.size());
  }
};

TEST_F(InversePowerHSTest, EnergyInfinity) {
    pele::InversePowerHS<3> pot(pow, eps, sigma, radii);
    double d_hs = radii[0] + radii[1];
    x[3] = d_hs - 0.1;
    double e = pot.get_energy(x);
    ASSERT_TRUE(std::isinf(e));
}

TEST_F(InversePowerHSTest, EnergyZero) {
    pele::InversePowerHS<3> pot(pow, eps, sigma, radii);
    double d_hs = radii[0] + radii[1];
    double d_c = d_hs * (1.0 + sigma);
    x[3] = d_c + 0.1;
    double e = pot.get_energy(x);
    ASSERT_NEAR(e, 0.0, 1e-12);
}

TEST_F(InversePowerHSTest, EnergyGradient_AgreesWithNumerical) {
  pele::InversePowerHS<3> pot(pow, eps, sigma, radii);
  double d_hs = radii[0] + radii[1];
  double d_c = d_hs * (1.0 + sigma);
  x[3] = d_hs + (d_c-d_hs)/2.0; // midpoint
  double r = x[3];

  double e = pot.get_energy_gradient(x, g);

  double expected_e = std::pow(1.0 - r / d_c, pow) * eps / pow;
  ASSERT_NEAR(e, expected_e, 1e-10);
  
  pot.numerical_gradient(x, gnum, 1e-6);
  for (size_t k = 0; k < 6; ++k) {
    ASSERT_NEAR(g[k], gnum[k], 1e-6);
  }
}

TEST_F(InversePowerHSTest, EnergyGradientHessian_AgreesWithNumerical) {
  pele::InversePowerHS<3> pot(pow, eps, sigma, radii);
  double d_hs = radii[0] + radii[1];
  double d_c = d_hs * (1.0 + sigma);
  x[3] = d_hs + (d_c-d_hs)/2.0;
  
  Array<double> h(x.size() * x.size());
  Array<double> hnum(h.size());
  double e = pot.get_energy_gradient_hessian(x, g, h);
  
  pot.numerical_gradient(x, gnum);
  pot.numerical_hessian(x, hnum);
  
  double r = x[3];
  double expected_e = std::pow(1.0 - r / d_c, pow) * eps / pow;
  EXPECT_NEAR(e, expected_e, 1e-10);
  
  for (size_t i = 0; i < g.size(); ++i) {
    ASSERT_NEAR(g[i], gnum[i], 1e-6);
  }
  for (size_t i = 0; i < h.size(); ++i) {
    ASSERT_NEAR(h[i], hnum[i], 1e-3);
  }
}


TEST_F(InversePowerHSTest, InverseIntPowerHS_AgreesWithInversePowerHS) {
  const int int_pow = 4;
  pele::InversePowerHS<3> pot(int_pow, eps, sigma, radii);
  pele::InverseIntPowerHS<3, int_pow> pot_int(eps, sigma, radii);
  
  double d_hs = radii[0] + radii[1];
  double d_c = d_hs * (1.0 + sigma);
  x[3] = d_hs + (d_c-d_hs)/2.0;

  const double e = pot.get_energy(x);
  const double e_int = pot_int.get_energy(x);
  ASSERT_NEAR(e, e_int, 1e-10);
  
  pot.numerical_gradient(x, gnum, 1e-7);
  pele::Array<double> gnum_int(gnum.size());
  pot_int.numerical_gradient(x, gnum_int, 1e-7);
  for (size_t k = 0; k < gnum.size(); ++k) {
    ASSERT_NEAR(gnum[k], gnum_int[k], 1e-6);
  }
}

TEST_F(InversePowerHSTest, InverseHalfIntPowerHS_AgreesWithInversePowerHS) {
  const double half_int_pow = 2.5;
  const int pow2 = 5;
  pele::InversePowerHS<3> pot(half_int_pow, eps, sigma, radii);
  pele::InverseHalfIntPowerHS<3, pow2> pot_half_int(eps, sigma, radii);
  
  double d_hs = radii[0] + radii[1];
  double d_c = d_hs * (1.0 + sigma);
  x[3] = d_hs + (d_c-d_hs)/2.0;

  const double e = pot.get_energy(x);
  const double e_half_int = pot_half_int.get_energy(x);
  ASSERT_NEAR(e, e_half_int, 1e-10);
  
  pot.numerical_gradient(x, gnum, 1e-7);
  pele::Array<double> gnum_half_int(gnum.size());
  pot_half_int.numerical_gradient(x, gnum_half_int, 1e-7);
  for (size_t k = 0; k < gnum.size(); ++k) {
    ASSERT_NEAR(gnum[k], gnum_half_int[k], 1e-6);
  }
}

TEST_F(InversePowerHSTest, AgreesWithInversePowerOutsideHSBoundary) {
  // 1. Create InversePowerHS potential
  pele::InversePowerHS<3> pot_hs(pow, eps, sigma, radii);

  // 2. Create InversePower potential with scaled radii
  Array<double> radii_scaled(radii.size());
  for (size_t i=0; i<radii.size(); ++i) {
      radii_scaled[i] = radii[i] * (1.0 + sigma);
  }
  pele::InversePower<3> pot_ip(pow, eps, radii_scaled);

  // 3. Set particle distance r such that d_hs < r < d_c
  double d_hs = radii[0] + radii[1];
  double d_c = d_hs * (1.0 + sigma);
  x[3] = d_hs + (d_c - d_hs) / 2.0; // r at midpoint of the shell
  
  // Check energy
  double e_hs = pot_hs.get_energy(x);
  double e_ip = pot_ip.get_energy(x);
  EXPECT_NEAR(e_hs, e_ip, 1e-10);

  // Check gradient
  Array<double> g_hs(x.size());
  Array<double> g_ip(x.size());
  pot_hs.get_energy_gradient(x, g_hs);
  pot_ip.get_energy_gradient(x, g_ip);
  for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_NEAR(g_hs[i], g_ip[i], 1e-9);
  }

  // Check hessian
  Array<double> h_hs(x.size() * x.size());
  Array<double> h_ip(x.size() * x.size());
  pot_hs.get_energy_gradient_hessian(x, g_hs, h_hs);
  pot_ip.get_energy_gradient_hessian(x, g_ip, h_ip);
  for (size_t i = 0; i < h_hs.size(); ++i) {
      EXPECT_NEAR(h_hs[i], h_ip[i], 1e-9);
  }
}

TEST_F(InversePowerHSTest, IntAgreesWithInversePowerOutsideHSBoundary) {
  const int int_pow = 4;
  
  // 1. Create InverseIntPowerHS potential
  pele::InverseIntPowerHS<3, int_pow> pot_hs(eps, sigma, radii);

  // 2. Create InverseIntPower potential with scaled radii
  Array<double> radii_scaled(radii.size());
  for (size_t i=0; i<radii.size(); ++i) {
      radii_scaled[i] = radii[i] * (1.0 + sigma);
  }
  pele::InverseIntPower<3, int_pow> pot_ip(eps, radii_scaled);

  // 3. Set particle distance r such that d_hs < r < d_c
  double d_hs = radii[0] + radii[1];
  double d_c = d_hs * (1.0 + sigma);
  x[3] = d_hs + (d_c - d_hs) / 2.0;
  
  // Check energy
  double e_hs = pot_hs.get_energy(x);
  double e_ip = pot_ip.get_energy(x);
  EXPECT_NEAR(e_hs, e_ip, 1e-10);
  
  // Check gradient
  Array<double> g_hs(x.size());
  Array<double> g_ip(x.size());
  pot_hs.get_energy_gradient(x, g_hs);
  pot_ip.get_energy_gradient(x, g_ip);
  for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_NEAR(g_hs[i], g_ip[i], 1e-9);
  }
}

TEST_F(InversePowerHSTest, HalfIntAgreesWithInversePowerOutsideHSBoundary) {
  const int pow2 = 5;
  
  // 1. Create InverseHalfIntPowerHS potential
  pele::InverseHalfIntPowerHS<3, pow2> pot_hs(eps, sigma, radii);

  // 2. Create InverseHalfIntPower potential with scaled radii
  Array<double> radii_scaled(radii.size());
  for (size_t i=0; i<radii.size(); ++i) {
      radii_scaled[i] = radii[i] * (1.0 + sigma);
  }
  pele::InverseHalfIntPower<3, pow2> pot_ip(eps, radii_scaled);

  // 3. Set particle distance r such that d_hs < r < d_c
  double d_hs = radii[0] + radii[1];
  double d_c = d_hs * (1.0 + sigma);
  x[3] = d_hs + (d_c - d_hs) / 2.0;
  
  // Check energy
  double e_hs = pot_hs.get_energy(x);
  double e_ip = pot_ip.get_energy(x);
  EXPECT_NEAR(e_hs, e_ip, 1e-10);
  
  // Check gradient
  Array<double> g_hs(x.size());
  Array<double> g_ip(x.size());
  pot_hs.get_energy_gradient(x, g_hs);
  pot_ip.get_energy_gradient(x, g_ip);
  for (size_t i = 0; i < x.size(); ++i) {
      EXPECT_NEAR(g_hs[i], g_ip[i], 1e-9);
  }
} 