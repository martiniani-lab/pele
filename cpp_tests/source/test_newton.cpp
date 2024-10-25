// Tests for Newton's method with singular values
#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <pele/harmonic.hpp>
#include <pele/newton.hpp>
#include <pele/rosenbrock.hpp>

#include "pele/array.hpp"
#include "pele/newton_trust.hpp"

using pele::Array;
using std::cout;

// Test that the Newton's Method should take a single step for the Harmonic
// oscillator
TEST(Newton, harmonic) {
  double k = 1.0;
  size_t dim = 2;
  Array<double> origin(2);
  origin[0] = 0;
  origin[1] = 0;

  // define potential
  auto harmonic = std::make_shared<pele::Harmonic>(origin, k, dim);
  // define the initial position

  Array<double> x(dim);
  x[0] = 1;
  x[1] = 1;

  pele::Newton newton(harmonic, x);
  newton.run();
  double nfev = newton.get_nfev();
  double niter = newton.get_niter();
  double f_final = newton.get_f();
  Array<double> x_final(dim);
  x_final = newton.get_x();

  ASSERT_NEAR(x_final[0], 0, 1e-10);
  ASSERT_NEAR(x_final[1], 0, 1e-10);

  ASSERT_EQ(niter, 1);
  ASSERT_NEAR(f_final, 0, 1e-10);
}

// Test that the our Newton's Method should ignore singular values
TEST(Newton, harmonic_singular) {
  double k = 1.0;

  auto flatharmonic = std::make_shared<pele::FlatHarmonic>(1.0);

  // define the initial position
  Array<double> x(2);

  // Direction along potential increase
  x[0] = 1;
  x[1] = -1;

  pele::Newton newton(flatharmonic, x);

  newton.run();

  double nfev = newton.get_nfev();
  double niter = newton.get_niter();
  double f_final = newton.get_f();
  Array<double> x_final(2);
  x_final = newton.get_x();

  ASSERT_NEAR(x_final[0], 0, 1e-10);
  ASSERT_NEAR(x_final[1], 0, 1e-10);

  ASSERT_EQ(niter, 1);
  ASSERT_NEAR(f_final, 0, 1e-10);
}

TEST(Newton, rosenbrock) {
  cout << "Testing Newton's Method for the Rosenbrock function" << std::endl;
  auto rosenbrock = std::make_shared<pele::RosenBrock>();
  Array<double> x0(2, 0);

  pele::Newton newton(rosenbrock, x0);

  newton.run();
  Array<double> x = newton.get_x();
  double nfev = newton.get_nfev();
  double niter = newton.get_niter();
  double f_final = newton.get_f();

  cout << "x :" << x << "\n";
  cout << "nfev :" << nfev << "\n";
  cout << "niter :" << niter << "\n";
  cout << "f_final :" << f_final << "\n";

  // error tolerance can be reduced if optimizer convergence tolerance is
  // decreased
  ASSERT_NEAR(x[0], 1, 1e-5);
  ASSERT_NEAR(x[1], 1, 1e-5);
}

// TEST(NEWTONTRUST, HARMONIC) {
//   double k = 1.0;
//   size_t dim = 2;
//   Array<double> origin(2);
//   origin[0] = 0;
//   origin[1] = 0;

//   // define potential

//   auto harmonic = std::make_shared<pele::Harmonic>(origin, k, dim);
//   // define the initial position

//   Array<double> x(dim);
//   x[0] = 1;
//   x[1] = 1;

//   pele::NewtonTrust newton(harmonic, x);
//   // newton.run();
//   newton.run(2);
//   double nfev = newton.get_nfev();
//   double niter = newton.get_niter();
//   double f_final = newton.get_f();
//   Array<double> x_final(dim);
//   x_final = newton.get_x();
//   ASSERT_NEAR(x_final[0], 0, 1e-10);
//   ASSERT_NEAR(x_final[1], 0, 1e-10);
//   ASSERT_EQ(niter, 1);
//   ASSERT_NEAR(f_final, 0, 1e-10);
// }

// TEST(NEWTONTRUST, ROSENBROCK) {
//   cout << "Testing Newton's Method for the Rosenbrock function" << std::endl;
//   auto rosenbrock = std::make_shared<pele::RosenBrock>();
//   Array<double> x0(2, 0);

//   pele::NewtonTrust newton(rosenbrock, x0);

//   newton.run(1000);
//   Array<double> x = newton.get_x();
//   double nfev = newton.get_nfev();
//   double niter = newton.get_niter();
//   double f_final = newton.get_f();

//   cout << "x :" << x << "\n";
//   cout << "nfev :" << nfev << "\n";
//   cout << "niter :" << niter << "\n";
//   cout << "f_final :" << f_final << "\n";

//   // error tolerance can be reduced if optimizer convergence tolerance is
//   // decreased
//   ASSERT_NEAR(x[0], 1, 1e-5);
//   ASSERT_NEAR(x[1], 1, 1e-5);
// }