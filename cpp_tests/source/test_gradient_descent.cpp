// Tests for Newton's method with singular values
#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <pele/harmonic.hpp>
#include <pele/inversepower.hpp>
#include <pele/newton.hpp>
#include <pele/rosenbrock.hpp>

#include "pele/array.hpp"
#include "pele/cosine_gradient_descent.hpp"

using namespace pele;

TEST(CosineGradientDescent, rosenbrock) {
  std::cout << "Testing Cosine Gradient Descent for the Rosenbrock function"
            << std::endl;

  auto rosenbrock = std::make_shared<pele::RosenBrock>();
  Array<double> x0(2, 0);  // Initial guess for the optimization

  pele::CosineGradientDescent optimizer(rosenbrock, x0);

  optimizer.run(10000000);  // Run the optimizer
  Array<double> x = optimizer.get_x();
  double nfev = optimizer.get_nfev();
  double niter = optimizer.get_niter();
  double f_final = optimizer.get_f();

  std::cout << "x :" << x << "\n";
  std::cout << "nfev :" << nfev << "\n";
  std::cout << "niter :" << niter << "\n";
  std::cout << "f_final :" << f_final << "\n";

  // Validate the solution - the Rosenbrock minimum is at (1, 1)
  ASSERT_NEAR(x[0], 1, 1e-5);
  ASSERT_NEAR(x[1], 1, 1e-5);
  ASSERT_NEAR(f_final, 0,
              1e-5);  // The minimum value of the Rosenbrock function is 0
}

TEST(CosineGradientDescent, harmonic) {
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

  pele::CosineGradientDescent cos_gd(harmonic, x);
  cos_gd.run();
  double nfev = cos_gd.get_nfev();
  double niter = cos_gd.get_niter();
  double f_final = cos_gd.get_f();
  Array<double> x_final(dim);
  x_final = cos_gd.get_x();

  ASSERT_NEAR(x_final[0], 0, 1e-10);
  ASSERT_NEAR(x_final[1], 0, 1e-10);
  ASSERT_NEAR(f_final, 0, 1e-10);
}