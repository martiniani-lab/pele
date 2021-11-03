// Tests for Newton's method with singular values


#include "pele/array.hpp"


#include <cstddef>
#include <iostream>
#include <gtest/gtest.h>
#include <memory>
#include <pele/harmonic.hpp>
#include <pele/newton.hpp>
#include <pele/rosenbrock.hpp>




using pele::Array;
using std::cout;



// Test that the Newton's Method should take a single step for the Harmonic oscillator
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







