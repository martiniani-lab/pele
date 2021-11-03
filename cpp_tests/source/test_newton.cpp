// Tests for Newton's method with singular values


#include "pele/array.hpp"


#include <cstddef>
#include <iostream>
#include <gtest/gtest.h>
#include <memory>
#include <pele/harmonic.hpp>
#include <pele/newton.hpp>




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
    newton.one_iteration();

}








