#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "pele/harmonic.hpp"
#include "pele/meta_pow.hpp"

#include<hoomd/md/AllPairPotentials.h>
#include<hoomd/Initializers.h>

#include "test_utils.hpp"

using pele::Array;
using pele::pos_int_pow;

static double const EPS = std::numeric_limits<double>::min();
#define EXPECT_NEAR_RELATIVE(A, B, T)  EXPECT_NEAR(A/(fabs(A)+fabs(B) + EPS), B/(fabs(A)+fabs(B) + EPS), T)

//using hoomd::md;

TEST(HOOMD, GET_FORCE_AND_ENERGY){
    
}