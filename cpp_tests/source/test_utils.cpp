
#include "pele/utils.hpp"
#include "pele/array.h"

#include <iostream>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>
#include <cmath>
#include <memory>

using pele::Array;

TEST(utils, box_length_3d) {
    Array<double> hs_radii {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; // radii

    double dim = 3.0;

    double phi = 0.6;


    double box_length = pele::get_box_length(hs_radii, dim, phi);

    std::cout << "box_length = " << box_length << std::endl;

}


TEST(utils, box_length_2d) {
    Array<double> hs_radii {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; // radii

    double dim = 2.0;

    double phi = 0.9;


    double box_length = pele::get_box_length(hs_radii, dim, phi);

    std::cout << "box_length = " << box_length << std::endl;
}

TEST(utils, generate_random_coordinates) {
    double box_length = 1.0;
    int nparticles = 10;
    int dim = 2;

    Array<double> coords = pele::generate_random_coordinates(box_length, nparticles, dim);

    for (int i = 0; i < nparticles; ++i) {
        for (int j = 0; j < dim; ++j) {
            ASSERT_LE(coords[i * dim + j], box_length);
            ASSERT_GE(coords[i * dim + j], 0.0);
        }
    }
}


TEST(utils, generate_radii) { 
    int n_1 = 5;
    int n_2 = 5;

    double r_1 = 1.0;
    double r_2 = 1.4;
    double r_std_1 = 0.05;
    double r_std_2 = 0.05*1.4;

    Array<double> radii = pele::generate_radii(n_1, n_2, r_1, r_2, r_std_1, r_std_2);

    std::cout << "radii = " << radii << std::endl;
    // TODO: check that radii are within the correct range
}

    
    
