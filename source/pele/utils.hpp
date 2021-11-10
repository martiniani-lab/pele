/**
 * Utility functions for testing and debugging.
 */


#include "pele/array.hpp"
#include <cmath>
#include <cstddef>
#include <math.h>
#include <random>
#include <vector>

// Float definition of pi
#define PI 3.14159265358979323846


typedef std::vector<std::vector<double>> matrix;



namespace pele {
/*
 * @brief      Generate a radii for a bidisperse mixture of particles
 *
 * @details    Generate a radii for a bidisperse mixture of particles.
 *             The radii are picked from gaussian distributions with
 *             mean and standard deviation given by the parameters
 *             r_1, r_2, r_std_1, r_std_2.
 *
 * @param      n_1:         number of particles of type 1
 *             n_2:         number of particles of type 2
 *             r_1:         average radius of type 1
 *             r_2:         average radius of type 2
 *             r_std_1:     standard deviation of type 1
 *             r_std_2:     standard deviation of type 2
 *
 * @return     Array of radii
 */
 inline Array<double> generate_radii(int n_1, int n_2, double r_1, double r_2, double r_std_1, double r_std_2) {

     Array<double> radii(n_1+n_2);
    
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int i=0; i<n_1; i++) {
        radii[i] = r_1 + r_std_1 * std::normal_distribution<double>(0,1)(gen);
    }
    for (int i=n_1; i<n_1+n_2; i++) {
        radii[i] = r_2 + r_std_2 * std::normal_distribution<double>(0,1)(gen);
    }
    return radii;
}



/**
 * @brief      Gets box length
 *
 * @details    Gets box length
 *
 * @param      hs_radii: radii of spheres in the box
 *             dim:      dimension
 *             phi:      packing fraction
 *
 * @return     length of box for target packing fraction
 */
inline double get_box_length(Array<double> hs_radii, int dim,double phi) {
    if (dim==3) {
        double vol_spheres = 4./3. * PI * (hs_radii*hs_radii*hs_radii).sum();
        return pow(vol_spheres/phi, 1./3.);
    }
    else if (dim == 2) {
        double vol_discs = PI * (hs_radii*hs_radii).sum();
        return pow(vol_discs/phi, 1./2.);
    }
    else {
        throw std::runtime_error("get_box_length: dimension not implemented");
    }
}

/*
 * @brief      Generates random coordinates in a box
 *
 * @details    Generates random coordinates in a box
 *
 * @param      box_length: length of box
 *             n_particles: number of particles
 *             dim:        dimension
 *
 * @return     Array of random coordinates
 */
inline Array<double> generate_random_coordinates(double box_length, int n_particles, int dim) {

    Array<double> coords(n_particles*dim);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0, box_length);

    for (int i=0; i<n_particles; i++) {
        for (int j=0; j<dim; j++) {
            coords[i*dim+j] = dist(gen);
        }
    }
    return coords;
}

inline bool origin_in_hull_2d(matrix disp_list) {

}

inline std::vector<double> cross_2d(matrix x,  matrix y) {

    std::vector<double> cross_product(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        cross_product[i] = x[i][1] * y[i][2] - x[i][2] - y[i][1];
    }
    return cross_product;
}


inline std::vector<size_t> sort_circle(matrix vertices) {
    // Split the vertices based on whether the x coordinate is to the right or left of the origin
    std::vector<size_t> left_indices_list;
    std::vector<size_t> right_indices_list;
    for (size_t i = 0; i < vertices.size(); i++) {
        if (vertices[i][0] < 0) {
            left_indices_list.push_back(i);
        }
        else {
            right_indices_list.push_back(i);
        }
    }
    // get the 
    std::vector<double> left_tan_list;
    std::vector<double> right_tan_list;

    for (size_t i = 0; i < left_indices_list.size(); i++) {
        left_tan_list.push_back(vertices[left_indices_list[i]][1] / vertices[left_indices_list[i]][0]);
    }
    for (size_t i = 0; i < right_indices_list.size(); i++) {
        right_tan_list.push_back(vertices[right_indices_list[i]][1] / vertices[right_indices_list[i]][0]);
    }

    // Sort both lists by tangent
    std::sort(left_tan_list.begin(), left_tan_list.end());
    std::sort(right_tan_list.begin(), right_tan_list.end());

    // concanetate the two lists and return
    std::vector<size_t> sorted_indices_list;
    sorted_indices_list.insert(sorted_indices_list.end(), left_indices_list.begin(), left_indices_list.end());
    sorted_indices_list.insert(sorted_indices_list.end(), right_indices_list.begin(), right_indices_list.end());
    return sorted_indices_list;
}




}