/**
 * Utility functions for testing and debugging.
 */


#include "pele/array.hpp"
#include <cmath>
#include <math.h>
#include <random>

// Float definition of pi
#define PI 3.14159265358979323846



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
}