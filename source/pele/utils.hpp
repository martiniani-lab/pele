#ifndef PELE_UTILS_HPP
#define PELE_UTILS_HPP
/**
 * Utility functions for testing and debugging.
 */

#include <cmath>
#include <cstddef>
#include <iostream>
#include <math.h>
#include <random>
#include <vector>

#include "pele/array.hpp"

// Float definition of pi
#define PI 3.14159265358979323846

typedef std::vector<std::vector<double>> matrix;
using namespace std;
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
inline Array<double> generate_bidisperse_radii(int n_1, int n_2, double r_1,
                                               double r_2, double r_std_1,
                                               double r_std_2) {
  Array<double> radii(n_1 + n_2);

  std::random_device rd;
  std::mt19937 gen(rd());

  for (int i = 0; i < n_1; i++) {
    radii[i] = r_1 + r_std_1 * std::normal_distribution<double>(0, 1)(gen);
  }
  for (int i = n_1; i < n_1 + n_2; i++) {
    radii[i] = r_2 + r_std_2 * std::normal_distribution<double>(0, 1)(gen);
  }
  return radii;
}

class BerthierDistribution3d {
 public:
  /**
   * Distribution is given by
   * p(x) = norm/(x^3)  where x in [dmin, dmax]
   *
   * Parameters
   * ----------
   * delta : float
   *     stdev(diameter) / mean(diameter)
   * dmin_by_dmax : float
   *     dmin / dmax
   * d_av : float
   *     mean(diameter)
   * seed : int
   */
  BerthierDistribution3d(double dmin_by_dmax, double d_mean, int seed = 0)
      : norm(0.5 * d_mean * d_mean * (1 + dmin_by_dmax) / (1 - dmin_by_dmax)),
        dmin(d_mean * 0.5 * (1 + dmin_by_dmax)),
        dmax(dmin / dmin_by_dmax),
        d_mean(d_mean),
        uniform(0., 1.) {}

  inline double _inverse_cdf(double x) {
    /**
     * Inverse of the cdf of the distribution.
     * used to sample, see
     * https://en.wikipedia.org/wiki/Inverse_transform_sampling
     *
     * converts a sample from a uniform random distribution in [0, 1] to a
     * sample from the berthier distribution
     */
    if (x < 0 || x > 1) {
      throw std::invalid_argument("x must be in [0,1]");
    }
    // comes from solving the integral of the pdf
    double inv_x_2 = 1 / (dmin * dmin) - 2 * x / norm;
    return 1 / sqrt(inv_x_2);
  }

  Array<double> sample(size_t n) {
    Array<double> sample(n);
    for (size_t i = 0; i < n; ++i) {
      sample[i] = _inverse_cdf(uniform(generator));
    }
    return sample;
  }

  double pdf(double x) {
    if (x < dmin || x > dmax) {
      return 0;
    }
    return norm / (x * x * x);
  }

 private:
  double norm;
  double dmin;
  double dmax;
  double d_mean;
  std::default_random_engine generator;
  std::uniform_real_distribution<double> uniform;
};

template <typename T>
vector<size_t> sort_indexes(const vector<T> &v) {
  // initialize original index locations
  vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values
  stable_sort(idx.begin(), idx.end(),
              [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

  return idx;
}

/* @brief Utility function for printing a matrix
 */

inline void print_matrix(matrix m) {
  for (size_t i = 0; i < m.size(); i++) {
    for (size_t j = 0; j < m[i].size(); j++) {
      std::cout << m[i][j] << " ";
    }
    std::cout << std::endl;
  }
}

template <class T>
inline void print_vector(std::vector<T> v) {
  for (size_t i = 0; i < v.size(); i++) {
    std::cout << v[i] << " ";
  }
  std::cout << std::endl;
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
inline double get_box_length(Array<double> hs_radii, int dim, double phi) {
  if (dim == 3) {
    double vol_spheres = 4. / 3. * PI * (hs_radii * hs_radii * hs_radii).sum();
    return pow(vol_spheres / phi, 1. / 3.);
  } else if (dim == 2) {
    double vol_discs = PI * (hs_radii * hs_radii).sum();
    return pow(vol_discs / phi, 1. / 2.);
  } else {
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
 *             seed:       seed for random number generator. Default is -1
 *                         if -1, then the seed is generated from random_device
 *
 *
 * @return     Array of random coordinates
 */
inline Array<double> generate_random_coordinates(double box_length,
                                                 int n_particles, int dim,
                                                 int seed = -1) {
  Array<double> coords(n_particles * dim);

  std::random_device rd;
  if (seed == -1) {
    seed = rd();
  }
  std::mt19937 gen(seed);
  std::uniform_real_distribution<> dist(0, box_length);

  for (int i = 0; i < n_particles; i++) {
    for (int j = 0; j < dim; j++) {
      coords[i * dim + j] = dist(gen);
    }
  }
  return coords;
}

inline std::vector<double> cross_shifted(matrix x, std::vector<size_t> shift_x,
                                         std::vector<size_t> shift_y) {
  std::vector<double> cross_product(x.size());
  for (size_t i = 0; i < x.size(); i++) {
    cross_product[i] = x[shift_x[i]][0] * x[shift_y[i]][1] -
                       x[shift_x[i]][1] * x[shift_y[i]][0];
  }
  return cross_product;
}

// terribly inefficient by memory allocation, but it works
inline std::vector<size_t> sort_circle(matrix vertices) {
  // Split the vertices based on whether the x coordinate is to the right or
  // left of the origin
  std::vector<size_t> left_global_indices;
  std::vector<size_t> right_global_indices;
  for (size_t i = 0; i < vertices.size(); i++) {
    if (vertices[i][0] < 0) {
      left_global_indices.push_back(i);
    } else {
      right_global_indices.push_back(i);
    }
  }
  // get the
  std::vector<double> left_tan_list;
  std::vector<double> right_tan_list;

  for (size_t i = 0; i < left_global_indices.size(); i++) {
    left_tan_list.push_back(vertices[left_global_indices[i]][1] /
                            vertices[left_global_indices[i]][0]);
  }
  for (size_t i = 0; i < right_global_indices.size(); i++) {
    right_tan_list.push_back(vertices[right_global_indices[i]][1] /
                             vertices[right_global_indices[i]][0]);
  }

  // indices in only left and right ar
  std::vector<size_t> left_local_indices = sort_indexes(left_tan_list);
  std::vector<size_t> right_local_indices = sort_indexes(right_tan_list);

  // rearrange global indices by their respective local indices
  std::vector<size_t> left_global_indices_sorted;
  std::vector<size_t> right_global_indices_sorted;
  for (size_t i = 0; i < left_local_indices.size(); i++) {
    left_global_indices_sorted.push_back(
        left_global_indices[left_local_indices[i]]);
  }
  for (size_t i = 0; i < right_local_indices.size(); i++) {
    right_global_indices_sorted.push_back(
        right_global_indices[right_local_indices[i]]);
  }

  // concanetate the two lists and return
  std::vector<size_t> sorted_indices_list;
  sorted_indices_list.insert(sorted_indices_list.end(),
                             left_global_indices_sorted.begin(),
                             left_global_indices_sorted.end());
  sorted_indices_list.insert(sorted_indices_list.end(),
                             right_global_indices_sorted.begin(),
                             right_global_indices_sorted.end());
  return sorted_indices_list;
}

inline bool origin_in_hull_2d(matrix vertices) {
  std::vector<size_t> sort_indices;
  sort_indices = sort_circle(vertices);

  // shift the vertices so the first vertex is pushed last
  std::vector<size_t> shifted_indices;
  shifted_indices.insert(shifted_indices.end(), sort_indices.begin() + 1,
                         sort_indices.end());
  // push the first vertex to the end
  shifted_indices.push_back(sort_indices[0]);

  // get the cross product of shifted vertices and the normal vertices
  // Doing it this way considering there's no easy way to translate julia's
  // masks to c++
  std::vector<double> cross_product =
      cross_shifted(vertices, sort_indices, shifted_indices);
  for (size_t i = 0; i < cross_product.size(); i++) {
    if (cross_product[i] < 0) {
      return true;
    }
  }
  return false;
}

}  // namespace pele
#endif  // !1
