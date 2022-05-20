/**
 * @file translation_symmetry_matrices.hpp
 * @author Praharsh Suryadevara
 * @brief File contains functions that generate translation symmetries and
 * matrices.
 * @version 0.1
 * @date 2022-05-20
 *
 *
 */

#ifndef TRANSLATION_SYMMETRY_MATRICES_HPP
#define TRANSLATION_SYMMETRY_MATRICES_HPP

#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <cstddef>
#include <stdexcept>

namespace pele {

void generate_2d_symmetries(size_t n_particles, Eigen::MatrixXd &projector_x,
                            Eigen::MatrixXd &projector_y) {

  projector_x.setZero();
  projector_y.setZero();

  if (projector_x.rows() != 2 * n_particles) {
    throw std::logic_error("projector_x.rows() != 2 * n_particles");
  }
  if (projector_y.rows() != 2 * n_particles) {
    throw std::logic_error("projector_y.rows() != 2 * n_particles");
  }

  for (size_t i = 0; i < n_particles * 2; i++) {
    for (size_t j = i + 1; j < n_particles * 2; j++) {
      if (i % 2 == 0 && j % 2 == 0) {
        projector_x(i, j) = 1;
        projector_x(j, i) = 1;
      }
      if (i % 2 == 1 && j % 2 == 1) {
        projector_y(i, j) = 1;
        projector_y(j, i) = 1;
      }
    }
  }
}

void generate_3d_symmetries(size_t n_particles, Eigen::MatrixXd &projector_x,
                            Eigen::MatrixXd &projector_y,
                            Eigen::MatrixXd &projector_z) {

  projector_x.setZero();
  projector_y.setZero();
  projector_z.setZero();

  if (projector_x.rows() != 3 * n_particles) {
    throw std::logic_error("projector_x.rows() != 3 * n_particles");
  }
  if (projector_y.rows() != 3 * n_particles) {
    throw std::logic_error("projector_y.rows() != 3 * n_particles");
  }
  if (projector_z.rows() != 3 * n_particles) {
    throw std::logic_error("projector_z.rows() != 3 * n_particles");
  }

  for (size_t i = 0; i < n_particles * 3; i++) {
    for (size_t j = i + 1; j < n_particles * 3; j++) {
      if (i % 3 == 0 && j % 3 == 0) {
        projector_x(i, j) = 1;
        projector_x(j, i) = 1;
      }
      if (i % 3 == 1 && j % 3 == 1) {
        projector_y(i, j) = 1;
        projector_y(j, i) = 1;
      }
      if (i % 3 == 2 && j % 3 == 2) {
        projector_z(i, j) = 1;
        projector_z(j, i) = 1;
      }
    }
  }
}

}
#endif // TRANSLATION_SYMMETRY_MATRICES_HPP
