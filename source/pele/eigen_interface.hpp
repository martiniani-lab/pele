#ifndef _PELE_TO_EIG_H__
#define _PELE_TO_EIG_H__

/**
 * @brief      Interface from pele array to Eigen
 */
// #define EIGEN_USE_MKL_ALL

#include "Eigen/src/Core/Matrix.h"
#include "array.hpp"
#include <Eigen/Dense>

namespace pele {
// implicitly assumes that the arrays of same size
inline void eig_eq_pele(Eigen::VectorXd &mat, Array<double> &x) {
  for (size_t i = 0; i < x.size(); ++i) {
    mat[i] = x[i];
  }
}

inline void pele_eq_eig(Array<double> x, Eigen::VectorXd &mat) {
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = mat[i];
  }
}

inline void eig_mat_eq_pele(Eigen::MatrixXd &mat, Array<double> &x) {
  int n = mat.rows();
  for (size_t i = 0; i < mat.rows(); ++i) {
    for (size_t j = 0; j < mat.cols(); ++j) {
      mat(i, j) = x[n * i + j];
    }
  }
}

// Copies by wrapping data
inline Array<double> pele_eq_eig_data(Eigen::VectorXd &vec) {
  // get raw data from eigen vector
  double *data = vec.data();
  // wrap data into array
  Array<double> x(data, vec.size());
  return x;
}

inline Eigen::VectorXd eig_eq_pele_data(Array<double> &x) {
  // get raw data from array
  double *data = x.data();
  // wrap data into eigen vector
  Eigen::Map<Eigen::VectorXd> vec(data, x.size());
  return vec;
}
} // end namespace pele
#endif // end #ifndef _PELE_TO_EIG_H__