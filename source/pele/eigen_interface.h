#ifndef _PELE_TO_EIG_H__
#define _PELE_TO_EIG_H__


/**
 * @brief      Interface from pele array to Eigen
 */
// #define EIGEN_USE_MKL_ALL
#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE

#include "array.h"
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
            mat(i, j) = x[n*i + j];
        }
    }
}

} // end namespace pele


#endif  // end #ifndef _PELE_TO_EIG_H__
