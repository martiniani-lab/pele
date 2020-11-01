#ifndef _PELE_TO_EIG_H__
#define _PELE_TO_EIG_H__


/**
 * @brief      Interface from pele array to Eigen
 */
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



} // end namespace pele


#endif  // end #ifndef _PELE_TO_EIG_H__
