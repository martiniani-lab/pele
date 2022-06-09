/**
 * @file mixed_descent_with_fire.cpp
 * @author Praharsh Suryadevara (praharsharmm@gmail.com)
 * @brief Mixed Descent with Fire Algorithm. TODO: This should be refactored
 * Into a general mixed descent that can combine any two optimizers. not just
 * CVODE and FIRE. Into a general mixed descent that can combine any two
 * optimizers. not just CVODE and FIRE.
 * @version 0.1
 * @date 2022-06-07
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef PELE_MIXED_DESCENT_WITH_FIRE_HPP
#define PELE_MIXED_DESCENT_WITH_FIRE_HPP

#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <cstddef>
#include <memory>
#include <pele/array.hpp>
#include <pele/base_potential.hpp>
#include <pele/optimizer.hpp>

namespace pele {

/**
 * @brief Generalized Mixed Descent that combines two optimizers.
 * @details Base class for combining two optimizers.
 */
class GenericMixedDescent : public GradientOptimizer {
public:
  GenericMixedDescent(std::shared_ptr<BasePotential> pot, pele::Array<double> x,
                      double tol,
                      std::shared_ptr<pele::GradientOptimizer> opt_non_convex,
                      std::shared_ptr<pele::GradientOptimizer> opt_convex,
                      double translation_offset)
      : opt_non_convex_(opt_non_convex), opt_convex_(opt_convex),
        translation_offset_(translation_offset), hessian_(x.size(), x.size()),
        hessian_copy_for_cholesky_(x.size(), x.size()), hessian_evaluations_(0),
        iterations_(0), GradientOptimizer(pot, x, tol) {
    // Ensure that both optimizers have the same potential
    // There is redundancy because we add the potential to each optimizer
    // but this overrides the potential in the optimizer.
    opt_non_convex_->set_potential(pot);
    opt_convex_->set_potential(pot);
    opt_convex_->set_tolerance(tol);
    opt_non_convex_->set_x(x);
    opt_non_convex_->set_tolerance(tol);
    uplo = 'U';
  }

  void one_iteration();

private:
  std::shared_ptr<pele::GradientOptimizer> opt_non_convex_;
  std::shared_ptr<pele::GradientOptimizer> opt_convex_;

  /**
   * @brief criterion for switching from non-convex to convex optimizer.
   * @details Checks whether the current position of Mixed Descent is in a
   * convex region or not
   *
   */
  bool x_in_convex_region();

  /**
   * @brief Calculate the hessian at the current position.
   * @details Calculate the hessian at the current position.
   *
   * @param hessian the hessian to be calculated
   */
  void get_hess(Eigen::MatrixXd &hessian);

  /**
   * @brief Uses convex method to find the next point if true, else uses
   * non-convex method.
   */
  bool use_convex_method_;

  void add_translation_offset_2d(Eigen::MatrixXd &hessian, double offset);

  size_t hessian_evaluations_;
  size_t iterations_;

  double translation_offset_;
  Eigen::MatrixXd hessian_;

  // Space allocated for a copy of the hessian, this will be
  // rewritten with the Cholesky factorization of the hessian.
  Eigen::MatrixXd hessian_copy_for_cholesky_;

  char uplo;
};
} // namespace pele

#endif // !1
