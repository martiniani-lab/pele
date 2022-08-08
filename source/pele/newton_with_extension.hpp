// Newton Method that adds a potential extension to the gradient, and a
// translation offset to ensure that the hessian is positive definite, and the
// steps are well scaled.

#ifndef _PELE_NEWTON_WITH_EXTENSION_H_
#define _PELE_NEWTON_WITH_EXTENSION_H_

#include "array.hpp"
#include "base_potential.hpp"
#include "eigen_interface.hpp"
#include "pele/backtracking.hpp"
#include "pele/optimizer.hpp"
#include <Eigen/Dense>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <pele/combine_potentials.hpp>

namespace pele {

class NewtonWithExtendedPotential : public GradientOptimizer {
private:
  Eigen::MatrixXd _hessian;
  Eigen::VectorXd _gradient;
  Eigen::VectorXd _x;
  Eigen::VectorXd _x_old;
  Eigen::VectorXd _gradient_old;
  Eigen::VectorXd _step;
  BacktrackingLineSearch _line_search;
  double _energy;
  double _translation_offset;
  double _max_step;
  /**
   * @brief Extended Potential, helps regularize the hessian eigenvalues
   *        to be positive with rattlers
   */
  std::shared_ptr<BasePotential> _potential_extension;
  std::shared_ptr<ExtendedPotential> _extended_potential_wrapper;

  // Checks whether the hessian was calculated for the current x
  // Gets reset at the end of each iteration when a new step is taken
  // Sets the hessian
  bool hessian_calculated;

public:
  /**
   * @brief Newton method that ignores singular while calculating the inverse.
   * Uses a backtracking linesearch. If the hessian has singular values uses the
   * minimum norm solution for solving the inverse.
   *
   * @param potential potential to minimize
   * @param x0 initial guess
   * @param tol stopping tolerance
   * @param threshold threshold for the singular values of the hessian
   * @param max_iter maximum number of iterations
   */
  NewtonWithExtendedPotential(
      std::shared_ptr<BasePotential> potential, const pele::Array<double> &x0,
      double tol, std::shared_ptr<BasePotential> potential_extension,
      double translation_offset = 1.0, double max_step = 1.0);

  /**
   * Destructor
   */
  virtual ~NewtonWithExtendedPotential() {}

  /**
   * @brief Do one iteration of the Newton method
   *
   */
  void one_iteration();

  /**
   * @brief Reset the optimizer to start a new minimization from x0
   */
  // TODO: implement
  // void reset(pele::Array<double> x0);
  inline int get_nhev() const { return nhev_; }
  inline Eigen::VectorXd get_step() const { return _step; }
  void reset(Array<double> x);

  void set_hessian(Eigen::MatrixXd hessian) {
    hessian_calculated = true;
    _hessian = hessian;
  }
};

} // namespace pele

#endif // !_PELE_NEWTON_H_
