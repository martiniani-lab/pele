// Newton method that ignores singular while calculating the inverse
// of a matrix.

#ifndef _PELE_NEWTON_H_
#define _PELE_NEWTON_H_

#include "array.hpp"
#include "base_potential.hpp"
#include "eigen_interface.hpp"
#include "pele/backtracking.hpp"
#include "pele/optimizer.hpp"
#include <Eigen/Dense>

namespace pele {

class Newton : public GradientOptimizer {
private:
  Eigen::MatrixXd _hessian;
  Eigen::VectorXd _gradient;
  Eigen::VectorXd _x;
  Eigen::VectorXd _x_old;
  Eigen::VectorXd _gradient_old;
  Eigen::VectorXd _step;
  BacktrackingLineSearch _line_search;
  double _energy;
  double _tolerance;
  double _threshold;
  int _nhev;

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
  Newton(std::shared_ptr<BasePotential> potential,
         const pele::Array<double> &x0, double tol = 1e-6,
         double threshold = std::numeric_limits<double>::epsilon());

  /**
   * Destructor
   */
  virtual ~Newton() {}

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
  inline int get_nhev() const { return _nhev; }
};

} // namespace pele

#endif // !_PELE_NEWTON_H_
