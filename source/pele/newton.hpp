// Newton method that ignores singular while calculating the inverse
// of a matrix. Modified to remove rattlers.

#ifndef _PELE_NEWTON_H_
#define _PELE_NEWTON_H_

#include "array.hpp"
#include "base_potential.hpp"
#include "eigen_interface.hpp"
#include "pele/backtracking.hpp"
#include "pele/optimizer.hpp"
#include <Eigen/Dense>
#include <array>
#include <cstddef>
#include <cstdint>

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
  bool _use_rattler_mask;
  Array<uint8_t> _not_rattlers;
  bool _jammed;
  bool _rattlers_found;

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
         double threshold = std::numeric_limits<double>::epsilon(),
         bool use_rattler_mask = false);

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
  inline Eigen::VectorXd get_step() const { return _step; }
  void reset(Array<double> x);
  void get_rattler_details(Array<uint8_t> &not_rattlers, bool &jammed) {
    if (!_rattlers_found) {
      throw std::runtime_error(" Rattler based method not called");
    }
    not_rattlers = not_rattlers;
    jammed = _jammed;
  }
  // Warning: does not pass info about whether we're near a minimum or not.
  bool is_jammed() {if(_rattlers_found) { return _jammed;} else {return false;}}

  void compute_func_gradient(Array<double> x, double &func,
                             Array<double> gradient);
};

} // namespace pele

#endif // !_PELE_NEWTON_H_
