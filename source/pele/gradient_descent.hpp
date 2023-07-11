#ifndef _PELE_GD_H__
#define _PELE_GD_H__

#include <memory>
#include <pele/vecn.hpp>
#include <vector>

#include "array.hpp"
#include "base_potential.hpp"
#include "optimizer.hpp"

// line search methods
#include "backtracking.hpp"
#include "bracketing.hpp"
#include "linesearch.hpp"
#include "more_thuente.hpp"
#include "nwpele.hpp"
#include "optimizer.hpp"

namespace pele {

/**
 * An implementation of the standard gradient descent optimization algorithm in
 * c++. The learning rate (step size) is set via initial_step. defaults to
 * backtracking line search
 * This optimizer can also be viewed as a Forward Euler Integrator with the
 * backtracking line search as a time step controller. Hence this derives from
 * ODEBasedOptimizer which allows for storing the trajectory as a function of
 * the time in the ODE.
 */
class GradientDescent : public ODEBasedOptimizer {
 private:
  Array<double> xold;  //!< Coordinates before taking a step
  Array<double> step;  //!< Step direction
  double
      inv_sqrt_size;  //!< The inverse square root the the number of components
  BacktrackingLineSearch line_search_method;  // Line search method
  double step_size_;                          //!< Step size
 public:
  /**
   * Constructor
   */
  GradientDescent(std::shared_ptr<pele::BasePotential> potential,
                  const pele::Array<double> x0, double tol = 1e-4,
                  double step_size = 1e-4, bool save_trajectory = true,
                  int iterations_before_save = 1)
      : ODEBasedOptimizer(potential, x0, tol, save_trajectory,
                          iterations_before_save),
        xold(x_.size()),
        step(x_.size()),
        line_search_method(this),
        step_size_(step_size) {
    // set precision of printing
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
    inv_sqrt_size = 1 / sqrt(x_.size());
  }

  /**
   * Destructor
   */
  virtual ~GradientDescent() {}

  /**
   * Do one iteration iteration of the optimization algorithm
   */
  void one_iteration() {
    if (!func_initialized_) {
      initialize_func_gradient();
    }

    // make a copy of the position and gradient
    xold.assign(x_);
    //  really need to refactor and clean this up
    Array<double> gold = g_.copy();
    // Gradient defines step direction
    step = -g_.copy() * step_size_;

    // reduce the stepsize if necessary
    line_search_method.set_xold_gold_(xold, gold);
    line_search_method.set_g_f_ptr(g_);
    step_norm_ = line_search_method.line_search(x_, step);
    Array<double> gdiff = gold - g_;
    Array<double> xdiff = xold - x_;
    // Forward Euler time
    // dx = -dt * g => dt = |dx| / |g|
    time_ += step_norm_ / norm(gold);
    // print some status information
    if ((iprint_ > 0) && (iter_number_ % iprint_ == 0)) {
      std::cout << "steepest descent: " << iter_number_ << " E " << f_
                << " rms " << gradient_norm_ << " nfev " << nfev_
                << " step norm " << step_norm_ << std::endl;
    }
    iter_number_ += 1;
  }
  /**
   * reset the optimizer to start a new minimization from x0
   *
   */
  virtual void reset(pele::Array<double> &x0) {
    if (x0.size() != x_.size()) {
      throw std::invalid_argument(
          "The number of degrees of freedom (x0.size()) cannot change when "
          "calling reset()");
    }
    iter_number_ = 0;
    nfev_ = 0;
    x_.assign(x0);
    initialize_func_gradient();
  }
};
}  // namespace pele

#endif
