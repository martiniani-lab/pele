#ifndef _PELE_COS_GD_H__
#define _PELE_COS_GD_H__

#include <memory>
#include <pele/vecn.hpp>
#include <vector>

#include "array.hpp"
#include "base_potential.hpp"
#include "optimizer.hpp"

// line search methods
#include "optimizer.hpp"

namespace pele {

/**
 * An implementation of the standard gradient descent optimization algorithm
 * with a step size adaptation based on the cosine similarity between successive
 * gradients. This is described in
 * https://journals.aps.org/pre/pdf/10.1103/PhysRevE.108.054102. In practice,
 * you may be better off using the standard gradient descent implementation in
 * gradient_descent.cpp for optimization or CVODE if you're interested in
 inherent structures.
 * This diverges from the description in the sense that instead of using the
 * initial stepsize based on the largest unbalanced force, init step size is set
 outside. This implementation was written purely for comparison purposes.
 */
class CosineGradientDescent : public ODEBasedOptimizer {
 private:
  Array<double> xold;   //!< Coordinates before taking a step
  Array<double> gold;   //!< Gradient before taking a step
  Array<double> step;   //!< Step direction
  double cos_sim_tol_;  //!< Cosine similarity tolerance
  double dt_;           //!< Time step
  int back_track_n_;  //! if 1 - cosine similarity is greater than cos_sim_tol_
                      //! keep dividing step by n until we end up with a step
                      //! less than the tolerance
                      //  if step keeps satisfying the condition for
                      //  back_track_n_ times,
                      // i.e we're not making progress, multiply the step by
                      // back_track_n_
  const int max_back_track = 100;  //! maximum number of times to back track
  int n_success_without_backtrack = 0;

 public:
  /**
   * Constructor
   */
  CosineGradientDescent(std::shared_ptr<pele::BasePotential> potential,
                        const pele::Array<double> x0, double tol = 1e-4,
                        double cos_sim_tol = 1e-2, double dt_initial = 1e-5,
                        int back_track_n = 5, bool save_trajectory = true,
                        int iterations_before_save = 1)
      : ODEBasedOptimizer(potential, x0, tol, save_trajectory,
                          iterations_before_save),
        xold(x_.size(), 0),
        gold(x_.size(), 0),
        step(x_.size(), 0),
        cos_sim_tol_(cos_sim_tol),
        dt_(dt_initial),
        back_track_n_(back_track_n) {
    std::cout << "cosine gradient descent initialized" << std::endl;
    std::cout << "with parameters: " << std::endl;
    std::cout << "tol: " << tol << std::endl;
    std::cout << "cos_sim_tol: " << cos_sim_tol << std::endl;
    std::cout << "dt_initial: " << dt_initial << std::endl;
    std::cout << "back_track_n: " << back_track_n << std::endl;
    std::cout << "save_trajectory: " << save_trajectory << std::endl;
    std::cout << "iterations_before_save: " << iterations_before_save
              << std::endl;
    if (back_track_n_ < 1) {
      throw std::invalid_argument(
          "back_track_n must be greater than or equal to 1");
    }
  }

  /**
   * Destructor
   */
  virtual ~CosineGradientDescent() {}

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
    gold.assign(g_);
    // Gradient defines step direction
#if OPTIMIZER_DEBUG_LEVEL > 2
    std::cout << "cosine gradient descent: " << iter_number_ << " E " << f_
              << " rms " << gradient_norm_ << " nfev " << nfev_ << std::endl;
#endif
#if OPTIMIZER_DEBUG_LEVEL > 3
    std::cout << "x" << x_ << std::endl;
    std::cout << g_ << std::endl;
#endif
    // if the cosine similarity is less than the tolerance, reset the step size

    for (int i = 0; i < max_back_track; i++) {
      x_.assign(xold - dt_ * g_);
      f_ = potential_->get_energy_gradient(x_, g_);
      nfev_ += 1;
      double cos_sim_eps = abs(1 - dot(gold, g_) / (norm(gold) * norm(g_)));
#if OPTIMIZER_DEBUG_LEVEL > 2
      std::cout << "cosine similarity: " << cos_sim_eps << std::endl;
#endif
      if (cos_sim_eps < cos_sim_tol_) {
        if (i == 0) {
          n_success_without_backtrack += 1;
        }
        break;
      }
      // backtracking failed
      n_success_without_backtrack = 0;
      dt_ = dt_ / back_track_n_;
    }
    time_ += dt_;
    step.assign(x_ - xold);
    if (n_success_without_backtrack == back_track_n_) {
      dt_ = dt_ * back_track_n_;
      n_success_without_backtrack = 0;
    }
    step_norm_ = norm(step);
    gradient_norm_ = norm(g_);
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
