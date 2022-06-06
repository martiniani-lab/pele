#include "pele/backtracking.hpp"
#include "pele/array.hpp"
#include "pele/base_potential.hpp"
#include "pele/lsparameters.hpp"
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <pele/eigen_interface.hpp>
#include <stdexcept>

using namespace std;
namespace pele {

double BacktrackingLineSearch::line_search(Array<double> &x,
                                           Array<double> step) {

  eig_eq_pele(xoldvec, xold_);
  eig_eq_pele(gradvec, gold_);
  Scalar stpsize = initial_stpsize;
  // absolute original stepnorm
  double absolute_step_norm = norm(step);
  //     if (absolute_step_norm*initial_stpsize>0.3) {
  //         stpsize *= 0.3/absolute_step_norm;
  // #if OPTIMIZER_DEBUG_LEVEL >= 1
  //         std::cout << stpsize << " step size rescaled \n";
  // #endif
  //     }
  eig_eq_pele(step_direction, step);
  Scalar f = opt_->get_f();
  // start with the initial stepsize

  step_moving_away_from_min = false; // only true when step is opposite to gradient
  LSFunc(f, xvec, gradvec, stpsize, step_direction, xoldvec, params);
  if (step_moving_away_from_min) {
    return 0; // Line search will not work
  }
  pele_eq_eig(x, xvec);
  pele_eq_eig(g_, gradvec);
  step = xold_ - x;
  opt_->set_f(f);
  opt_->set_rms(norm(g_) / sqrt(x.size()));
  pele_eq_eig(step, step_direction);
#if OPTIMIZER_DEBUG_LEVEL >= 1
  std::cout << stpsize << " stepsize final \n";
  std::cout << opt_->compute_pot_norm(step) << " step norm \n";
  std::cout << stpsize * opt_->compute_pot_norm(step)
            << "absolute step size \n";
#endif
  double step_norm;
  step_norm = norm(step);
  if (step_norm == 0) {
    throw std::runtime_error("step norm is zero");
  }
  return step_norm;
};

/**
 * wrapper for the function; should define prolly as an extra in
 * GradientOptimizer
 */
double BacktrackingLineSearch::func_grad_wrapper(Vector &x, Vector &grad) {
  pele_eq_eig(xdum, x);
  pele_eq_eig(gdum, grad);
  double f;
  opt_->compute_func_gradient(xdum, f, gdum);
  eig_eq_pele(grad, gdum);
  return f;
}

/**
 * Line Search function
 */
void BacktrackingLineSearch::LSFunc(Scalar &fx, Vector &x, Vector &grad,
                                    Scalar &step, const Vector &drt,
                                    const Vector &xp, const LBFGSParam &param) {

  // Decreasing and increasing factors
  const Scalar dec = 0.5;
  const Scalar inc = 2.1;

  // Check the value of step
  if (step <= Scalar(0))
    std::invalid_argument("'step' must be positive");

  // Save the function value at the current x
  const Scalar fx_init = fx;
  // Projection of gradient on the search direction

  Vector gradinit = grad;
  const Scalar dg_init = grad.dot(drt);
#if OPTIMIZER_DEBUG_LEVEL >= 3
  std::cout << dg_init << "dginit must be less than 0 \n";
#endif
  // Make sure d points to a descent direction
  if (dg_init > 0) {
    step_moving_away_from_min = true; // step points away from minimum
    return;
  }
  const Scalar test_decr = param.ftol * dg_init;
  Scalar width;
  int iter;
  for (iter = 0; iter < param.max_linesearch; iter++) {
    // x_{k+1} = x_k + step * d_k
    x.noalias() = xp + step * drt;

    // Evaluate this candidate
    fx = func_grad_wrapper(x, grad);
    if (fx > fx_init + step * test_decr) {
      width = dec;
    } else {
      break;
    }

    if (iter >= param.max_linesearch)
      throw std::runtime_error(
          "the line search routine reached the maximum number of iterations");

    if (step < param.min_step)
      throw std::runtime_error(
          "the line search step became smaller than the minimum value allowed");

    if (step > param.max_step)
      throw std::runtime_error(
          "the line search step became larger than the maximum value allowed");

    step *= width;
  }
}

} // namespace pele