#include "pele/mxopt.hpp"
#include "Eigen/src/Core/Matrix.h"
#include "cvode/cvode.h"
#include "pele/array.hpp"
#include "pele/base_potential.hpp"
#include "pele/debug.hpp"
#include "pele/eigen_interface.hpp"
#include "pele/lsparameters.hpp"
#include "pele/optimizer.hpp"
#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <memory>
#include <sunlinsol/sunlinsol_dense.h> // access to dense SUNLinearSolver
#include <sunnonlinsol/sunnonlinsol_newton.h>

namespace pele {

MixedOptimizer::MixedOptimizer(std::shared_ptr<pele::BasePotential> potential,
                               const pele::Array<double> x0, double tol, int T,
                               double step, double conv_tol, double conv_factor,
                               double rtol, double atol)
    : GradientOptimizer(potential, x0, tol),
      cvode_mem(CVodeCreate(CV_BDF)), 
      N_size(x_.size()), t0(0), tN(100.0), rtol(rtol), atol(atol),
      xold(x_.size()), gold(x_.size()), step(x_.size()), T_(T),
      usephase1(true), conv_tol_(conv_tol), conv_factor_(conv_factor),
      line_search_method(this, step) {
    // set precision of printing
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
    // dummy t0
    double t0 = 0;
    std::cout << x0 << "\n";
    Array<double> x0copy = x0.copy();
    x0_N = N_Vector_eq_pele(x0copy);
    // initialization of everything CVODE needs
    int ret = CVodeInit(cvode_mem, f, t0, x0_N);
    // initialize userdata
    udata.rtol = rtol;
    udata.atol = atol;
    udata.nfev = 0;
    udata.nhev = 0;
    udata.pot_ = potential_;
    udata.stored_grad = Array<double>(x0.size(), 0);
    // set tolerances
    CVodeSStolerances(cvode_mem, udata.rtol, udata.atol);
    ret = CVodeSetUserData(cvode_mem, &udata);
    // initialize hessian
    A = SUNDenseMatrix(N_size, N_size);
    LS = SUNLinSol_Dense(x0_N, A);
    CVodeSetLinearSolver(cvode_mem, LS, A);
    CVodeSetJacFn(cvode_mem, Jac);
    // pass hessian information
    g_ = udata.stored_grad;
    // initialize CVODE steps and stop time
    CVodeSetMaxNumSteps(cvode_mem, 100000);
    CVodeSetStopTime(cvode_mem, tN);
    inv_sqrt_size = 1 / sqrt(x_.size());
    // optmizeer debug level
    std::cout << OPTIMIZER_DEBUG_LEVEL << "optimizer debug level \n";

#if OPTIMIZER_DEBUG_LEVEL >= 1
    std::cout << "Mixed optimizer constructed"
              << "T=" << T_ << "\n";
#endif
}
/**
 * Does one iteration of the optimization algorithm
 */
void MixedOptimizer::one_iteration() {
  if (!func_initialized_) {
    initialize_func_gradient();
  }

  // declare that the hessian hasn't been calculated for this iteration
  hessian_calculated = false;
  // make a copy of the position and gradient
  xold.assign(x_);


#if OPTIMIZER_DEBUG_LEVEL >= 3
  std::cout << xold << "starting position \n";
#endif  
  
  gold.assign(g_);
  // copy the gradient into step
  step.assign(g_);
  bool switchtophase2 = false;
  // does a convexity check every T iterations
  // but always starts off in differential equation solver mode
  if (iter_number_ % T_ == 0 and iter_number_>0) {
#if OPTIMIZER_DEBUG_LEVEL >= 3
      std::cout << "checking convexity"
                << "\n";
#endif
      usephase1 = convexity_check();
  }
  if (usephase1 and not switchtophase2) {
#if OPTIMIZER_DEBUG_LEVEL >= 3
    std::cout << " computing phase 1 step"
              << "\n";
#endif

    compute_phase_1_step(step);
  } else {

#if OPTIMIZER_DEBUG_LEVEL >= 3
    std::cout << " computing phase 2 step"
              << "\n";
#endif
    compute_phase_2_step(step);
    line_search_method.set_xold_gold_(xold, gold);
    line_search_method.set_g_f_ptr(g_);
    double stepnorm = line_search_method.line_search(x_, step);
    switchtophase2 = true;
  }

  // should think of using line search method within the phase steps

  // update inverse hessian estimate

#if OPTIMIZER_DEBUG_LEVEL >= 3
  std::cout << "mixed optimizer: " << iter_number_ << " E " << f_ << " rms "
            << rms_ << " nfev " << nfev_ << std::endl;
#endif
  /**
   * Checks whether the stop criterion is satisfied: if stop criterion is
   * satisfied The new version is done. at this point the corresponding values
   * are added
   */

  iter_number_ += 1;

}



/**
 * resets the minimizer for usage again
 */
void MixedOptimizer::reset(pele::Array<double> &x0) {
  if (x0.size() != x_.size()) {
    throw std::invalid_argument("The number of degrees of freedom (x0.size()) "
                                "cannot change when calling reset()");
  }
  iter_number_ = 0;
  nfev_ = 0;
  x_.assign(x0);
  initialize_func_gradient();
}

/**
 * checks convexity in the region and updates the convexity flag accordingly
 * convexity flag 0/False -> Not near the minimum: Use an differential equation
 * solver (CVODE_BDF in this example). convexity flag true -> concavity below a
 * tolerance, we think we're near a minimum. can be solved with newton steps
with help convexity flag false ->
 * convex function, We think we're near a minimum, use a newton method to figure
out the result.
 */

bool MixedOptimizer::convexity_check() {

  hessian = get_hessian();
  hessian_calculated = true; // pass on the fact that the hessian has been calculated
  Eigen::VectorXd eigvals = hessian.eigenvalues().real();
  minimum = eigvals.minCoeff();
  double maximum = eigvals.maxCoeff();
  if (maximum == 0) {
      maximum = 1e-8;
  }
  double convexity_estimate = std::abs(minimum / maximum);

  if (minimum < 0 and convexity_estimate >= conv_tol_) {
    // note minimum is negative
#if OPTIMIZER_DEBUG_LEVEL >= 1
    std::cout << "minimum less than 0"
              << " convexity tolerance condition not satisfied \n";
#endif
    minimum_less_than_zero = true;
    return true;
  } else if (convexity_estimate < conv_tol_ and minimum < 0) {

#if OPTIMIZER_DEBUG_LEVEL >= 1
    std::cout << "minimum less than 0"
              << " convexity tolerance condition satisfied \n";
#endif
    scale = 1.;
    minimum_less_than_zero = true;
    return false;
  }

  else {
    scale = 1;
#if OPTIMIZER_DEBUG_LEVEL >= 1
    std::cout << "minimum greater than 0"
              << " convexity tolerance condition satisfied \n";
#endif
    minimum_less_than_zero = false;
    return false;
  }
}

/**
 * Gets the hessian. involves a dense hessian for now. #TODO replace with a
 * sparse hessian. TODO: Allocate memory once
 */
Eigen::MatrixXd MixedOptimizer::get_hessian() {
  Array<double> hess(xold.size() * xold.size());
  Array<double> grad(xold.size());
  

  double e = potential_->get_energy_gradient_hessian(
      x_, grad, hess); // preferably switch this to sparse Eigen

  Eigen::MatrixXd hess_dense(xold.size(), xold.size());
  udata.nhev += 1;
  hess_dense.setZero();
  for (size_t i = 0; i < xold.size(); ++i) {
    for (size_t j = 0; j < xold.size(); ++j) {
      hess_dense(i, j) = hess[i + grad.size() * j];
    }
  }
  return hess_dense;
 }

// /**
//  * Phase 1 The problem does not look convex, Try solving using with an
//  adaptive differential equationish approach
//  */
// void MixedOptimizer::compute_phase_1_step(Array<double> step) {
//     // use a a scaled steepest descent step
//     step *= -std::abs(H0_);
// }

/**
 * Phase 1 The problem does not look convex, Try solving using with sundials
 */
void MixedOptimizer::compute_phase_1_step(Array<double> step) {
  /* advance solver just one internal step */
  Array<double> xold = x_;
  // use this variable to compute differences and add to nhev later
  double udatadiff = udata.nfev;
  int flag = CVode(cvode_mem, tN, x0_N, &t0, CV_ONE_STEP);
  udatadiff = udata.nfev - udatadiff;
  nfev_ += udatadiff;
  x_ = pele_eq_N_Vector(x0_N);
  g_ = udata.stored_grad;
  rms_ = (norm(g_) / sqrt(x_.size()));
  f_ = udata.stored_energy;
  step = xold - x_;
}

/**
 * Phase 2 The problem looks convex enough to switch to a newton method
 */
void MixedOptimizer::compute_phase_2_step(Array<double> step) {
  if (hessian_calculated == false) {
    // we can afford to perform convexity checks at every steps
    // assuming the rate limiting cost is the hessian which is calculated
    // in the convexity check anyway
    convexity_check();
  }
  // this can mess up accuracy if we aren't close to a minimum preferably switch
  // sparse
  // std::cout << hessian.eigenvalues() << "hessian eigenvalues before \n";
  printf(minimum_less_than_zero ? "true" : "false");
  std::cout  << "\n";
  
  std::cout << hessian.eigenvalues() << " eigenvaluess   \n";
  std::cout << minimum << "minimum value\n";
  
  
  
  if (minimum_less_than_zero) {
      hessian -= conv_factor_ * minimum *
          Eigen::MatrixXd::Identity(x_.size(), x_.size());
  }


  Eigen::VectorXd r(step.size());
  Eigen::VectorXd q(step.size());
  q.setZero();
  eig_eq_pele(r, step);
  // negative sign to switch direction
  // TODO change this to banded
  q = -scale * hessian.fullPivLu().solve(r);
  pele_eq_eig(step, q);
  
#if OPTIMIZER_DEBUG_LEVEL >= 3
  // debug information for gradients  
      std::cout << q <<" step \n";
      std::cout << r << "gradient \n";
      std::cout << hessian.eigenvalues().real() << "hessian eigenvalues \n";
      Eigen::FullPivLU<Eigen::MatrixXd> lu(hessian);
      std::cout << lu.kernel() << "\n";
#endif    
}
} // namespace pele
