#include "pele/mxopt.h"
#include "Eigen/src/Core/util/Constants.h"
#include "cvode/cvode.h"
#include "pele/array.h"
#include "pele/base_potential.h"
#include "pele/debug.h"
#include "pele/eigen_interface.h"
#include "pele/lbfgs.h"
#include "pele/lowest_eig_potential.h"
#include "pele/lsparameters.h"
#include "pele/ngt.hpp"
#include "pele/optimizer.h"
#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <memory>
#include <ostream>
#include <sunlinsol/sunlinsol_dense.h> // access to dense SUNLinearSolver
#include <sunlinsol/sunlinsol_spgmr.h> /* access to SPGMR SUNLinearSolver */
#include <sunnonlinsol/sunnonlinsol_newton.h>

using namespace Spectra;

namespace pele {

MixedOptimizer::MixedOptimizer(std::shared_ptr<pele::BasePotential> potential,
                               const pele::Array<double> x0, double tol, int T,
                               double step, double conv_tol, double conv_factor,
                               double rtol, double atol, bool iterative)
    : GradientOptimizer(potential, x0, tol), cvode_mem(CVodeCreate(CV_BDF)),
      N_size(x_.size()), t0(0), tN(100.0), rtol(rtol), atol(atol),
      xold(x_.size()), gold(x_.size()), step(x_.size()), T_(T), usephase1(true),
      conv_tol_(conv_tol), conv_factor_(conv_factor), n_phase_1_steps(0),
      n_phase_2_steps(0),hessian(x_.size(), x_.size()), hessian_shifted(x_.size(), x_.size()),  line_search_method(this, step) {

  // set precision of printing
  std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
  // dummy t0
  double t0 = 0;
  sparse_le_solver_type_ = 1;


  if (sparse_le_solver_type_ == 0) {
    lowest_eig_pot_ = std::shared_ptr<pele::LowestEigPotential>(
        new pele::LowestEigPotential(potential, x0, potential->get_ndim()));
    lbfgs_lowest_eigval = std::shared_ptr<pele::LBFGS>(
        new pele::LBFGS(lowest_eig_pot_, x0, 1e-6, 10));
  } else if (sparse_le_solver_type_ == 1) {
    // Sparse eigen constructor
  }

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
  if (iterative) {
    LS = SUNLinSol_SPGMR(x0_N, PREC_NONE, 0);
    CVodeSetLinearSolver(cvode_mem, LS, NULL);
  } else {
    A = SUNDenseMatrix(N_size, N_size);
    LS = SUNLinSol_Dense(x0_N, A);
    CVodeSetLinearSolver(cvode_mem, LS, A);
    CVodeSetJacFn(cvode_mem, Jac);
  }

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
  gold.assign(g_);
  // copy the gradient into step
  step.assign(g_);
  // does a convexity check every T iterations
  // but always starts off in differential equation solver mode
  if (iter_number_ % T_ == 0 and iter_number_ > 0) {
#if OPTIMIZER_DEBUG_LEVEL >= 3
    std::cout << "checking convexity"
              << "\n";
#endif
    usephase1 = convexity_check();
  }
  if (usephase1) {
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
    // switchtophase2 = true;
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



  get_hess(hessian);

  hessian_shifted = hessian;
  // shift diagonal by conv tol

  hessian_shifted.diagonal().array() -= conv_tol_;

  cout << "hessian_shifted" << hessian_shifted.eigenvalues() << std::endl;

  // check if the hessian is positive definite by doing a cholesky decomposition
  // and checking if it is successful
  std::cout << "checking if hessian is positive definite" << "\n";

  Eigen::ComputationInfo info = hessian_shifted.llt().info();
  std::cout << info << " success info of cholesky decomposition \n";

  
  if (info == Eigen::Success) {
    // if it is positive definite, we can use the newton method to solve the
    // problem
    return true;
  }
  else {
  return false;
  }
}

/**
 * Gets the hessian. involves a dense hessian for now. #TODO replace with a
 * sparse hessian. TODO: Allocate memory once
 */
void MixedOptimizer::get_hess(Eigen::MatrixXd & hessian) {
  Array<double> hessian_pele = Array<double>(hessian.data(), hessian.size());
  potential_->get_hessian(
      x_, hessian_pele); // preferably switch this to sparse Eigen
  udata.nhev += 1;
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
  n_phase_1_steps += 1;
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
  // std::cout << hessian.eigenvalues() << "hessian eigenvalues before \n";
  hessian.diagonal().array() += conv_factor_ * conv_tol_ *10;

  // print conv factor added
  std::cout << "conv factor " << conv_factor_ << " conv tol " << conv_tol_ << std::endl;
  std::cout << "conv factor added: " << conv_factor_ * conv_tol_ *10 << "\n";

  Eigen::VectorXd r(step.size());
  Eigen::VectorXd q(step.size());
  q.setZero();
  eig_eq_pele(r, step);
  // negative sign to switch direction
  // TODO change this to banded
  std::cout << "r " << r << "\n";
  std::cout << "hessian " << hessian << "\n";
  std::cout << "hessian eigenvalues before \n";
  std::cout << hessian.eigenvalues() << "\n";
  q = -scale * hessian.householderQr().solve(r);
  pele_eq_eig(step, q);

  std::cout << "step norm " << q.norm() << "\n";
  n_phase_2_steps += 1;
}
} // namespace pele
