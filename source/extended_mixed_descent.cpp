
#include "pele/extended_mixed_descent.hpp"
#include "pele/array.hpp"
#include "pele/base_potential.hpp"
#include "pele/eigen_interface.hpp"
#include "pele/lbfgs.hpp"
#include "pele/lsparameters.hpp"
#include "pele/optimizer.hpp"
#include "pele/preprocessor_directives.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <algorithm>
#include <complex>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <sunlinsol/sunlinsol_dense.h> // access to dense SUNLinearSolver
#include <sunlinsol/sunlinsol_spgmr.h> /* access to SPGMR SUNLinearSolver */
#include <sunnonlinsol/sunnonlinsol_newton.h> /* access to Newton SUNNonLinearSolver */
#include <unistd.h>
// whether to print trajectory to file
using namespace Spectra;

namespace pele {

ExtendedMixedOptimizer::ExtendedMixedOptimizer(
    std::shared_ptr<pele::BasePotential> potential,
    std::shared_ptr<pele::BasePotential> potential_extension,
    const pele::Array<double> x0, double tol, int T, double step,
    double conv_tol, double conv_factor, double rtol, double atol,
    bool iterative)
    : GradientOptimizer(potential, x0, tol),
      extended_potential(
          std::make_shared<ExtendedPotential>(potential, potential_extension)),
      cvode_mem(CVodeCreate(CV_BDF)), N_size(x_.size()), t0(0), tN(100.0),
      rtol(rtol), atol(atol), xold(x_.size()), gold(x_.size()), step(x_.size()),
      xold_old(x_.size()), T_(T), use_phase_1(true), conv_tol_(conv_tol),
      conv_factor_(conv_factor), n_phase_1_steps(0), n_phase_2_steps(0),
      hessian(x_.size(), x_.size()), x_last_cvode(x_.size()),
      hessian_copy_for_cholesky(x_.size(), x_.size()),
      line_search_method(this, step) {

  // assume previous phase is phase 1
  prev_phase_is_phase1 = true;
  set_potential(
      extended_potential); // because we can only create the extended potential
  // after we instantiate the extended potential
  // set precision of printing
  // dummy t0
  double t0 = 0;
#if PRINT_TO_FILE == 1
  trajectory_file.open("trajectory.txt");
#endif
  uplo = 'U';
  // set the initial step size
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
void ExtendedMixedOptimizer::one_iteration() {
  if (!func_initialized_) {
    initialize_func_gradient();
  }

  // declare that the hessian hasn't been calculated for this iteration
  hessian_calculated = false;
  // make a copy of the position and gradient
  if (!use_phase_1) {
    xold_old.assign(xold);
  }
  xold.assign(x_);
  gold.assign(g_);
#if PRINT_TO_FILE == 1
  // save to file
  trajectory_file << x_;
#endif
  // copy the gradient into step

  // does a convexity check every T iterations
  // but always starts off in differential equation solver mode
  // also checks whether the gradient is zero
  if ((iter_number_ % T_ == 0 and iter_number_ > 0) or !use_phase_1) {
#if OPTIMIZER_DEBUG_LEVEL >= 3
    std::cout << "checking convexity"
              << "\n";
#endif
    // check if landscape is convex
    bool landscape_is_convex = convexity_check();
    // If lanscape is convex and we're in phase 1, switch to phase 2
    if (use_phase_1 && landscape_is_convex) {
      use_phase_1 = false;
    }
    // If we're in phase 2 and the landscape is not convex, backtrack
    // Line search hackily implemented. Put this in the main line search
    if (!use_phase_1 && !landscape_is_convex) {
      // xold_old -> old step
      // xold -> new step
      // xold_old - xold -> old - new
      int ls_iter = 0;
      while (!landscape_is_convex and ls_iter < 5) {
        step *= 0.5;
        x_.assign(xold_old + step);
        landscape_is_convex = convexity_check();
        std::cout << "convex backtracking -------" << ls_iter << "\n";
        ls_iter++;
      }
      if (ls_iter == 5) {
        // Could not backtrack to a convex landscape, switch back to CVODE.
        std::cout << "------------this is happening---" << std::endl;
        use_phase_1 = true;
        x_.assign(x_last_cvode);
      }
    }
    hessian_calculated = true;
  }
  if (use_phase_1) {
#if OPTIMIZER_DEBUG_LEVEL >= 3
    std::cout << " computing phase 1 step"
              << "\n";
#endif
    compute_phase_1_step(step);
  } else {
    extended_potential->switch_on_extended_potential();
    std::cout << "switched on extended potential" << std::endl;

#if OPTIMIZER_DEBUG_LEVEL >= 3
    std::cout << " computing phase 2 step"
              << "\n";
#endif
    step.assign(g_);
    compute_phase_2_step(step);


    n_phase_2_steps += 1;
    prev_phase_is_phase1 = false;

    // find abs max of the step
    double max_step = 0;
    for (int i = 0; i < step.size(); i++) {
      if (abs(step[i]) > max_step) {
        max_step = abs(step[i]);
      }
    }
    // if max step is too big, switch bag to CVODE
    if (max_step > conv_tol_) {
      use_phase_1 = true;
      x_.assign(x_last_cvode);
    }
    else {
    line_search_method.set_xold_gold_(xold, gold);
    line_search_method.set_g_f_ptr(g_);
    double stepnorm = line_search_method.line_search(x_, step);
    // if step is going uphill, phase two has failed.
    phase_2_failed_ = line_search_method.get_step_moving_away_from_min();
    if (phase_2_failed_) {
      throw std::logic_error("step moving away from minimum");
    }
    }


  }

  // should think of using line search method within the phase steps

  // update inverse hessian estimate

#if OPTIMIZER_DEBUG_LEVEL >= 3
  std::cout << "mixed optimizer: " << iter_number_ << " E " << f_ << " rms "
            << rms_ << " nfev " << nfev_ << " nhev " << udata.nhev << std::endl;
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
void ExtendedMixedOptimizer::reset(pele::Array<double> &x0) {
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
bool ExtendedMixedOptimizer::convexity_check() {

  get_hess_extended(hessian);

  // check if the hessian is positive definite by doing a cholesky decomposition
  // and checking if it is successful

  // Eigen::ComputationInfo info = hessian_shifted.llt().info();
  // std::cout << info << " success info of cholesky decomposition \n";
  // add translation to ensure that the hessian is positive definite
  add_translation_offset_2d(hessian, 1);

  // add negative diagonal offset to ensure hessian is sufficiently positive

  // hessian.diagonal().array() -= 1e-9;

  int N_int = x_.size();

  hessian_copy_for_cholesky = hessian;

  hess_data = hessian_copy_for_cholesky.data();

  // 1 is strlen for uplo for new fortran compilers
  dpotrf_(&uplo, &N_int, hess_data, &N_int, &info, 1);

  std::cout << "info of dpotrf " << info << "\n";
  if (info == 0) {
    // if it is positive definite, we can use the newton method to solve the
    // problem
    return true;
  } else {
    return false;
  }
}

/**
 * Gets the hessian. involves a dense hessian for now. #TODO replace with a
 * sparse hessian.
 */
void ExtendedMixedOptimizer::get_hess(Eigen::MatrixXd &hessian) {
  // Does not allocate memory for hessian just wraps around the data
  Array<double> hessian_pele = Array<double>(hessian.data(), hessian.size());
  potential_->get_hessian(
      x_, hessian_pele); // preferably switch this to sparse Eigen
  udata.nhev += 1;
}

void ExtendedMixedOptimizer::get_hess_extended(Eigen::MatrixXd &hessian) {
  // Does not allocate memory for hessian just wraps around the data
  Array<double> hessian_pele = Array<double>(hessian.data(), hessian.size());
  extended_potential->get_hessian_extended(
      x_, hessian_pele); // preferably switch this to sparse Eigen
  udata.nhev += 1;
}

/**
 * Phase 1 The problem does not look convex, Try solving using with sundials
 */
void ExtendedMixedOptimizer::compute_phase_1_step(Array<double> step) {
  /* advance solver just one internal step */
  xold = x_;

  // use this variable to compute differences and add to nhev later
  double udatadiff = udata.nfev;

  int flag = CVode(cvode_mem, tN, x0_N, &t0, CV_ONE_STEP);
  udatadiff = udata.nfev - udatadiff;
  nfev_ += udatadiff;
  x_ = pele_eq_N_Vector(x0_N);
  // here we have to translate from ODE solver language to optimizer language.
  // this is because CVODE calculates the negative of the gradient.
  g_ = -udata.stored_grad;
  rms_ = (norm(g_) / sqrt(x_.size()));
  f_ = udata.stored_energy;
  step = xold - x_;
  n_phase_1_steps += 1;
  prev_phase_is_phase1 = true;
}

/**
 * Phase 2 The problem looks convex enough to switch to a newton method
 */
void ExtendedMixedOptimizer::compute_phase_2_step(Array<double> step) {

  if (!prev_phase_is_phase1) {
    convexity_check();
  }

  // hessian eigenvalues
  std::cout << "hessian eigenvalues" << hessian.eigenvalues() << std::endl;

  // hessian.diagonal().array() += conv_factor_ * conv_tol_ * 10;

  Eigen::VectorXd r(step.size());
  Eigen::VectorXd q(step.size());

  q.setZero();
  eig_eq_pele(r, step);
}

/**
 * @brief  adds a translation offset to the hessian. This should take care of
 * translational symmetries
 * @param  hessian the hessian to be modified
 * @param  offset the offset to be added
 */
void ExtendedMixedOptimizer::add_translation_offset_2d(Eigen::MatrixXd &hessian,
                                                       double offset) {

  // factor so that the added translation operators are unitary
  double factor = 2.0 / hessian.rows();
  offset = offset * factor;

  for (size_t i = 0; i < hessian.rows(); ++i) {
    for (size_t j = i + 1; j < hessian.cols(); ++j) {
      if ((i % 2 == 0 && j % 2 == 0) || (i % 2 == 1 && j % 2 == 1)) {
        hessian(i, j) += offset;
        hessian(j, i) += offset;
      }
    }
  }
  hessian.diagonal().array() += offset;
}

} // namespace pele
