#include "pele/extended_mixed_descent.hpp"
#include <algorithm>
#include <complex>
#include <cstddef>
#include <cvode/cvode.h>
#include <iostream>
#include <limits>
#include <memory>
#include <ostream>
#include <pele/cvode.hpp>
#include <stdexcept>
#include <sundials/sundials_matrix.h>
#include <sunlinsol/sunlinsol_dense.h>  // access to dense SUNLinearSolver
#include <sunlinsol/sunlinsol_spgmr.h>  /* access to SPGMR SUNLinearSolver */
#include <sunnonlinsol/sunnonlinsol_newton.h> /* access to Newton SUNNonLinearSolver */
#include <unistd.h>

#include "pele/array.hpp"
#include "pele/base_potential.hpp"
#include "pele/eigen_interface.hpp"
#include "pele/hessian_translations.hpp"
#include "pele/lbfgs.hpp"
#include "pele/lsparameters.hpp"
#include "pele/optimizer.hpp"
#include "pele/preprocessor_directives.hpp"

namespace pele {

ExtendedMixedOptimizer::ExtendedMixedOptimizer(
    std::shared_ptr<pele::BasePotential> potential,
    const pele::Array<double> x0,
    std::shared_ptr<pele::BasePotential> potential_extension, double tol, int T,
    double step, double conv_tol, double rtol, double atol, bool iterative,
    const Array<double> global_symmetry_offset, bool save_trajectory,
    int iterations_before_save, StopCriterionType stop_criterion)
    : GradientOptimizer(potential, x0, tol, save_trajectory,
                        iterations_before_save, stop_criterion),
      N_size(x_.size()),
      t0(0),
      tN(100.0),
      rtol(rtol),
      atol(atol),
      extended_potential(
          std::make_shared<ExtendedPotential>(potential, potential_extension)),
      xold(x_.size()),
      gold(x_.size()),
      step(x_.size()),
      xold_old(x_.size()),
      x_last_cvode(x_.size()),
      T_(T),
      hessian(x_.size(), x_.size()),
      hessian_copy_for_cholesky(x_.size(), x_.size()),
      use_phase_1(true),
      prev_phase_is_phase1(true),
      n_phase_1_steps(0),
      n_phase_2_steps(0),
      n_failed_phase_2_steps(0),
      conv_tol_(conv_tol),
      line_search_method(this, step),
      iterative_(iterative),
      m_global_symmetry_offset(global_symmetry_offset.copy()) {
  std::cout << "ExtendedMixedOptimizer Parameters" << std::endl;
  std::cout << "x0: " << x0 << std::endl;
  std::cout << "tol: " << tol << std::endl;
  std::cout << "T: " << T << std::endl;
  std::cout << "step: " << step << std::endl;
  std::cout << "conv_tol: " << conv_tol << std::endl;
  std::cout << "rtol: " << rtol << std::endl;
  std::cout << "atol: " << atol << std::endl;
  if (iterative) {
    std::cout << "iterative: true" << std::endl;
  } else {
    std::cout << "iterative: false " << std::endl;
  }

  if (iterative) {
    hessian_type_ = ITERATIVE;
  } else {
    hessian_type_ = DENSE;
  }
  std::cout << "initial x0" << x0 << std::endl;
  setup_cvode();
  uplo = 'U';
  if (T <= 1) {
    throw std::runtime_error(
        "T must be greater than 1. switching back to CVODE from Newton will "
        "get stuck in an infinite loop");
  }
  // Check whether potentials are Null
  if (!potential_) {
    throw std::runtime_error("ExtendedMixedOptimizer: potential is null");
  }
  if (!potential_extension) {
    std::cout << "No potential extension being used" << std::endl;
  }
  set_potential(
      extended_potential);  // because we can only create the extended potential
#if PRINT_TO_FILE == 1
  trajectory_file.open("trajectory.txt");
#endif

#if OPTIMIZER_DEBUG_LEVEL >= 1
  std::cout << "Mixed optimizer constructed"
            << "T=" << T_ << "\n";
#endif
}
/**
 * Does one iteration of the optimization algorithm
 */
/**
 * setup the CVODE solver. extracted for use in assigment operators/constructors
 */
void ExtendedMixedOptimizer::setup_cvode() {
  cvode_mem = NULL;
  x0_N = NULL;
  A = NULL;
  sunctx = NULL;
  LS = NULL;
  int ret = 0;

  sunctx = NULL;
  ret = SUNContext_Create(NULL, &sunctx);
  if (check_sundials_retval(&ret, "SUNContext_Create", 1)) {
    throw std::runtime_error("SUNContext_Create failed");
  }

  cvode_mem = CVodeCreate(CV_BDF, sunctx);
  if (cvode_mem == NULL) {
    std::cerr << "CVodeCreate failed to create CVODE solver" << std::endl;
    exit(1);
  }
  t0 = 0;
  Array<double> x0copy = x_.copy();
  x0_N = N_Vector_eq_pele(x0copy, sunctx);

  ret = CVodeInit(cvode_mem, f, t0, x0_N);
  if (check_sundials_retval(&ret, "CVodeInit", 1)) {
    throw std::runtime_error("CVODE initialization failed");
  }

  // initialize userdata
  udata.rtol = rtol;
  udata.atol = atol;
  udata.nfev = 0;
  udata.nhev = 0;
  udata.pot_ = potential_;
  udata.stored_grad = Array<double>(x_.size(), 0);

  ret = CVodeSStolerances(cvode_mem, udata.rtol, udata.atol);
  if (check_sundials_retval(&ret, "CVodeSStolerances", 1)) {
    throw std::runtime_error("CVODE tolerances failed");
  }

  ret = CVodeSetUserData(cvode_mem, &udata);
  if (check_sundials_retval(&ret, "CVodeSetUserData", 1)) {
    throw std::runtime_error("CVODE user data failed");
  }
  if (hessian_type_ == ITERATIVE) {
    LS = SUNLinSol_SPGMR(x0_N, SUN_PREC_NONE, 0, sunctx);
    if (check_sundials_retval((void *)LS, "SUNLinSol_SPGMR", 0)) {
      throw std::runtime_error("SUNLinSol_SPGMR failed");
    }

    ret = CVodeSetLinearSolver(cvode_mem, LS, NULL);
    if (check_sundials_retval(&ret, "CVodeSetLinearSolver", 1)) {
      throw std::runtime_error("CVODE linear solver failed");
    }

  } else if (hessian_type_ == DENSE) {
    A = SUNDenseMatrix(N_size, N_size, sunctx);
    if (check_sundials_retval((void *)A, "SUNDenseMatrix", 0)) {
      throw std::runtime_error("SUNDenseMatrix failed");
    }
    LS = SUNLinSol_Dense(x0_N, A, sunctx);
    if (check_sundials_retval((void *)LS, "SUNLinSol_Dense", 0)) {
      throw std::runtime_error("SUNLinSol_Dense failed");
    }

    ret = CVodeSetLinearSolver(cvode_mem, LS, A);
    if (check_sundials_retval(&ret, "CVodeSetLinearSolver", 1)) {
      throw std::runtime_error("CVODE linear solver failed");
    }
    ret = CVodeSetJacFn(cvode_mem, Jac);
    if (check_sundials_retval(&ret, "CVodeSetJacFn", 1)) {
      throw std::runtime_error("CVODE set jacobian failed");
    }
  } else {
    throw std::runtime_error("Unknown Hessian type");
  }
  g_.assign(udata.stored_grad);
  ret = CVodeSetMaxNumSteps(cvode_mem, 1000000);
  if (check_sundials_retval(&ret, "CVodeSetMaxNumSteps", 1)) {
    throw std::runtime_error("CVODE set max num steps failed");
  }
  ret = CVodeSetStopTime(cvode_mem, tN);
  if (check_sundials_retval(&ret, "CVodeSetStopTime", 1)) {
    throw std::runtime_error("CVODE set stop time failed");
  }
#if PRINT_TO_FILE == 1
  trajectory_file.open("trajectory_cvode.txt");
  gradient_file.open("gradient_cvode.txt");
  time_file.open("time_cvode.txt");
#endif
}

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
      x_last_cvode.assign(x_);
    }
    hessian_calculated = true;
  }
  if (use_phase_1) {
#if OPTIMIZER_DEBUG_LEVEL >= 3
    std::cout << " computing phase 1 step"
              << "\n";
#endif
    compute_phase_1_step();
  } else {
    extended_potential->switch_on_extended_potential();
#if OPTIMIZER_DEBUG_LEVEL >= 3
    std::cout << " computing phase 2 step"
              << "\n";
#endif
    step.assign(g_);
    compute_phase_2_step();

    n_phase_2_steps += 1;
    prev_phase_is_phase1 = false;

    auto compare = [](double a, double b) { return abs(a) < abs(b); };
    auto max_step_it = std::max_element(step.begin(), step.end(), compare);
    double max_step = abs(*max_step_it);

    // if max step is too big, switch back to CVODE
    if (max_step > conv_tol_) {
#if OPTIMIZER_DEBUG_LEVEL >= 3
      std::cout << "max step too big, switching back to CVODE"
                << "\n";
      std::cout << "using phase 1 step"
                << "\n";
      std::cout << "switching off extended potential"
                << "\n";
#endif
      use_phase_1 = true;
      x_.assign(x_last_cvode);
      extended_potential->switch_off_extended_potential();
      n_failed_phase_2_steps += 1;
    } else {
      line_search_method.set_xold_gold_(xold, gold);
      line_search_method.set_g_f_ptr(g_);
      line_search_method.line_search(x_, step);
      // if step is going uphill, phase two has failed.
      phase_2_failed_ = line_search_method.get_step_moving_away_from_min();

      if (phase_2_failed_) {
#if OPTIMIZER_DEBUG_LEVEL >= 3
        std::cout << "phase 2 failed, step moving away from minimum"
                  << "\n";
        std::cout << "switching back to CVODE"
                  << "\n";
#endif
        use_phase_1 = true;
        x_.assign(x_last_cvode);
        extended_potential->switch_off_extended_potential();
        n_failed_phase_2_steps += 1;
      }
    }
  }

  // should think of using line search method within the phase steps

  // update inverse hessian estimate

#if OPTIMIZER_DEBUG_LEVEL >= 3
  std::cout << "mixed optimizer: " << iter_number_ << " E " << f_ << " n "
            << gradient_norm_ << " nfev " << nfev_ << " nhev " << udata.nhev
            << std::endl;
  std::cout << "optimizer position \n" << x_ << "\n";
  std::cout << "optimizer step \n" << step << "\n";
#endif
  /**
   * Checks whether the stop criterion is satisfied: if stop criterion is
   * satisfied The new version is done. at this point the corresponding values
   * are added
   */
  step_norm_ = norm(step);
  iter_number_ += 1;
}

ExtendedMixedOptimizer::~ExtendedMixedOptimizer() {
#if PRINT_TO_FILE == 1
  trajectory_file.close();
#endif
  free_cvode_objects();
}

void ExtendedMixedOptimizer::free_cvode_objects() {
  N_VDestroy(x0_N);
  SUNLinSolFree(LS);
  if (!iterative_) {
    SUNMatDestroy(A);
  }
  CVodeFree(&cvode_mem);
  SUNContext_Free(&sunctx);
}

void ExtendedMixedOptimizer::reset(Array<double> &x0) {
  // reset vectors
  x_.assign(x0);
  xold.assign(x0);
  gold.assign(0);
  step.assign(0);
  xold_old.assign(0);
  x_last_cvode.assign(0);
  reset_cvode();
  reset_newton();
  use_phase_1 = true;
  n_phase_1_steps = 0;
  n_phase_2_steps = 0;
  hessian_calculated = false;
}

void ExtendedMixedOptimizer::reset_cvode() {
  // std::cout << "resetting cvode objects" << std::endl;
  // std::cout << "previous nfev: " << nfev_ << std::endl;
  // std::cout << "previous nhev: " << udata.nhev << std::endl;
  // std::cout << "previous n_phase_1_steps: " << n_phase_1_steps << std::endl;
  // std::cout << "previous n_phase_2_steps: " << n_phase_2_steps << std::endl;
  nfev_ = 0;
  nhev_ = 0;
  succeeded_ = false;
  iter_number_ = 0;
  func_initialized_ = false;
  reset_user_data(&udata);
  this->free_cvode_objects();
  this->setup_cvode();
}

void ExtendedMixedOptimizer::reset_newton() {
  hessian.setZero();
  hessian_copy_for_cholesky.setZero();
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
  get_hess(hessian);

  // check if the hessian is positive definite by doing a cholesky decomposition
  // and checking if it is successful

  // Eigen::ComputationInfo info = hessian_shifted.llt().info();
  // std::cout << info << " success info of cholesky decomposition \n";
  // add translation to ensure that the hessian is positive definite

  if (m_global_symmetry_offset.size() == 0) {
    add_translation_offset_2d(hessian, 1);
  } else {
    // add translation offset to hessian data
    // wrap global symmetry data in a eigen matrix
    add_symmetry_offset(hessian, m_global_symmetry_offset);
  }

  int N_int = x_.size();

  hessian_copy_for_cholesky = hessian;

  hess_data = hessian_copy_for_cholesky.data();

  // 1 is strlen for uplo for new fortran compilers
  dpotrf_(&uplo, &N_int, hess_data, &N_int, &info
#ifdef LAPACK_FORTRAN_STRLEN_END
          ,
          1
#endif
  );

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
      x_, hessian_pele);  // preferably switch this to sparse Eigen
  udata.nhev += 1;
}

void ExtendedMixedOptimizer::get_hess_extended(Eigen::MatrixXd &hessian) {
  // Does not allocate memory for hessian just wraps around the data
  Array<double> hessian_pele = Array<double>(hessian.data(), hessian.size());
  extended_potential->get_hessian_extended(
      x_, hessian_pele);  // preferably switch this to sparse Eigen
  udata.nhev += 1;
}

/**
 * Phase 1 The problem does not look convex, Try solving using with sundials
 */
void ExtendedMixedOptimizer::compute_phase_1_step() {
  /* advance solver just one internal step */
  xold.assign(x_);

  // use this variable to compute differences and add to nhev later
  double udatadiff = udata.nfev;

  retval = CVode(cvode_mem, tN, x0_N, &t0, CV_ONE_STEP);
  udatadiff = udata.nfev - udatadiff;
  nfev_ += udatadiff;
  x_.assign(pele_eq_N_Vector(x0_N));
  // here we have to translate from ODE solver language to optimizer language.
  // this is because CVODE calculates the negative of the gradient.
  g_ = -udata.stored_grad;
  gradient_norm_ = (norm(g_) / sqrt(x_.size()));
  f_ = udata.stored_energy;
  step = xold - x_;
  n_phase_1_steps += 1;
  prev_phase_is_phase1 = true;
}

/**
 * Phase 2 The problem looks convex enough to switch to a newton method
 */
void ExtendedMixedOptimizer::compute_phase_2_step() {
  if (!prev_phase_is_phase1) {
    convexity_check();
  }

  Eigen::VectorXd r(step.size());
  Eigen::VectorXd q(step.size());

  q.setZero();
  eig_eq_pele(r, step);

  q = -hessian.householderQr().solve(r);
  pele_eq_eig(step, q);
}

}  // namespace pele
