#ifndef _PELE_EXTENDED_MIXED_DESCENT_H__
#define _PELE_EXTENDED_MIXED_DESCENT_H__

#include "array.hpp"
#include "base_potential.hpp"
#include "debug.hpp"

// #define EIGEN_USE_MKL_ALL
// Eigen linear algebra library
#include "eigen_interface.hpp"
#include "pele/combine_potentials.hpp"
#include "pele/lbfgs.hpp"
#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <Spectra/SymEigsSolver.h>
#include <array>
#include <memory>

// Lapack for cholesky
extern "C" {
#include <lapacke.h>
}

// line search methods
#include "backtracking.hpp"
#include "bracketing.hpp"
#include "cvode.hpp"
#include "linesearch.hpp"
#include "more_thuente.hpp"
#include "nwpele.hpp"
#include "optimizer.hpp"

#include <cvode/cvode.h>               /* access to CVODE                 */
#include <nvector/nvector_serial.h>    /* access to serial N_Vector       */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix       */

extern "C" {
#include "xsum.h"
}

namespace pele {

// /**
//  * user data passed to CVODE
//  */
// typedef struct UserData_
// {
//     double rtol; /* integration tolerances */
//     double atol;
//     size_t nfev;                // number of gradient(function) evaluations
//     size_t nhev;                // number of hessian (jacobian) evaluations
//     double stored_energy = 0;       // stored energy
//     Array<double>    stored_grad;      // stored gradient. need to pass this
//     on to std::shared_ptr<pele::BasePotential> pot_;
// } * UserData;

/**
 * Mixed optimization scheme that uses different methods of solving far and away
 * from the minimum/ somewhat close to the minimum and near the minimum
 */
class ExtendedMixedOptimizer : public GradientOptimizer {
private:
  /**
   * H0 is the initial estimate for the inverse hessian. This is as small as
   * possible, for making a good inverse hessian estimate next time.
   */
  /**
   * User_Data object. This is important for mxopt because it stores the number
   * of getEnergyGradient and getEnegyGradientHessian evaluations. We're saving
   * them here instead of the optimizer because we'd need to keep a double count
   * since SUNDIALS needs this object.
   */
  UserData_ udata;
  void *cvode_mem; /* CVODE memory         */
  size_t N_size;
  SUNMatrix A;
  SUNLinearSolver LS;
  double t0;
  double tN;
  N_Vector x0_N;
  double rtol;
  double atol;
  /**
   * extension of the original potential that ensures the hessian is positive
   * definite. This has a switch to switch on/off the extension.
   */
  std::shared_ptr<pele::ExtendedPotential> extended_potential;

  Array<double> xold; //!< Coordinates before taking a step
  Array<double> gold; //!< Gradient before taking a step
  Array<double> step; //!< Step size and direction

  /**
  * Coordinate for phase 1. Helps us revert back if newton fails.
  */
  Array<double> xold_old; //!< Save for backtracking if newton fails. TODO: think of better name

  double inv_sqrt_size; //!< The inverse square root the the number of components
  // Preconditioning
  int T_; // number of steps after which the lowest eigenvalues are recalculated
          // in the first phase
  // std::shared_ptr<Eigen::ColPivHouseholderQR<Eigen::MatrixXd>> solver;
  // solver for H x = b
  Eigen::VectorXd
  update_solver(Eigen::VectorXd r); // updates the solver with the new hessian

  char uplo; /* We ask LAPACK for the lower diagonal matrix L */
  int info;  /* "Info" return value, used for error-checking */

  // Calculates hess + delta I where delta makes the new eigenvalue positive
  Eigen::MatrixXd hessian;
  Eigen::MatrixXd hessian_copy_for_cholesky;

  double *hess_data;
  bool use_phase_1;
  bool phase_2_failed_; // if true, we've failed in phase 2 and need to revert
                       // back to where we were in phase 1
  // what phase was used in previous step
  bool prev_phase_is_phase1;
  /**
   * number of phase 1 steps
   */
  size_t n_phase_1_steps;
  /**
   * number of phase 2 steps
   */
  size_t n_phase_2_steps;

  /**
   * tolerance for convexity. the smaller, the more convex the problem
   * needs to be before switching to newton
   */
  double conv_tol_;
  /**
   * -conv_factor*\lambda_min is added to the hessian to make it
   * covex if it is not convex
   */
  double conv_factor_;
  // Need to refactor line searches
  BacktrackingLineSearch line_search_method;

public:
  /**
   * Constructor
   */
  ExtendedMixedOptimizer(
      std::shared_ptr<pele::BasePotential> potential,
      std::shared_ptr<pele::BasePotential> potential_extension,
      const pele::Array<double> x0, double tol = 1e-4, int T = 10,
      double step = 1, double conv_tol = 1e-8, double conv_factor = 2,
      double rtol = 1e-5, double atol = 1e-5, bool iterative = false);
  /**
   * Destructor
   */
  virtual ~ExtendedMixedOptimizer() {}

  /**
   * Do one iteration iteration of the optimization algorithm
   */
  void one_iteration();

  // functions for accessing the results

  /**
   * reset the optimizer to start a new minimization from x0
   */
  virtual void reset(pele::Array<double> &x0);
  inline int get_nhev() const { return udata.nhev; }
  inline int get_nhev_extended() const {
    return extended_potential->get_nhev_extension();
  }
  inline int get_n_phase_1_steps() { return n_phase_1_steps; }
  inline int get_n_phase_2_steps() { return n_phase_2_steps; }

private:
  bool hessian_calculated; // checks whether the hessian has been calculated for
                           // updating.
  void compute_phase_1_step(Array<double> step);

  void compute_phase_2_step(Array<double> step);
  bool convexity_check();
  void get_hess(Eigen::MatrixXd &hess);
  void get_hess_extended(Eigen::MatrixXd &hess);

  void update_H0_(Array<double> x_old, Array<double> &g_old,
                  Array<double> x_new, Array<double> &g_new);

  void add_translation_offset_2d(Eigen::MatrixXd & hessian, double offset);
};

} // namespace pele

#endif // _PELE_EXTENDED_MIXED_DESCENT_H__
