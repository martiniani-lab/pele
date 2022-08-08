#ifndef _PELE_MIXED_OPT_H__
#define _PELE_MIXED_OPT_H__

#include "array.hpp"
#include "base_potential.hpp"
#include "preprocessor_directives.hpp"

// #define EIGEN_USE_MKL_ALL
// Eigen linear algebra library
#include "eigen_interface.hpp"
#include "pele/lbfgs.hpp"
#include "sundials/sundials_context.h"
#include <Eigen/Dense>
#include <Spectra/SymEigsSolver.h>

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
class MixedOptimizer : public GradientOptimizer {
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
  SUNContext sunctx; // SUNDIALS context

  Array<double> xold; //!< Coordinates before taking a step
  Array<double> gold; //!< Gradient before taking a step
  Array<double> step; //!< Step size and direction
  double
      inv_sqrt_size; //!< The inverse square root the the number of components
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
  Eigen::MatrixXd hessian_shifted;

  double *hess_shifted_data;
  bool usephase1;
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
   * -conv_factor*\lambda_min is addeed to the hessian to make it
   * covex if it is not convex
   */
  double conv_factor_;
  // Need to refactor line searches
  BacktrackingLineSearch line_search_method;

public:
  /**
   * Constructor
   */
  MixedOptimizer(std::shared_ptr<pele::BasePotential> potential,
                 const pele::Array<double> x0, double tol = 1e-4, int T = 1,
                 double step = 1, double conv_tol = 1e-8,
                 double conv_factor = 2, double rtol = 1e-5, double atol = 1e-5,
                 bool iterative = false);
  /**
   * Destructor
   */
  virtual ~MixedOptimizer() { SUNContext_Free(&sunctx); }

  /**
   * Do one iteration iteration of the optimization algorithm
   */
  void one_iteration();

  // functions for accessing the results

  /**
   * reset the optiimzer to start a new minimization from x0
   */
  virtual void reset(pele::Array<double> &x0);
  inline int get_nhev() const { return udata.nhev; }
  inline int get_n_phase_1_steps() { return n_phase_1_steps; }
  inline int get_n_phase_2_steps() { return n_phase_2_steps; }

private:
  bool hessian_calculated; // checks whether the hessian has been calculated for
                           // updating.
  void compute_phase_1_step(Array<double> step);

  void compute_phase_2_step(Array<double> step);
  bool convexity_check();
  bool minimum_less_than_zero;
  void get_hess(Eigen::MatrixXd &hess);

  void update_H0_(Array<double> x_old, Array<double> &g_old,
                  Array<double> x_new, Array<double> &g_new);
  // Does normal LBFGS without preconditioning
  double hessnorm;
  double minimum;
  double pf;
};

} // namespace pele

#endif
