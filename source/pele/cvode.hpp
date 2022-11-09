#ifndef _PELE_CVODE_OPT_H__
#define _PELE_CVODE_OPT_H__

#include "array.hpp"
#include "base_potential.hpp"
#include "preprocessor_directives.hpp"

// #define EIGEN_USE_MKL_ALL

#include "array.hpp"
#include "base_potential.hpp"
#include "cvode/cvode_proj.h"
#include <cstddef>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <sundials/sundials_context.h>
#include <vector>

// #define EIGEN_USE_MKL_ALL
// Eigen linear algebra library
#include "eigen_interface.hpp"
#include <Eigen/Dense>

// line search methods
#include "backtracking.hpp"
#include "bracketing.hpp"
#include "eigen_interface.hpp"
#include "linesearch.hpp"
#include "more_thuente.hpp"
#include "nwpele.hpp"
#include "optimizer.hpp"
#include "sundials/sundials_linearsolver.h"
#include "sundials/sundials_matrix.h"
#include "sundials/sundials_nvector.h"

#include <cvode/cvode.h> /* access to CVODE                 */
#include <fstream>
#include <iostream>
#include <nvector/nvector_serial.h>    /* access to serial N_Vector       */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver */
#include <sunlinsol/sunlinsol_spgmr.h> /* access to SPGMR SUNLinearSolver */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix       */

extern "C" {
#include "xsum.h"
}

namespace pele {

enum HessianType {
  DENSE = 0,
  ITERATIVE = 1,
};
/**
 * user data passed to CVODE
 */
typedef struct UserData_ {

  double rtol; /* integration tolerances */
  double atol;
  size_t nfev;               // number of gradient(function) evaluations
  size_t nhev;               // number of st (jacobian) evaluations
  double stored_energy = 0;  // stored energy
  Array<double> stored_grad; // stored gradient. need to pass this on to
  SUNMatrix stored_J;        // stored dense hessian
  std::shared_ptr<pele::BasePotential> pot_;
} * UserData;

/**
 * Not exactly an optimizer but solves for the differential equation $ dx/dt = -
 * \grad{V(x)} $ to arrive at the trajectory to the corresponding minimum
 */
class CVODEBDFOptimizer : public ODEBasedOptimizer {
private:
  UserData_ udata;
  void *cvode_mem; /* CVODE memory         */
  size_t N_size;
  SUNMatrix A;
  SUNLinearSolver LS;
  double tN;
  int ret;
  SUNContext sunctx; // SUNDIALS context

  // save construction parameters
  double rtol_;
  double atol_;
  HessianType hessian_type_;
#if PRINT_TO_FILE == 1
  std::ofstream trajectory_file;
  std::ofstream time_file;
  std::ofstream hessian_eigvals_file;
  std::ofstream gradient_file;
  std::ofstream step_file;
  std::ofstream newton_step_file;
  std::ofstream lbfgs_m_1_step_file;
  Array<double> g_old;
#endif
  N_Vector x0_N;
  Array<double> xold;
  bool stop_criterion_satisfied();
  bool use_newton_stop_criterion_;
  Eigen::MatrixXd hessian;
  void add_translation_offset_2d(Eigen::MatrixXd &hessian, double offset);
  void setup_cvode();
  void free_cvode_objects();

public:
  void one_iteration();
  // int f(realtype t, N_Vector y, N_Vector ydot, void *user_data);
  // static int Jac(realtype t, N_Vector y, N_Vector fy, SUNMatrix J,
  //                void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector
  //                tmp3);
  CVODEBDFOptimizer(std::shared_ptr<pele::BasePotential> potential,
                    const pele::Array<double> x0, double tol = 1e-5,
                    double rtol = 1e-5, double atol = 1e-5,
                    HessianType hessian_type = DENSE,
                    bool use_newton_stop_criterion = false);

  ~CVODEBDFOptimizer();

  // Copy Constructor (raises a not implemented exception)
  CVODEBDFOptimizer(const CVODEBDFOptimizer &) : ODEBasedOptimizer() {
    throw std::runtime_error("CVODEBDFOptimizer is not implemented");
  };

  //
  CVODEBDFOptimizer &operator=(const CVODEBDFOptimizer &other) {
    this->potential_ = other.potential_;
    this->x_ = other.x_;
    this->tol_ = other.tol_;
    this->rtol_ = other.rtol_;
    this->atol_ = other.atol_;
    this->hessian_type_ = other.hessian_type_;
    this->use_newton_stop_criterion_ = other.use_newton_stop_criterion_;

    // free cvode memory and setup again
    this->free_cvode_objects();
    this->setup_cvode();
    return *this;
  }

  inline int get_nhev() const { return udata.nhev; }

protected:
  double H02;
};

/**
 * creates a new N_vector array that wraps around the pele array data
 */
inline N_Vector N_Vector_eq_pele(pele::Array<double> x, SUNContext sunctx) {
  N_Vector y;
  y = N_VNew_Serial(x.size(), sunctx);
  for (size_t i = 0; i < x.size(); ++i) {
    NV_Ith_S(y, i) = x[i];
  }
  // NV_DATA_S(y) = x.copy().data();
  return y;
}

/**
 * wraps the sundials data in a pele array and passes it on
 */
inline pele::Array<double> pele_eq_N_Vector(N_Vector x) {
  Array<double> y = Array<double>(N_VGetLength(x));
  for (size_t i = 0; i < y.size(); ++i) {
    y[i] = NV_Ith_S(x, i);
  }
  return pele::Array<double>(NV_DATA_S(x), N_VGetLength(x)).copy();
}

int f(realtype t, N_Vector y, N_Vector ydot, void *user_data);
int Jac(realtype t, N_Vector y, N_Vector fy, SUNMatrix J, void *user_data,
        N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);
static int Jac2(realtype t, N_Vector y, N_Vector fy, SUNMatrix J,
                void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);
static int f2(realtype t, N_Vector y, N_Vector ydot, void *user_data);

static int check_sundials_retval(void *return_value, const char *funcname,
                                 int opt);

} // namespace pele

#endif