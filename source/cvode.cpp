#include "pele/cvode.hpp"

#include "cvode/cvode.h"
#include "cvode/cvode_ls.h"
#include "nvector/nvector_serial.h"
#include "pele/array.hpp"
#include "pele/base_potential.hpp"
#include "pele/optimizer.hpp"
#include "pele/preprocessor_directives.hpp"
#include "sundials/sundials_linearsolver.h"
#include "sundials/sundials_nvector.h"
#include "sunmatrix/sunmatrix_dense.h"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <memory>
#include <ostream>
#include <stdio.h>
#include <sundials/sundials_context.h>
#include <sundials/sundials_matrix.h>
#include <sunlinsol/sunlinsol_klu.h> /* access to KLU sparse direct solver   */
#include <sunmatrix/sunmatrix_sparse.h> /* access to sparse SUNMatrix           */
//#include <hoomd/HOOMDMath.h>

using namespace std;

#define NUMERICAL_ZERO 1e-15
#define THRESHOLD 1e-9
#define NEWTON_TOL 1e-5
#define ZERO 0.0
namespace pele {
CVODEBDFOptimizer::CVODEBDFOptimizer(
    std::shared_ptr<pele::BasePotential> potential,
    const pele::Array<double> x0, double tol, double rtol, double atol,
    HessianType hessian_type, bool use_newton_stop_criterion)
    : GradientOptimizer(potential, x0, tol), N_size(x0.size()), rtol_(rtol),
      atol_(atol), hessian_type_(hessian_type), hessian(x0.size(), x0.size()), t0(0),
      tN(10000000.0), ret(0),
      use_newton_stop_criterion_(use_newton_stop_criterion) {
  setup_cvode();
};

/**
 * setup the CVODE solver. extracted for use in assigment operators/constructors
 */
void CVODEBDFOptimizer::setup_cvode() {
  cvode_mem = NULL;
  x0_N = NULL;
  A = NULL;
  sunctx = NULL;
  LS = NULL;

#if OPTIMIZER_DEBUG_LEVEL > 0
  std::cout << "CVODE constructed with parameters: " << std::endl;
  std::cout << "x0: " << x_ << std::endl;
  std::cout << "tol: " << tol_ << std::endl;
  std::cout << "rtol: " << rtol_ << std::endl;
  std::cout << "atol: " << atol_ << std::endl;
  std::cout << "iterative: " << iterative_ << std::endl;
  std::cout << "use_newton_stop_criterion: " << use_newton_stop_criterion_
            << std::endl;
#endif

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

  // dummy t0
  double t0 = 0;
  Array<double> x0copy = x_.copy();
  x0_N = N_Vector_eq_pele(x0copy, sunctx);

  // initialization of everything CVODE needs
  ret = CVodeInit(cvode_mem, f, t0, x0_N);
  if (check_sundials_retval(&ret, "CVodeInit", 1)) {
    throw std::runtime_error("CVODE initialization failed");
  }

  // initialize userdata
  udata.rtol = rtol_;
  udata.atol = atol_;
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
  bool sparse = true;
  if (hessian_type_ == ITERATIVE) {
    LS = SUNLinSol_SPGMR(x0_N, SUN_PREC_NONE, 0, sunctx);
    if (check_sundials_retval((void *)LS, "SUNLinSol_SPGMR", 0)) {
      throw std::runtime_error("SUNLinSol_SPGMR failed");
    }

    ret = CVodeSetLinearSolver(cvode_mem, LS, NULL);
    if (check_sundials_retval(&ret, "CVodeSetLinearSolver", 1)) {
      throw std::runtime_error("CVODE linear solver failed");
    }

  } else if (hessian_type_ == SPARSE) {
    A = SUNSparseMatrix(N_size, N_size, N_size * N_size, CSC_MAT, sunctx);
    udata.stored_J = SUNDenseMatrix(N_size, N_size, sunctx);
    LS = SUNLinSol_KLU(x0_N, A, sunctx);
    if (LS == NULL) {
      throw std::runtime_error("SUNLinSol_KLU failed");
    }
    CVodeSetLinearSolver(cvode_mem, LS, A);
    CVodeSetJacFn(cvode_mem, Jac_sparse);
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
  }
  else {
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
  // Also saving hessian eigenvalues to understand what's happening
  // along the run
  hessian_eigvals_file.open("hessian_eigvals_cvode.txt");
  grad_file.open("grad_cvode.txt");
  step_file.open("step_cvode.txt");
  newton_step_file.open("newton_step_cvode.txt");
  lbfgs_m_1_step_file.open("lbfgs_m_1_step_cvode.txt");
  g_old = Array<double>(x0.size(), 0);
#endif
}

CVODEBDFOptimizer::~CVODEBDFOptimizer() {
  // free other sundials objects in the constructor
  if (hessian_type_ == SPARSE) {
    SUNMatDestroy(udata.stored_J);
  }
  // reused in the move/copy constructor so the function has been extracted
  free_cvode_objects();
}

/**
 * @brief Free CVODE objects.
 * @details This function is called in the destructor and in the move
 * constructor.
 */
void CVODEBDFOptimizer::free_cvode_objects() {
  N_VDestroy(x0_N);
  SUNLinSolFree(LS);
  SUNMatDestroy(A);
  CVodeFree(&cvode_mem);
  SUNContext_Free(&sunctx);
}

void CVODEBDFOptimizer::one_iteration() {
  /* advance solver just one internal step */
  Array<double> xold = x_.copy();

  ret = CVode(cvode_mem, tN, x0_N, &t0, CV_ONE_STEP);

  // if (check_sundials_retval(&ret, "CVode", 1)) {
  //   throw std::runtime_error("CVODE single step failed");
  // }
  iter_number_ += 1;

  // Assert length of x0_N is the same as x_
  assert(N_VGetLength_Serial(x0_N) == x_.size());
  x_.assign(pele_eq_N_Vector(x0_N));
  g_.assign(udata.stored_grad);
  rms_ = (norm(g_) / sqrt(x_.size()));
  f_ = udata.stored_energy;
  nfev_ = udata.nfev;
  Array<double> step = xold - x_;

  // really hacky way to output $lambdamin/lambdamax on a low tolerance run
  // simply print the energy and lambdamin/lambdamax as csv values and write
  // stdout to file then plot them using python

#if PRINT_TO_FILE == 1
  // get hessian routine, useful for understanding conditioning near minimum
  Array<double> hess(xold.size() * xold.size());
  Array<double> grad(xold.size());
  double e = potential_->get_energy_gradient_hessian(x_, grad, hess);
  Eigen::MatrixXd hess_dense(xold.size(), xold.size());
  udata.nhev += 1;
  hess_dense.setZero();
  for (size_t i = 0; i < xold.size(); ++i) {
    for (size_t j = 0; j < xold.size(); ++j) {
      hess_dense(i, j) = hess[i + grad.size() * j];
    }
  }

  // calculate minimum and maximum eigenvalue
  Eigen::VectorXd eigvals = hess_dense.eigenvalues().real();
  add_translation_offset_2d(hess_dense, 1.0);
  Eigen::VectorXd r(hess_dense.rows());
  Eigen::VectorXd q(hess_dense.rows());
  q.setZero();

  eig_eq_pele(r, g_);

  q = -hess_dense.householderQr().solve(r);

  double minimum = eigvals.minCoeff();
  double maximum = eigvals.maxCoeff();

  Array<double> eigvals_pele = Array<double>(xold.size());
  for (size_t i = 0; i < xold.size(); ++i) {
    eigvals_pele[i] = eigvals(i);
  }

  Array<double> q_pele = Array<double>(xold.size());

  for (size_t i = 0; i < xold.size(); ++i) {
    q_pele[i] = q(i);
  }
  Array<double> dg_old = Array<double>(xold.size());
  Array<double> dx_old = Array<double>(xold.size());
  if (iter_number_ >= 0) {
    dg_old = g_ - g_old;
    dx_old = x_ - xold;
  }
  double dg_dot_dx;
  double dg_dot_dg;
  for (size_t i = 0; i < xold.size(); ++i) {
    dg_dot_dx += dg_old[i] * dx_old[i];
    dg_dot_dg += dg_old[i] * dg_old[i];
  }

  double H0 = dg_dot_dx / dg_dot_dg;

  Array<double> H0_g = Array<double>(xold.size());

  H0_g.assign(H0 * g_);

  // write to file
  trajectory_file << std::setprecision(17) << x_;
  hessian_eigvals_file << std::setprecision(17) << eigvals_pele;
  grad_file << std::setprecision(17) << grad;
  step_file << std::setprecision(17) << step;
  newton_step_file << std::setprecision(17) << q_pele;
  lbfgs_m_1_step_file << std::setprecision(17) << H0_g;
  g_old.assign(g_);
#endif
}

void CVODEBDFOptimizer::add_translation_offset_2d(Eigen::MatrixXd &hessian,
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

bool CVODEBDFOptimizer::stop_criterion_satisfied() {
  if (!func_initialized_) {
    // For some reason clang throws an error here, but gcc compiles properly
    initialize_func_gradient();
  }
  if (rms_ < tol_) {
    if (use_newton_stop_criterion_) {
      if (iter_number_ % 100 == 0) {
        // Check with a more stringent hessian condition
        Array<double> pele_hessian =
            Array<double>(hessian.data(), hessian.size());

        Eigen::Map<Eigen::VectorXd, 0, Eigen::InnerStride<1>> g_eigen(
            g_.data(), int(g_.size()));
        potential_->get_hessian(x_, pele_hessian);

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es =
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(hessian);
        Eigen::VectorXd eigenvalues = es.eigenvalues();
        Eigen::MatrixXd eigenvectors = es.eigenvectors();

        // replace eigenvalues with their absolute values
        for (size_t i = 0; i < eigenvalues.size(); ++i) {
          eigenvalues(i) = std::abs(eigenvalues(i));
        }

        // cout << "eigenvalues" << eigenvalues << endl;
        // postprocess inverse Eigenvalues
        Eigen::VectorXd inv_eigenvalues(eigenvalues.size());
        for (size_t i = 0; i < eigenvalues.size(); ++i) {
          double mod_eig = abs(eigenvalues(i));
          if (mod_eig < NUMERICAL_ZERO) {
            // case for translational symmetries
            inv_eigenvalues(i) = 0.0;
          } else if (mod_eig < THRESHOLD) {
            // case for really small eigenvalues
            inv_eigenvalues(i) = 1.0 / THRESHOLD;
          } else {
            // case for normal eigenvalues
            inv_eigenvalues(i) = 1.0 / mod_eig;
          }
        }
        Eigen::VectorXd newton_step(g_.size());
        newton_step.setZero();

        // find the largest eigenvalue
        newton_step =
            -eigenvectors *
            ((eigenvectors.transpose() * g_eigen).cwiseProduct(inv_eigenvalues))
                .matrix();
        if (newton_step.norm() < NEWTON_TOL) {
          std::cout << "converged in " << iter_number_ << " iterations\n";
          std::cout << "rms = " << rms_ << "\n";
          std::cout << "tol = " << tol_ << "\n";
          return true;
        } else {
          return false;
        }
      } else {
        return false;
      }
    } else {
      std::cout << "converged in " << iter_number_ << " iterations\n";
      std::cout << "rms = " << rms_ << "\n";
      std::cout << "tol = " << tol_ << "\n";
      return true;
    }

    // Wrap into a matrix
  } else {
    return false;
  }
}

int f(double t, N_Vector y, N_Vector ydot, void *user_data) {
  UserData udata = (UserData)user_data;
  pele::Array<double> yw = pele_eq_N_Vector(y);
  // double energy = udata->pot_->get_energy_gradient(yw, g);
  double *fdata = NV_DATA_S(ydot);
  // g = Array<double>(fdata, NV_LENGTH_S(ydot));
  // calculate negative grad g
  double energy = udata->pot_->get_energy_gradient(yw, udata->stored_grad);
  udata->nfev += 1;
#pragma simd
  for (size_t i = 0; i < yw.size(); ++i) {
    fdata[i] = -udata->stored_grad[i];
  }
  udata->stored_energy = energy;
  return 0;
}

int Jac(realtype t, N_Vector y, N_Vector fy, SUNMatrix J, void *user_data,
        N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
  UserData udata = (UserData)user_data;

  pele::Array<double> yw = pele_eq_N_Vector(y);
  Array<double> g = Array<double>(yw.size());
  // TODO: don't keep allocating memory for every jaocobian calculation
  Array<double> h = Array<double>(yw.size() * yw.size());
  udata->pot_->get_energy_gradient_hessian(pele_eq_N_Vector(y), g, h);
  udata->nhev += 1;

  double *hessdata = SUNDenseMatrix_Data(J);
  for (size_t i = 0; i < yw.size(); ++i) {
    for (size_t j = 0; j < yw.size(); ++j) {
      hessdata[i * yw.size() + j] = -h[i * yw.size() + j];
    }
  }
  return 0;
};

int Jac_sparse(realtype t, N_Vector y, N_Vector fy, SUNMatrix J,
               void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
  UserData udata = (UserData)user_data;

  pele::Array<double> yw = pele_eq_N_Vector(y);
  Array<double> g = Array<double>(yw.size());
  // TODO: don't keep allocating memory for every jaocobian calculation
  Array<double> h = Array<double>(yw.size() * yw.size());
  udata->pot_->get_energy_gradient_hessian(pele_eq_N_Vector(y), g, h);
  udata->nhev += 1;

  double *hessdata = SUNDenseMatrix_Data(udata->stored_J);
  for (size_t i = 0; i < yw.size(); ++i) {
    for (size_t j = 0; j < yw.size(); ++j) {
      hessdata[i * yw.size() + j] = -h[i * yw.size() + j];
      std::cout << hessdata << std::endl;
    }
  }
  std::cout << "sparse jacobian before" << std::endl;
  SUNSparseMatrix_Print(J, stdout);
  std::cout << "dense jacobian before" << std::endl;
  SUNDenseMatrix_Print(udata->stored_J, stdout);

// SUNSparseFromDenseMatrix_inplace(udata->stored_J, J, 1e-12, CSC_MAT);

  std::cout << "sparse jacobian after" << std::endl;
  SUNSparseMatrix_Print(J, stdout);
  std::cout << "dense jacobian after" << std::endl;
  SUNDenseMatrix_Print(udata->stored_J, stdout);
  return 0;
};

SUNMatrix SUNSparseFromDenseMatrix_inplace(SUNMatrix Ad, SUNMatrix As,
                                           realtype droptol, int sparsetype) {
  sunindextype i, j, nnz;
  sunindextype M, N;

  /* check for legal sparsetype, droptol and input matrix type */
  if ((sparsetype != CSR_MAT) && (sparsetype != CSC_MAT))
    return NULL;
  if (droptol < ZERO)
    return NULL;
  if (SUNMatGetID(Ad) != SUNMATRIX_DENSE)
    return NULL;

  /* set size of new matrix */
  M = SM_ROWS_D(Ad);
  N = SM_COLUMNS_D(Ad);

  if (As == NULL)
    return NULL;

  /* copy nonzeros from Ad into As, based on CSR/CSC type */
  nnz = 0;
  if (sparsetype == CSC_MAT) {
    for (j = 0; j < N; j++) {
      (SM_INDEXPTRS_S(As))[j] = nnz;
      for (i = 0; i < M; i++) {
        if (abs(SM_ELEMENT_D(Ad, i, j)) > droptol) {
          (SM_INDEXVALS_S(As))[nnz] = i;
          (SM_DATA_S(As))[nnz++] = SM_ELEMENT_D(Ad, i, j);
        }
      }
    }
    (SM_INDEXPTRS_S(As))[N] = nnz;
  } else { /* CSR_MAT */
    for (i = 0; i < M; i++) {
      (SM_INDEXPTRS_S(As))[i] = nnz;
      for (j = 0; j < N; j++) {
        if (abs(SM_ELEMENT_D(Ad, i, j)) > droptol) {
          (SM_INDEXVALS_S(As))[nnz] = j;
          (SM_DATA_S(As))[nnz++] = SM_ELEMENT_D(Ad, i, j);
        }
      }
    }
    (SM_INDEXPTRS_S(As))[M] = nnz;
  }

  return (As);
}

/**
 * @brief Checks sundials error code and prints out error message. Copied from
 * the sundials examples. Honestly this is 3 functions in one. it needs to be
 * split
 *
 * @param flag
 * @param funcname
 * @param opt
 */
static int check_sundials_retval(void *return_value, const char *funcname,
                                 int opt) {

  int *retval;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */

  if (opt == 0 && return_value == NULL) {
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return (1);
  }

  /* Check if retval < 0 */

  else if (opt == 1) {
    retval = (int *)return_value;
    if (*retval < 0) {
      fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with retval = %d\n\n",
              funcname, *retval);
      return (1);
    }
  }

  /* Check if function returned NULL pointer - no memory allocated */

  else if (opt == 2 && return_value == NULL) {
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return (1);
  }

  return (0);
}

} // namespace pele
