#include "pele/cvode.hpp"
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
//#include <hoomd/HOOMDMath.h>

using namespace std;

namespace pele {
CVODEBDFOptimizer::CVODEBDFOptimizer(
    std::shared_ptr<pele::BasePotential> potential,
    const pele::Array<double> x0, double tol, double rtol, double atol,
    HessianType hessian_type, bool use_newton_stop_criterion,
    bool save_trajectory, int iterations_before_save, double newton_tol,
    double offset_factor)
    : ODEBasedOptimizer(potential, x0, tol, save_trajectory,
                        iterations_before_save),
      N_size(x0.size()),
      tN(1.0),
      ret(0),
      single_step_ret_code(0),
      rtol_(rtol),
      atol_(atol),
      hessian_type_(hessian_type),
      xold(x_.size()),
      use_newton_stop_criterion_(use_newton_stop_criterion),
      hessian(x0.size(), x0.size()),
      newton_tol_(newton_tol),
      offset_factor_(offset_factor) {
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
  switch (hessian_type_) {
    case HessianType::DENSE:
      std::cout << "hessian_type: DENSE" << std::endl;
      break;
    case HessianType::ITERATIVE:
      std::cout << "hessian_type: ITERATIVE" << std::endl;
      break;
  }
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

  time_ = 0;
  Array<double> x0copy = x_.copy();
  x0_N = N_Vector_eq_pele(x0copy, sunctx);

  // initialization of everything CVODE needs
  ret = CVodeInit(cvode_mem, f, time_, x0_N);
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

CVODEBDFOptimizer::~CVODEBDFOptimizer() {
  // free other sundials objects in the constructor
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
  xold.assign(x_);

  single_step_ret_code = CVode(cvode_mem, tN, x0_N, &time_, CV_ONE_STEP);

  assert(static_cast<size_t>(N_VGetLength_Serial(x0_N)) == x_.size());
  x_.assign(pele_eq_N_Vector(x0_N));
  g_.assign(udata.stored_grad);
  gradient_norm_ = (norm(g_) / sqrt(x_.size()));
  f_ = udata.stored_energy;
  nfev_ = udata.nfev;
  Array<double> step = xold - x_;
  step_norm_ = norm(step);
  iter_number_ += 1;

#if PRINT_TO_FILE == 1
  trajectory_file << std::setprecision(17) << x_;
  time_file << std::setprecision(17) << time_ << std::endl;
  gradient_file << std::setprecision(17) << g_;
#endif
}

void CVODEBDFOptimizer::add_translation_offset_2d(Eigen::MatrixXd &hessian,
                                                  double offset) {
  // factor so that the added translation operators are unitary
  double factor = 2.0 / hessian.rows();
  offset = offset * factor;

  for (long i = 0; i < hessian.rows(); ++i) {
    for (long j = i + 1; j < hessian.cols(); ++j) {
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
  if (norm(x_) > 1e10) {
    // do a safe exit
    // succeeded is false, so the optimizer will
    // have passed that the run has failed
    std::cout << "x is too large, exiting" << std::endl;
    return true;
  }

  if (single_step_ret_code < 0) {
    // CVODE failed so, no point continuing
    std::cout << "CVODE failed, exiting" << std::endl;
    succeeded_ = false;
    return true;
  }

  if (gradient_norm_ < tol_) {
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
        double abs_min_eigval = eigenvalues.minCoeff();
        if (abs_min_eigval < 0) {
          abs_min_eigval = -abs_min_eigval;
        } else {
          abs_min_eigval = 0;
        }

        // replace eigenvalues with their absolute values
        for (long i = 0; i < eigenvalues.size(); ++i) {
          eigenvalues(i) = std::abs(eigenvalues(i));
        }

        // cout << "eigenvalues" << eigenvalues << endl;
        // postprocess inverse Eigenvalues

        double average_eigenvalue = eigenvalues.mean();
        double offset = std::max(offset_factor_ * std::abs(average_eigenvalue),
                                 2 * abs_min_eigval);

        hessian.diagonal().array() += offset;

        Eigen::VectorXd newton_step(g_.size());
        newton_step.setZero();
        newton_step = -hessian.ldlt().solve(g_eigen);
#if OPTIMIZER_DEBUG_LEVEL > 0
        std::cout << "offset = " << offset << "\n";
        std::cout << "average eigenvalue = " << average_eigenvalue << "\n";
        std::cout << "abs min eigenvalue = " << abs_min_eigval << "\n";
        std::cout << "checking newton stop criterion\n";
        std::cout << "iter number = " << iter_number_ << "\n";
        std::cout << "gradient norm = " << gradient_norm_ << "\n";
        std::cout << "newton step norm = " << newton_step.norm() << "\n";
#endif
        if (newton_step.norm() < newton_tol_) {
#if OPTIMIZER_DEBUG_LEVEL > 0
          std::cout << "converged in " << iter_number_ << " iterations\n";
          std::cout << "rms = " << gradient_norm_ << "\n";
          std::cout << "tol = " << tol_ << "\n";
#endif
          succeeded_ = true;
          return true;
        } else {
          return false;
        }
      } else {
        return false;
      }
    } else {
#if OPTIMIZER_DEBUG_LEVEL > 0
      std::cout << "converged in " << iter_number_ << " iterations\n";
      std::cout << "rms = " << gradient_norm_ << "\n";
      std::cout << "tol = " << tol_ << "\n";
#endif
      succeeded_ = true;
      return true;
    }

    // Wrap into a matrix
  } else {
    return false;
  }
}

int f(double, N_Vector y, N_Vector ydot, void *user_data) {
  UserData udata = (UserData)user_data;
  pele::Array<double> yw = pele_eq_N_Vector(y);
  // double energy = udata->pot_->get_energy_gradient(yw, g);
  double *fdata = NV_DATA_S(ydot);
  // g = Array<double>(fdata, NV_LENGTH_S(ydot));
  // calculate negative grad g
  double energy = udata->pot_->get_energy_gradient(yw, udata->stored_grad);
  udata->nfev += 1;
  for (size_t i = 0; i < yw.size(); ++i) {
    fdata[i] = -udata->stored_grad[i];
  }
  udata->stored_energy = energy;
  return 0;
}

int Jac(realtype, N_Vector y, N_Vector, SUNMatrix J, void *user_data, N_Vector,
        N_Vector, N_Vector) {
  UserData udata = (UserData)user_data;

  pele::Array<double> yw = pele_eq_N_Vector(y);
  Array<double> g = Array<double>(yw.size());
  // TODO: don't keep allocating memory for every jacobian calculation
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

}  // namespace pele
