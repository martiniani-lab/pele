#include "pele/cvode.hpp"
#include "autodiff/reverse/eigen.hpp"
#include "cvode/cvode.h"
#include "cvode/cvode_ls.h"
#include "pele/array.hpp"
#include "pele/base_potential.hpp"
#include "pele/debug.hpp"
#include "pele/optimizer.hpp"

// cvode imports
#include "petscerror.h"
#include "petscksp.h"
#include "petscmat.h"
#include "petscsnes.h"
#include "petscsystypes.h"
#include "petscvec.h"
#include "petscviewer.h"
#include "sundials/sundials_linearsolver.h"
#include "sundials/sundials_nonlinearsolver.h"
#include "sundials/sundials_nvector.h"
#include "sunmatrix/sunmatrix_dense.h"
#include "sunnonlinsol/sunnonlinsol_petscsnes.h"
#include <cstddef>
#include <iostream>
#include <memory>
#include <nvector/nvector_petsc.h>
#include <petscsys.h>
#include <sys/types.h>
// #include <petscmat.h>

namespace pele {
CVODEBDFOptimizer::CVODEBDFOptimizer(
    std::shared_ptr<pele::BasePotential> potential,
    const pele::Array<double> x0, double tol, double rtol, double atol)
    : GradientOptimizer(potential, x0, tol),
      cvode_mem(CVodeCreate(CV_BDF)), // create cvode memory
      N_size(x0.size()), t0(0), tN(1.0) {
  // dummy t0
  // initialize petsc
  PetscInitializeNoArguments();
  // this assumes that the number of non zeros aren't different
  blocksize = 1;
  hessav = 10;
  // matrix initialization
  // MatCreateSeqSBAIJ(PETSC_COMM_SELF, blocksize, N_size, N_size, hessav, NULL,
  //                   &petsc_jacobian);

  // MatCreateDense(PETSC_COMM_SELF, N_size, N_size, PETSC_DECIDE, PETSC_DECIDE, NULL, &petsc_jacobian);
  VecCreateSeq(PETSC_COMM_SELF, N_size, &petsc_grad);

  // wrap nvector into petsc array
  nvec_grad_petsc = N_VMake_Petsc(petsc_grad);
  VecDuplicate(petsc_grad, &residual);
  
  
  /////// SNES function details
  SNESCreate(PETSC_COMM_SELF, &snes);
  // SNESSetType(snes, SNESNEWTONLS);
  udataptr = &udata;


  // SNESSetJacobian(snes, petsc_jacobian, petsc_jacobian, SNESJacobianWrapper,
  //                 udataptr);

  

  // PETSC VIEWER FORMAT
  PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_DEFAULT,
                             &vf);
  SNESMonitorSet(
      snes,
      (PetscErrorCode(*)(SNES, PetscInt, PetscReal, void *))CVODESNESMonitor,
      vf, (PetscErrorCode(*)(void **))PetscViewerAndFormatDestroy);






  // we need to pass the potential pointer on to the
  // Jacobian function as the application context
  // The potential pointer allows us to use the base potential class to
  // of the gradient i.e the hessian
  t0 = 0;
  std::cout << x0 << "\n";
  Array<double> x0copy = x0.copy();
  // We're going to do a bit of manipulation here
  // because we want the Vec to be owned by
  // the Nvec implementation, not have a wrapper
  PetscVec_eq_pele(x0_petsc, x0);
  x0_N = N_VMake_Petsc(x0_petsc);

  NLS = SUNNonlinSol_PetscSNES(x0_N, snes);


  // Jacobian setting
  MatCreateSNESMF(snes, &petsc_jacobian);
  SNESSetJacobian(snes, petsc_jacobian, petsc_jacobian, MatMFFDComputeJacobian,
                  0);

  
  int ierr = CVodeInit(cvode_mem, gradient_wrapper, t0, x0_N); CHKERRCV(ierr);
  
  std::cout << ierr << "init return value \n";
  


  ierr = CVodeSetNonlinearSolver(cvode_mem, NLS);
  CHKERRCV(ierr);
  // add additonal moniting information

  


  // initialize userdata
  udata.rtol = rtol;
  udata.atol = atol;
  udata.nfev = 0;
  udata.nhev = 0;
  // current_grad = N_VClone_Petsc(nvec_grad_petsc);
  udata.pot_ = potential_;
  udata.neq = x0.size();
  udata.stored_grad = Array<double>(x0.size(), 0);

  
  
  CVodeSStolerances(cvode_mem, udata.rtol, udata.atol);
  ierr = CVodeSetUserData(cvode_mem, &udata);
  CVodeSetMaxNumSteps(cvode_mem, 1000000);
  CVodeSetStopTime(cvode_mem, tN);
  std::cout << "step check"
            << "\n";
  int flag = CVode(cvode_mem, tN, x0_N, &t0, CV_ONE_STEP);
  flag = CVode(cvode_mem, tN, x0_N, &t0, CV_ONE_STEP);  
  CHKERRCV(flag);
  std::cout << "construction happened"
            << "\n";
  VecView(N_VGetVector_Petsc(x0_N), PETSC_VIEWER_STDOUT_SELF);
};

void CVODEBDFOptimizer::one_iteration() {
  /* advance solver just one internal step */
  Array<double> xold = x_;
  VecView(N_VGetVector_Petsc(x0_N), PETSC_VIEWER_STDOUT_SELF);
  int flag = CVode(cvode_mem, tN, x0_N, &t0, CV_ONE_STEP);  CHKERRCV(flag);
  iter_number_ += 1;
  double t;
  CVodeGetCurrentTime(cvode_mem, &t);
  // first derivative
  CVodeGetDky(cvode_mem, t, 1, current_grad);
  double norm2 = N_VDotProd(current_grad, current_grad);
  x_ = pele_eq_N_Vector(x0_N);
  rms_ = (sqrt(norm2 / udata.neq));
  f_ = udata.stored_energy;
  nfev_ = udata.nfev;
  Array<double> step = xold - x_;
};

CVODEBDFOptimizer::~CVODEBDFOptimizer() {
  VecDestroy(&petsc_grad);
  MatDestroy(&petsc_jacobian);
  N_VDestroy(nvec_grad_petsc);
  CVodeFree(&cvode_mem);
  PetscFinalize();
};

/**
 * @brief      Routine to calculate - \grad{V(x)}
 *
 * @details    Calculates -\grad{V(y)} to solve the differential equation ydot =
 * - \grad{V(y)} using CVODE. the potential V is stored as a shared pointer in
 * the user data
 *
 * @param      t: dummy variable for time. Not used
 *
 * @return     return type
 */
int gradient_wrapper(double t, N_Vector y, N_Vector ydot, void *user_data) {

  UserData udata = (UserData)user_data;
  // wrap local vector as a pele vector
  // Array<double> x_pele = pele_eq_PetscVec(N_VGetVector_Petsc(y));
  Vec y_petsc = N_VGetVector_Petsc(y);
  Vec ydot_petsc = N_VGetVector_Petsc(ydot);
  Array<double> g;

  // Vec ydot_petsc = N_VGetVector_Petsc(ydot);
  double energy = udata->pot_->get_energy_gradient_petsc(y_petsc, ydot_petsc);
  
  // routine to reverse sign
  VecScale(ydot_petsc, -1.0);
  // func data reversed
  // udata->stored_grad = (g);
  udata->stored_energy = energy;
  return 0;
}

int Jac(realtype t, N_Vector y, N_Vector fy, SUNMatrix J, void *user_data,
        N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
  UserData udata = (UserData)user_data;

  pele::Array<double> yw = pele_eq_N_Vector(y);
  Array<double> g(yw.size());
  Array<double> h(yw.size() * yw.size());
  udata->pot_->get_energy_gradient_hessian(pele_eq_N_Vector(y), g, h);
  udata->nhev += 1;
  double *hessdata = SUNDenseMatrix_Data(J);
  for (size_t i = 0; i < h.size(); ++i) {
    hessdata[i] = -h[i];
  }
  return 0;
};

/**
 * @brief      Calculates the Jacobian of the negative gradient i.e
 * -\grad{-\grad{V(x)}}
 *
 * @details    Calculates the Jacobian of the negative gradient to solve the
 * differential equation ydot = - \grad{V(y)}. This wrapper for using CVODE with
 * the PETSc SNES solver
 *
 * @param      NLS: Nonlinear SNES solver
 *             x  : coordinates of the particles
 *             Amat: The hessian
 *             Precon: Preonditioner
 *             user_data: userdata
 * @return     PetscErrorCode
 */
PetscErrorCode SNESJacobianWrapper(SNES NLS, Vec x, Mat Amat, Mat Precon,
                                   void *user_data) {
  PetscFunctionBeginUser;
  std::cout << "here"
            << "\n";
  UserData udata = (UserData)user_data;
  udata->pot_->get_hessian_petsc(x, Precon);
  MatScale(Precon, -1.0);
  MatCopy(Precon, Amat, SAME_NONZERO_PATTERN);
  PetscFunctionReturn(0);
};


/**
 * Monitor function attach for debugging
 */
PetscErrorCode CVODESNESMonitor(SNES snes, PetscInt its, PetscReal fnorm,
                                PetscViewerAndFormat *vf) {
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  // monitor function attach
  std::cout << "-------- solution " << "\n";
  // ierr = SNESMonitorSolution(snes, its, fnorm, vf);
  
  std::cout << "------------" << "\n";
  bool check = vf==NULL;
  std::cout << check << "\n";

  ierr = SNESMonitorDefaultShort(snes, its, fnorm, vf);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

} // namespace pele
