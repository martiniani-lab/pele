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

  PetscInitializeNoArguments();


  // The constructor is compartmentalized since there is a lot to setup
  // This function separation should help with debugging
  // also these functions could use parameters to be set differently



  setup_gradient();

  std::cout << "hello" << "\n";


  setup_coords();

  
  setup_cvode_data(rtol, atol);

  std::cout << "SNES Setup" << "\n";
  setup_SNES();


  std::cout << "coordinate setup" << "\n";

  setup_CVODE();

};

void CVODEBDFOptimizer::one_iteration() {  
  /* advance solver just one internal step */
    std::cout << "step no" << iter_number_ << "\n";
    
    
  Array<double> xold = x_;
  int flag = CVode(cvode_mem, tN, x0_N, &t0, CV_ONE_STEP);CHKERRCV_ONE_STEP(flag);
  iter_number_ += 1;
  double t;
  CVodeGetCurrentTime(cvode_mem, &t);

  N_VPrint_Petsc(x0_N);
  // first derivative
  CVodeGetDky(cvode_mem, t, 1, current_grad);
  double norm2 = N_VDotProd(current_grad, current_grad);
  std::cout << "grad:" << norm2 << "\n";
  std::cout << "tol" <<tol_ << "\n";


  x_ = pele_eq_N_Vector(x0_N);
  rms_ = (sqrt(norm2 / udata.neq));
  std::cout << "tol" << rms_ << "\n";
  f_ = udata.stored_energy;
  nfev_ = udata.nfev;
  Array<double> step = xold - x_; 
};


CVODEBDFOptimizer::~CVODEBDFOptimizer() {
    // created in setup grad
  VecDestroy(&petsc_grad);
  N_VDestroy(nvec_grad_petsc);
  N_VDestroy(current_grad);

  // created in setup coords
  VecDestroy(&x0_petsc);
  N_VDestroy(x0_N);

  // SNESfree
  SNESDestroy(&snes);
  
  MatDestroy(&petsc_jacobian);
  CVodeFree(&cvode_mem);
  SUNNonlinSolFree(NLS);
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

  // udata->stored_grad = (g);
  udata->stored_energy = energy;
  udata->nfev += 1;  
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
 * @brief SNESJacobianWrapper
 *
 * @param NLS Nonlinear solver
 * @param dummy dummy variable for position (which we're getting from CVODE)
 * @param Amat 
 * @param Precon Description of Precon
 * @param user_data Description of user_data
 * @return PetscErrorCode
 */
PetscErrorCode SNESJacobianWrapper(PetscReal t, Vec x, Mat J, void * user_data) {
  PetscFunctionBeginUser;
  UserData udata = (UserData)user_data;

  
  udata->pot_->get_hessian_petsc(x, J);

  // I-\gamma(-H) = I + \gamma H
  MatScale(J, -1.0);
  // MatView(Precon, PETSC_VIEWER_STDOUT_SELF);
  // MatView(Amat, PETSC_VIEWER_STDOUT_SELF);
  udata->nhev += 1;

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