#ifndef _PELE_CVODE_OPT_H__
#define _PELE_CVODE_OPT_H__

#include "array.hpp"
#include "base_potential.hpp"
#include "debug.hpp"

#include "array.hpp"
#include "base_potential.hpp"
#include "debug.hpp"
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

// Eigen linear algebra library
#include "eigen_interface.hpp"
#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

// line search methods
#include "backtracking.hpp"
#include "bracketing.hpp"
#include "linesearch.hpp"
#include "more_thuente.hpp"
#include "nvector/nvector_petsc.h"
#include "nwpele.hpp"
#include "optimizer.hpp"
#include "petsc_interface.hpp"

#include <petscmat.h>
#include <sunnonlinsol/sunnonlinsol_petscsnes.h>

#include "cvode/cvode_proj.h"
#include "petscksp.h"
#include "petscpc.h"
#include "petscpctypes.h"
#include "petscsnes.h"
#include "petscsys.h"
#include "petscsystypes.h"
#include "petscvec.h"
#include "sundials/sundials_linearsolver.h"
#include "sundials/sundials_matrix.h"
#include "sundials/sundials_nonlinearsolver.h"
#include "sundials/sundials_nvector.h"

#include <cvode/cvode.h>               /* access to CVODE                 */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix       */

/**
 * Checks sundials error and appropriately throws exception
 */
#define CHKERRCV(value)                                                        \
  {                                                                            \
    if (value != 0) {                                                          \
      throw std::runtime_error(                                                \
          "Sundials function returned the wrong exit code");                   \
    }                                                                          \
  }

/**
 * Checks sundials error for our single step case and appropriately throws
 * exception Ignores exception value of 1. since t for us is only a parameter
 */

#define CHKERRCV_ONE_STEP(value)                                               \
  {                                                                            \
    if (value != 0 && value != 1) {                                            \
      throw std::runtime_error(                                                \
          "Sundials single step returned the wrong value");                    \
    }                                                                          \
  }

extern "C" {
#include "xsum.h"
}

namespace pele {

int gradient_wrapper(double t, N_Vector y, N_Vector ydot, void *user_data);
int Jac(realtype t, N_Vector y, N_Vector fy, SUNMatrix J, void *user_data,
        N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);
static int Jac2(realtype t, N_Vector y, N_Vector fy, SUNMatrix J,
                void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);
static int f2(realtype t, N_Vector y, N_Vector ydot, void *user_data);

/**
 * @brief      Wrapper for CVODE to get the Jacobian for the Newton Calculation
 *
 * @details    Calculates I - \gamma J where J = -H and H is the hessian. Gamma
 *             is obtained from the User content
 *
 * @param      SNES NLS: Nonlinear solver
 *                Vec x: Coordinate at which the Jacobian is calculated
 *                Mat Amat: Matrix calculation
 *                Mat Precon: Preconditioner: since we're not using any, the
 *                            matrix is assumed the same as Amat.
 *                            WARNING: this is not explicitly set
 *                void *user_data: user context. Needs cvode_mem
 *
 * @return     void *
 */
PetscErrorCode SNESJacobianWrapper(SNES NLS, Vec x, Mat Amat, Mat Precon,
                                   void *user_data);
PetscErrorCode CVODESNESMonitor(SNES snes, PetscInt its, PetscReal fnorm,
                                PetscViewerAndFormat *vf);

/**
 * user data passed to CVODE
 */
typedef struct UserData_ {
  void *cvode_mem_ptr; // reference to CVODE for getting Gamma in Jacobian
  double rtol;         /* integration tolerances */
  double atol;
  size_t nfev;               // number of gradient(function) evaluations
  size_t nhev;               // number of hessian (jacobian) evaluations
  double stored_energy = 0;  // stored energy
  Array<double> stored_grad; // stored gradient. need to pass this on to

  Vec last_gradient;
  std::shared_ptr<pele::BasePotential> pot_;
  size_t neq; // number of equations set so that we don't have to keep setting
              // this over and over again
} * UserData;

// /**
//  * Data that helps determine the context in which petsc can calculate a
//  hessian
//  * Contains
//  * 1. shared pointer to base potential
//  * 2. reference to coords array
//  */
// typedef struct hessdata_
// {

// } * hessdata;

/**
 * Not exactly an optimizer but solves for the differential equation $ dx/dt = -
 * \grad{V(x)} $ to arrive at the trajectory to the corresponding minimum
 */
class CVODEBDFOptimizer : public GradientOptimizer {
private:
  UserData_ udata;
  void *cvode_mem; /* CVODE memory         */
  size_t N_size;
  SUNMatrix A;
  SUNLinearSolver LS;
  SUNNonlinearSolver NLS;
  double t0;
  double tN;
  void *udataptr;
  // These are kept together
  // Since they should point to the same data
  N_Vector x0_N;
  Vec x0_petsc;


  Array<double> xold;
  PetscViewerAndFormat *vf;
  // sparse calculation initializers
  Mat petsc_jacobian;
  Vec petsc_grad;
  N_Vector current_grad;
  N_Vector nvec_grad_petsc;
  Vec residual;
  PetscInt blocksize;
  // average number of non zeros per block for memory allocation purposes
  PetscInt nz_hess;

  // corresponding SNES, KSP and preconditioner nz_hesss
  SNES snes;
  KSP ksp;
  PC pc;

  ///////////////////////////////////////////////////////////////////////////
  //               functions that are part of the constructor              //
  ///////////////////////////////////////////////////////////////////////////

  /**
   * helper function to set up gradient and current gradient in the constructor
   */
  inline void setup_gradient() {

    VecCreateSeq(PETSC_COMM_SELF, N_size, &petsc_grad);
    nvec_grad_petsc = N_VMake_Petsc(petsc_grad);
    current_grad = N_VClone_Petsc(nvec_grad_petsc);
  }
  /**
   * helper function to set up coordinates in the constructor
   */
  inline void setup_coords() {
    // this should only be called in the constructor
    // since that is where x_=x0
    PetscVec_eq_pele(x0_petsc, x_);
    x0_N = N_VMake_Petsc(x0_petsc);
  }

  /**
   * helper function to set up coordinates in the constructor
   */
  inline void setup_cvode_data(double rtol, double atol) {
    // shouldn't need this line
    udataptr = &udata;
    // initialize userdata
    udata.rtol = rtol;
    udata.atol = atol;
    udata.nfev = 0;
    udata.nhev = 0;
    udata.pot_ = potential_;
    udata.neq = N_size;
    udata.cvode_mem_ptr = cvode_mem;
    // TODO: remove this
    udata.stored_grad = Array<double>(N_size, 0);
  }
  /**
   * Helper function to set up SNES wrapped into Petsc
   */
  inline void setup_SNES() {
    SNESCreate(PETSC_COMM_SELF, &snes);

    // // PETSC VIEWER FORMAT should make monitor setting optional
    PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_DEFAULT,
                               &vf);
    SNESMonitorSet(
        snes,
        (PetscErrorCode(*)(SNES, PetscInt, PetscReal, void *))CVODESNESMonitor,
        vf, (PetscErrorCode(*)(void **))PetscViewerAndFormatDestroy);

    SNESGetKSP(snes, &ksp);
    KSPGetPC(ksp, &pc);

    
    N_VPrint_Petsc(x0_N);
    NLS = SUNNonlinSol_PetscSNES(x0_N, snes);

    // // matrix approach
    PCSetType(pc, PCCHOLESKY);
    setup_Jacobian();
    SNESSetJacobian(snes, petsc_jacobian, petsc_jacobian, SNESJacobianWrapper,
                    &udata);

    // // Matrix free approach
    // MatCreateSNESMF(snes, &petsc_jacobian);
    // SNESSetJacobian(snes, petsc_jacobian, petsc_jacobian,
    // MatMFFDComputeJacobian,
    //                 0);
  };

  /**
   * Helper function to set up Jacobian. should be within setup_SNES
   */
  inline void setup_Jacobian() {
    // non zero structure for sparse matrices
    blocksize = 1;
    nz_hess = 16;

    MatCreateDense(PETSC_COMM_SELF, N_size, N_size, PETSC_DECIDE, PETSC_DECIDE, NULL, &petsc_jacobian);
    // MatCreateSeqSBAIJ(PETSC_COMM_SELF, blocksize, N_size, N_size, nz_hess, NULL,
    //                   &petsc_jacobian);
  }

  /**
   * CVODE setup needs to be called after SNES
   */
  inline void setup_CVODE() {

    int ierr = CVodeInit(cvode_mem, gradient_wrapper, t0, x0_N);
    CHKERRCV(ierr);

    ierr = CVodeSetNonlinearSolver(cvode_mem, NLS);
    CHKERRCV(ierr);
    ierr = CVodeSStolerances(cvode_mem, udata.rtol, udata.atol);
    CHKERRCV(ierr);
    ierr = CVodeSetUserData(cvode_mem, &udata);
    CHKERRCV(ierr);

    // should be infty tbh
    CVodeSetMaxNumSteps(cvode_mem, 1000000);
    CVodeSetStopTime(cvode_mem, tN);
  };

public:
  void one_iteration();
  // int f(realtype t, N_Vector y, N_Vector ydot, void *user_data);
  // static int Jac(realtype t, N_Vector y, N_Vector fy, SUNMatrix J,
  //                void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector
  //                tmp3);
  CVODEBDFOptimizer(std::shared_ptr<pele::BasePotential> potential,
                    const pele::Array<double> x0, double tol = 1e-5,
                    double rtol = 1e-4, double atol = 1e-4);
  ~CVODEBDFOptimizer();
  inline int get_nhev() const { return udata.nhev; }
  /**
   * helper functions to compartmentalize the constructor
   */
protected:
  double H02;
};

} // namespace pele
#endif