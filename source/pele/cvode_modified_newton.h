#ifndef _CVODE_PETSC_MN_H
#define _CVODE_PETSC_MN_H

#include "nvector/nvector_petsc.h"
#include <petscksp.h>
#include <petscmat.h>

#include <cvode/cvode.h>
#include <petscsnes.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <sundials/sundials_math.h>
#include <sundials/sundials_nonlinearsolver.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>

#ifdef __cplusplus /* wrapper to enable C++ usage */
extern "C" {
#endif

/*
  UNDER WORK
  Probably need to keep it here since
  it depends on implicit SUNDIALS variables until the corresponding interface
  is built.
*/

/* TODO remove this*/
#include "/home/praharsh/Dropbox/research/bv-libraries/sundials/src/cvode/cvode_impl.h"

/*****************************************************************************/
/*                                   General structs                         */
/*****************************************************************************/

/* User supplied Jacobian function prototype */
typedef PetscErrorCode (*CVSNESJacFn)(PetscReal t, Vec x, Mat J,
                                      void *user_data);

/*
  context passed on to the delayed hessian. TODO: encapsulate this data
  structure
*/
typedef struct {
  /* memory information */
  void *cvode_mem; /* cvode memory */

  /* TODO: remove this since user data is in cvode_mem */
  void *user_mem; /* user data */

  /* TODO */
  CVSNESJacFn user_jac_func; /* user defined Jacobian function */

  /* jacobian calculation information */
  PetscBool jcur; /* whether to use saved copy of jacobian */
  /* Linear solver, matrix and vector objects/pointers */
  /* NOTE: some of this might be uneccessary since it maybe stored
     in the KSP solver */
  Mat savedJ;         /* savedJ = old Jacobian                        */
  Vec ycur;           /* CVODE current y vector in Newton Iteration   */
  Vec fcur;           /* fcur = f(tn, ycur)                           */
  PetscReal gammap;   /* Previous gamma */
  PetscReal gammarat; /* ratio of previous gamma to current gamma */
  booleantype
      scalesol; /* exposed to user (Check delayed matrix versions later)*/
  PetscBool x;

  PetscBool
      recoverable; /* boolean that tells us whether the error is recoverable */

  /* Convergence test data to be passed */
  /* These are only pointers */
  N_Vector delta; /* pointer for the residue to be passed */
  N_Vector w;
  PetscReal tol;
  PetscInt m_local; /* nonlinear solver iterations */

  Vec yguess; /* Initial guess storage (HAS to be initialized) */
  N_Vector
      yguess_nvec; /* pointer to initial guess to save reallocating memory */
  PetscBool ctest_called;       /* whether convergence test has been called */
  SNESConvergedReason c_reason; /* convergence reason if test passes */
} * CVMNPETScMem;

/*****************************************************************************/
/*                           function declarations                           */
/*****************************************************************************/

/* Main Jacobian to be passed to SNES for delayed jacobian calculation */
PetscErrorCode CVDelayedJSNES(SNES snes, Vec X, Mat A, Mat Jpre, void *context);

/* presolve function for the KSP solver KSPPreOnly could do this for the
 * conditioner */
PetscErrorCode cvLSPresolveKSP(KSP ksp, Vec b, Vec x, void *ctx);

/* postsolve function for the KSP solver */
PetscErrorCode cvLSPostSolveKSP(KSP ksp, Vec b, Vec x, void *context);

/* Memory setup for modified newton */
/* TODO: break up the setup */
PetscErrorCode CVODEMNPETScCreate(void *cvode_mem, void *user_mem,
                                  CVSNESJacFn func, booleantype scalesol,
                                  Mat Jac, Vec y, CVMNPETScMem content);

/* destructor */
PetscErrorCode CVODEMNPTScFree(CVMNPETScMem *cvmnpetscmem);

/* SetUp */
PetscErrorCode CVSNESMNSetup(SNES snes, CVMNPETScMem *cvmnmem_ptr, Mat Jac_mat,
                             CVSNESJacFn func, void *user_mem, void *cvode_mem,
                             Vec y, booleantype scalesol);

/* convergence test written with for SNES */
PetscErrorCode CVodeConvergenceTest(SNES snes, PetscInt it, PetscReal xnorm,
                                    PetscReal gnorm, PetscReal f,
                                    SNESConvergedReason *reason, void *cctx);

/*
  SNES shell line search
*/
PetscErrorCode SNESLineSearchApply_CVODE(SNESLineSearch linesearch, void *ctx);

/* dummy linear solver to tell cvode to calculate constants for the linear
 * solver */
int Lsolve_dummy(struct CVodeMemRec *cv_mem, int convfail, N_Vector ypred,
                 N_Vector fpred, booleantype *jcurPtr, N_Vector vtemp1,
                 N_Vector vtemp2, N_Vector vtemp3);

#ifdef __cplusplus /* wrapper to enable C++ usage */
}
#endif

#endif /* end header guard */
