/*
  UNDER WORK
  Probably need to keep it here since
  it depends on implicit SUNDIALS variables until the corresponding interface
  is built.
*/

/* no more dependencies */

#include "cvode/cvode.h"
#include <petscsnes.h>
#include <petscsystypes.h>
#include <sundials/sundials_types.h>
#include "pele/cvode_modified_newton.h"

/* private macros */
#define ONE RCONST(1.0)
#define TWO RCONST(2.0)

/*****************************************************************************/
/*                                General functions                          */
/*****************************************************************************/

/*-----------------------------------------------------------------
  Wrapper to imitate what the linear system solver does in
  cvLsLinSys with what the Jacobian wrapper does in Petsc.

  Calculates a delayed J_{snes} = I - \gamma J_{cvode}

  Warning: In sundials language the Jacobian is the gradient of
  the RHS of the ODE,

  dx/dt = f(x, t)

  to be solved. This is the language we're
  using here. For SNES however the Jacobian is the gradient of
  the LHS of the Nonlinear system

  F(x)=b

  This difference in language is important to keep in mind. since
  this function is called by SNESSetJacobian which refers the
  Jacobian in SNES language.
  Both of these are related by A = J_{SNES} = I-\gamma J_{cvode}
  -----------------------------------------------------------------*/
PetscErrorCode CVDelayedJSNES(SNES snes, Vec X, Mat A, Mat Jpre,
                              void *context) {
  PetscFunctionBegin;
  /* storage for petsc memory */
  CVLSPETScMem cvls_petsc_mem;
  cvls_petsc_mem = (CVLSPETScMem)context;
  /* if solution scaling is not used this is a bad idea */
  if (!cvls_petsc_mem->scalesol) {
    PetscFunctionReturn(0);
  }
  PetscErrorCode ierr;
  /* cvode memory */
  CVodeMem cv_mem;
  int retval;
  PetscReal gamma;

  CVodeGetCurrentGamma(cvls_petsc_mem->cvode_mem, &gamma);

  cv_mem = (CVodeMem)(cvls_petsc_mem->cvode_mem);

  /* check jacobian needs to be updated */
  if (cvls_petsc_mem->jok) {
    /* use saved copy of jacobian */
    *(cvls_petsc_mem->jcur) = PETSC_FALSE;

    /* Overwrite linear system matrix with saved J
       Assuming different non zero structure */
    /* TODO: expose the NON zero structure usage */
    ierr = MatCopy(cvls_petsc_mem->savedJ, A, DIFFERENT_NONZERO_PATTERN);
    CHKERRQ(ierr);
  } else {
    /* call jac() to update the function */
    *(cvls_petsc_mem->jcur) = PETSC_TRUE;
    ierr = MatZeroEntries(A);
    CHKERRQ(ierr);

    /* compute new jacobian matrix */
    ierr = cvls_petsc_mem->user_jac_func(cvls_petsc_mem->t, X, A,
                                         cvls_petsc_mem->user_mem);
    CHKERRQ(ierr);
    /* Update saved jacobian copy */
    /* TODO: expose nonzero structure usage */
    /* I'm not sure this makes sense */
    MatCopy(A, cvls_petsc_mem->savedJ, DIFFERENT_NONZERO_PATTERN);
  }
  /* do A = I - \gamma J */
  ierr = MatScale(A, -gamma);
  CHKERRQ(ierr);
  ierr = MatShift(A, 1.0);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*-----------------------------------------------------------------
 Pre solve KSP routine. The goal of this function is to emulate what
 is happening before cvLsSolve in the sundials routine
 this should be passed on to KSP from KSPsetpresolve

 should probably call set ksp set reuse preconditioner though
 lets do this change once we get the normal version running

 TODO: THis is not necessary for us since we're planning to go with a
 solver. but iterative solvers need this set up
  -----------------------------------------------------------------*/
PetscErrorCode cvLSPresolveKSP(KSP ksp, Vec b, Vec x, void *ctx) { return 0; }

/*-----------------------------------------------------------------
  Calls the post solve routine for the KSP Solver to emulate the
  functionality of the cvode solver to rescale the step if
  Here the application context is the cvode memory
  -----------------------------------------------------------------*/
PetscErrorCode cvLSPostSolveKSP(KSP ksp, Vec b, Vec x, void *context) {
  /* storage for petsc memory */
  CVLSPETScMem cvls_petsc_mem;
  cvls_petsc_mem = (CVLSPETScMem)context;
  if (!cvls_petsc_mem->scalesol) {
    PetscFunctionReturn(0);
  }

  CVodeMem cv_mem;
  cv_mem = (CVodeMem)context;

  if (cv_mem->cv_gamrat != ONE) {
    VecScale(b, TWO / (ONE + cv_mem->cv_gamrat));
  }
}
