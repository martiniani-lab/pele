/*
  UNDER WORK
  Probably need to keep it here since
  it depends on implicit SUNDIALS variables until the corresponding interface
  is built.
*/

/* no more dependencies */

#include "pele/cvode_modified_newton.h"
#include "cvode/cvode.h"
#include <math.h>
#include <petscerror.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscpctypes.h>
#include <petscsnes.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <string.h>
#include <sundials/sundials_types.h>

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
  CVMNPETScMem cvls_petsc_mem;
  cvls_petsc_mem = (CVMNPETScMem)context;
  /* if solution scaling is not used this is a bad idea */
  if (!cvls_petsc_mem->scalesol) {
    PetscFunctionReturn(0);
  }
  PetscErrorCode ierr;
  /* cvode memory */
  CVodeMem cv_mem;
  int retval;
  PetscReal gamma;
  PetscReal t;

  CVodeGetCurrentGamma(cvls_petsc_mem->cvode_mem, &gamma);

  cv_mem = (CVodeMem)(cvls_petsc_mem->cvode_mem);

  /* check jacobian needs to be updated */
  if (cvls_petsc_mem->jok) {
    /* use saved copy of jacobian */
    (cvls_petsc_mem->jcur) = PETSC_FALSE;

    /* Overwrite linear system matrix with saved J
       Assuming different non zero structure */
    /* TODO: expose the NON zero structure usage */
    ierr = MatCopy(cvls_petsc_mem->savedJ, A, DIFFERENT_NONZERO_PATTERN);
    CHKERRQ(ierr);
  } else {
    /* call jac() to update the function */
    (cvls_petsc_mem->jcur) = PETSC_TRUE;
    ierr = MatZeroEntries(A);
    CHKERRQ(ierr);

    /* get current time */
    CVodeGetCurrentTime(cvls_petsc_mem->cvode_mem, &t);
    /* compute new jacobian matrix */

    ierr = cvls_petsc_mem->user_jac_func(t, X, A, cvls_petsc_mem->user_mem);
    CHKERRQ(ierr);
    /* Update saved jacobian copy */
    /* TODO: expose nonzero structure usage */
    /* I'm not sure this makes sense */
    /* TODO: fix savedJ not initalized */
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
PetscErrorCode cvLSPreSolveKSP(KSP ksp, Vec b, Vec x, void *ctx) { return 0; }

/*-----------------------------------------------------------------
  Calls the post solve routine for the KSP Solver to emulate the
  functionality of the cvode solver to rescale the step if
  Here the application context is the cvode memory
  -----------------------------------------------------------------*/
PetscErrorCode cvLSPostSolveKSP(KSP ksp, Vec b, Vec x, void *context) {
  /* storage for petsc memory */
  CVMNPETScMem cvls_petsc_mem;
  cvls_petsc_mem = (CVMNPETScMem)context;
  if (!cvls_petsc_mem->scalesol) {
    PetscFunctionReturn(0);
  }
  CVodeMem cv_mem;
  cv_mem = (CVodeMem) context;
  /* needs changes */
  if (cv_mem->cv_gamrat != ONE) {
    VecScale(b, TWO / (ONE + cv_mem->cv_gamrat));
  }
}

/* allocates extra memory for running the modified newton algorithm using
 * PETSc*/
CVMNPETScMem CVODEMNPETScCreate(void *cvode_mem, void *user_mem,
                                CVSNESJacFn func, booleantype scalesol, Mat Jac,
                                Vec y) {
  CVMNPETScMem content;

  /* allocate memory for content */
  content = (CVMNPETScMem)malloc(sizeof *content);

  /* assign memory pointers */
  content->cvode_mem = cvode_mem;
  content->user_mem = user_mem;
  CVodeMem cv_mem;
  

  /* assign user jacobian function pointer */
  content->user_jac_func = func;

  /* set up initial calculation information */
  content->jok = PETSC_FALSE;
  /* TODO: check whether it needs to be updated */
  content->jcur = PETSC_TRUE;  
  /* assign memory for the saved jacobian */
  /* we may not have to do this */
  /* assign memory for the vectors */
  VecDuplicate(y, &content->ycur);
  VecDuplicate(y, &content->fcur);

  /* whether to scale the solution after the solve or not */
  content->scalesol = scalesol;

  /* initialize saved jacobian */
  MatDuplicate(Jac, MAT_DO_NOT_COPY_VALUES, &content->savedJ);

  return content;
}

/* destructor for cvmnpetscmem */
PetscErrorCode CVODEMNPTScFree(CVMNPETScMem *cvmnpetscmem) {
  if (cvmnpetscmem == NULL) {
    return 0;
  }
  PetscErrorCode ierr;
  /* destroy vectors/matrices */
  ierr = MatDestroy(&((*cvmnpetscmem)->savedJ));
  CHKERRQ(ierr);
  ierr = VecDestroy(&((*cvmnpetscmem)->fcur));
  CHKERRQ(ierr);
  ierr = VecDestroy(&((*cvmnpetscmem)->ycur));
  CHKERRQ(ierr);
  /* Free matrix vectors */
  free(cvmnpetscmem);
  cvmnpetscmem = NULL;
  return 0;
}

/* -----------------------------------------------------------------------------
   Sets up SNES to use the delayed newton approach with SUNNonlinSol
   in SUNDIALS to save Jacobian evaluations for small systems.


   This routine has the following functions
   1. Sets the krylov solver to use the preconditioner only
      which we want to do if we want to use PCLU and PCCholesky
      as direct solvers.
   2. Sets Newton to use no line search. this is important
      considering that our step size modification is integrated
      into the solver itself.
   3. sets up the pre solve and postsolve routines to scale
      the solution appropriately according to stepsize.


   Developer's Note: This hasn't been setup for iterative methods.

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

   TODO: maybe remove the wrapper around the setup by defining a
   wrapped around free function
 * ---------------------------------------------------------------------------*/
PetscErrorCode CVSNESMNSetup(SNES snes, CVMNPETScMem cvmnmem, Mat Jac) {
  PetscFunctionBegin;
  KSP ksp;
  PC pc;
  PCType pc_type;
  /* get all necessary contexts */
  SNESGetKSP(snes, &ksp);
  KSPGetPC(ksp, &pc);
  PCGetType(pc, &pc_type);

  /* force the solver to be just use the preconditioner only */
  KSPSetType(ksp, KSPPREONLY);

  if (pc_type == PETSC_NULL) {
    PCSetType(pc, PCLU);
    printf("pc set type \n");
    PCGetType(pc, &pc_type);
  }

  /* If the Preconditioner is not LU or Cholesky, return no support exit code */
  if (!(strcmp(pc_type, PCLU) || strcmp(pc_type, PCCHOLESKY))) {
    PetscFunctionReturn(PETSC_ERR_SUP);
  }

  KSPSetPreSolve(ksp, cvLSPreSolveKSP, (void *)cvmnmem);
  KSPSetPostSolve(ksp, cvLSPostSolveKSP, (void *)cvmnmem);

  /* pass the wrapped jacobian function on to SNES */
  SNESSetJacobian(snes, Jac, Jac, CVDelayedJSNES, cvmnmem);

  /* Set the jacobian function */
  PetscFunctionReturn(0);
}