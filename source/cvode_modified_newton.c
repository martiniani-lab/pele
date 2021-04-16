/*
  UNDER WORK
  Probably need to keep it here since
  it depends on implicit SUNDIALS variables until the corresponding interface
  is built.
*/

/* no more dependencies */

#include "pele/cvode_modified_newton.h"
#include "cvode/cvode.h"
#include "nvector/nvector_petsc.h"

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
#include <stdio.h>
#include <string.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>

/* private macros */
#define ONE RCONST(1.0)
#define TWO RCONST(2.0)

#define CRDOWN RCONST(0.3) /* convergence rate estimate */
#define RDIV RCONST(2.0)   /* declare divergence if del/delp > RDIV */
#define NLS_MAXCOR 3       /* Maximum number of corrector iterations */
/*****************************************************************************/
/*                                General functions                          */
/*****************************************************************************/

/*-----------------------------------------------------------------
  Wrapper to imitate what the linear system solver does in
  cvLsLinSys with what the Jacobian wrapper does in Petsc.

  Calculates a delayed J_{snes}(delta_x_res) = I - \gamma J_{cvode}(x(0) +
  delta_x_res)
  (
  See equation 10.5, 10.6 and 10.7 in the CVODE manual
  )

  Warning: In sundials language the Jacobian is the gradient of
  the RHS of the ODE,

  dx/dt = f(x, t)

  to be solved. This is the language we're
  using here. For SNES however the Jacobian is the gradient of
  the LHS of the Nonlinear system

  F(x)=b

  This difference in language is important to keep in mind. since
  this function is called by SNESSetJacobian which refers the
  Jacobian in SNES language. We also have to note that these jacobians
  are evaluated at /different points/


  Both of these are related by

  A = J_{SNES}(delta_x_res) = I-\gamma J_{cvode}(x_shifted)

  where x_shifted = x_0 + x_res. x_0 is the initial guess of the ODE solver
  and x_shifted is done at a different value.

  -----------------------------------------------------------------*/
PetscErrorCode CVDelayedJSNES(SNES snes, Vec delta_x_res, Mat A, Mat Jpre,
                              void *context) {
  /* storage for petsc memory */

  CVMNPETScMem cvls_petsc_mem;
  SNESGetApplicationContext(snes, (void **)&cvls_petsc_mem);
  /* if solution scaling is not used this is a bad idea */
  if (!cvls_petsc_mem->scalesol) {
    return 0;
  }

  PetscErrorCode ierr;
  /* cvode memory */
  CVodeMem cv_mem;
  int retval;
  PetscReal gamma;
  PetscReal t;
  N_Vector x_n;
  Vec x_n_petsc;

  int steps;
  CVodeGetNumSteps(cvls_petsc_mem->cvode_mem, &steps);
  CVodeGetCurrentState(cvls_petsc_mem->cvode_mem, &x_n);
  CVodeGetCurrentGamma(cvls_petsc_mem->cvode_mem, &gamma);
  x_n_petsc = N_VGetVector_Petsc(x_n);
  int cvode_steps;

  cv_mem = (CVodeMem)(cvls_petsc_mem->cvode_mem);
  /* check jacobian needs to be updated */
  if (cvls_petsc_mem->jok) {
    /* use saved copy of jacobian */
    (cvls_petsc_mem->jcur) = PETSC_FALSE;
    /* Overwrite linear system matrix with saved J
       Assuming different non zero structure */
    /* TODO: expose the NON zero structure usage */
    /* ierr = MatCopy(cvls_petsc_mem->savedJ, A, DIFFERENT_NONZERO_PATTERN); */
    A = cvls_petsc_mem->savedJ;
    CVodeGetCurrentState(cvls_petsc_mem->cvode_mem, &x_n);

    /* Perform Gamma calculations */

    CHKERRQ(ierr);
  } else {
    /* call jac() to update the function */
    (cvls_petsc_mem->jcur) = PETSC_TRUE;
    ierr = MatZeroEntries(A);
    CHKERRQ(ierr);
    /* compute new jacobian matrix */

    /* TODO: add NULL checks for error handling */
    ierr = cvls_petsc_mem->user_jac_func(t, x_n_petsc, A,
                                         cvls_petsc_mem->user_mem);
    /* Update saved jacobian copy */
    /* TODO: expose nonzero structure usage */
    /* I'm not sure this makes sense */
    /* TODO: fix savedJ not initalized */
    /* MatCopy(A, cvls_petsc_mem->savedJ, DIFFERENT_NONZERO_PATTERN); */
    cvls_petsc_mem->savedJ = A;
  }


  /* do A = I - \gamma J */
  ierr = MatScale(A, -gamma);
  CHKERRQ(ierr);
  ierr = MatShift(A, 1.0);
  CHKERRQ(ierr);
  return 0;
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
  cv_mem = (CVodeMem)cvls_petsc_mem->cvode_mem;

  if (cv_mem->cv_gamrat != ONE) {
    VecScale(b, TWO / (ONE + cv_mem->cv_gamrat));
  }

  /* Tell the solver that the convergence test hasn't been called for this step */
  cvls_petsc_mem->ctest_called = PETSC_FALSE;

  return 0;
}

/* allocates extra memory for running the modified newton algorithm using
 * PETSc*/

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
  ierr = VecDestroy(&((*cvmnpetscmem)->yguess));
  CHKERRQ(ierr);

  N_VDestroy_Petsc(((*cvmnpetscmem)->yguess_nvec));
  N_VDestroy_Petsc(((*cvmnpetscmem)->delta));

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
   #TODO attaching func is duplicated
 * ---------------------------------------------------------------------------*/
PetscErrorCode CVSNESMNSetup(SNES snes, CVMNPETScMem cvmnmem, Mat Jac_mat,
                             CVSNESJacFn func, void *user_mem, void *cvode_mem,
                             Vec y, booleantype scalesol) {
    PetscErrorCode ierr;
  /* PetscFunctionBegin; */
  if (cvmnmem == NULL) {
    cvmnmem = (CVMNPETScMem)malloc(sizeof *cvmnmem);
  }
  CVodeMem cv_mem;
  SNESSetType(snes, SNESNEWTONLS);
  KSP ksp;
  PC pc;
  PCType pc_type;
  SNESLineSearch snes_linesearch;
  SNESGetKSP(snes, &ksp);
  KSPGetPC(ksp, &pc);
  PCGetType(pc, &pc_type);
  PCSetType(pc, PCCHOLESKY);
  SNESGetLineSearch(snes, &snes_linesearch);
  ierr = SNESLineSearchSetType(snes_linesearch, SNESLINESEARCHSHELL);
  printf("ierrrr %d \n", ierr);
  ierr = SNESLineSearchShellSetUserFunc(snes_linesearch, SNESLineSearchApply_CVODE, cvmnmem);
  printf("ierrrr %d \n", ierr);
  
  
  cvmnmem->jok = PETSC_FALSE;
  /* TODO: check whether it needs to be updated */
  cvmnmem->jcur = PETSC_TRUE;
  /* assign memory for the saved jacobian */
  MatDuplicate(Jac_mat, MAT_DO_NOT_COPY_VALUES, &cvmnmem->savedJ);
  /* we may not have to do this */
  /* assign memory for the vectors */
  VecDuplicate(y, &cvmnmem->ycur);
  VecDuplicate(y, &cvmnmem->fcur);

  /* Initialize guess data */
  VecDuplicate(y, &cvmnmem->yguess);
  cvmnmem->yguess_nvec = N_VMake_Petsc(cvmnmem->yguess);

  /* Assign residue data */
  Vec delta_petsc;
  SNESGetSolutionUpdate(snes, &delta_petsc);
  cvmnmem->delta = N_VMake_Petsc(y);

  /* whether to scale the solution after the solve or not */
  cvmnmem->scalesol = scalesol;
  /* get all necessary contexts */


  /* force the solver to be just use the preconditioner only */
  KSPSetType(ksp, KSPPREONLY);
  if (pc_type == PETSC_NULL) {
    PCSetType(pc, PCLU);
    PCGetType(pc, &pc_type);
  }
  /* set up memory for saved J */

  /* If the Preconditioner is not LU or Cholesky, return no support exit code */
  if (!(strcmp(pc_type, PCLU) || strcmp(pc_type, PCCHOLESKY))) {
    PetscFunctionReturn(PETSC_ERR_SUP);
  }

  KSPSetPreSolve(ksp, cvLSPreSolveKSP, (void *)cvmnmem);
  KSPSetPostSolve(ksp, cvLSPostSolveKSP, (void *)cvmnmem);

  /* Set tolerances note that only evaluations and steps matter since
     we have our own convergence test */
  SNESSetTolerances(snes, PETSC_DEFAULT, PETSC_DEFAULT,PETSC_DEFAULT, 3, 10);
  
  

  /* pass the wrapped jacobian function on to SNES */
  SNESSetJacobian(snes, Jac_mat, Jac_mat, CVDelayedJSNES, cvmnmem);
  /* TODO: make this the only approach */
  cvmnmem->user_mem = user_mem;
  cvmnmem->user_jac_func = func;
  SNESSetApplicationContext(snes, cvmnmem);
  SNESGetApplicationContext(snes, (void **)&cvmnmem);

  /* convergence test */
  SNESSetConvergenceTest(snes, CVodeConvergenceTest, NULL, NULL);

  /* Norm schedule */


  /* SNESGetLineSearch(snes, &snes_linesearch); */
  SNESLineSearchSetComputeNorms(snes_linesearch, PETSC_FALSE);



  /* defaults to no line search*/


  /* Make sure a lagged jacobian is used */
  SNESSetLagJacobianPersists(snes, PETSC_TRUE);
  SNESSetLagJacobian(snes, -2);

  /* assign memory pointers */
  cvmnmem->user_mem = user_mem;
  cvmnmem->user_jac_func = func;
  /* Set the jacobian function */
  /* PetscFunctionReturn(0); */
}

/* Private function */

/**
 * @brief      Reproduces CV convergence test from CVODE
 *
 * @details    Details are borrowed from cvNlsConvTest private function in cvode. Note: this function is not self contained.
 *             a second call to this outside the loop messes up the calculation. TODO: change this behavior
 *
 * @param      snes SNES object (should contain context to CVMNPETScMem)
 *             (NORMS are not calculated so they should be undefined)
 *             user defined context is NULL
 *
 * @return     SNESConvergedReason NOTE: THis is markedly different so we
               identify this into three paradigms (recoverable convergence
 failures, non recoverable
 */
PetscErrorCode CVodeConvergenceTest(SNES snes, PetscInt it, PetscReal xnorm,
                                    PetscReal gnorm, PetscReal fnorm,
                                    SNESConvergedReason *reason, void *cctx) {

  CVMNPETScMem cvls_petsc_mem;
  SNESGetApplicationContext(snes, (void **)&cvls_petsc_mem);


  /* checks */
  if (cvls_petsc_mem->ctest_called) {
      *reason = cvls_petsc_mem->c_reason;
      PetscFunctionReturn(0);
  }
  cvls_petsc_mem->ctest_called = PETSC_TRUE;

  PetscInt m, retval;
  PetscReal del;
  PetscReal dcon;

  Vec delta_petsc;
  KSP ksp;

  SNESGetKSP(snes, &ksp);

  /* Need to get some of the data from CVode */
  /* The calculations for the convergence test are done in cvNlsResidual */
  CVodeMem cv_mem;
  cv_mem = cvls_petsc_mem->cvode_mem;

  /* see how these change */
  N_Vector yextra = cv_mem->cv_zn[0];

  /* Get initial guess from earlier */
  N_Vector ycor = cvls_petsc_mem->yguess_nvec;

  N_Vector delta = cvls_petsc_mem->delta;
  /* obtain arguments from our SNESconverged */
  SNESGetSolutionUpdate(snes, &delta_petsc);

  N_VSetVector_Petsc(delta, delta_petsc);


  
  /* N_Vector delta = cvls_petsc_mem->y; */
  N_Vector ewt = cvls_petsc_mem->w;
  PetscReal tol = cvls_petsc_mem->tol;

  /* KSPGetSolution(ksp, &delta_petsc); */
  /* SNESGetSolutionUpdate(snes, &delta_petsc); */
  
  N_VLinearSum(ONE, ycor, ONE, delta, ycor);

  /* /\* TODO: Ideally this line should be in KSP *\/ */
  del = N_VWrmsNorm(delta, ewt);

  *reason = SNES_CONVERGED_ITERATING;
  cvls_petsc_mem->c_reason = SNES_CONVERGED_ITERATING;
  /* (Important) Prevents convergence from happening before getting to KSP
     KSP has a convergence test before the problem gets solved*/

  SNESGetLinearSolveIterations(snes, &m);
  if (m == 0) {
    return 0;
  }

  /* get the current nonlinear solver iteration count */
  SNESGetLinearSolveIterations(snes, &m);

  /* Test for convergence. If m > 1, an estimate of the convergence
     rate constant is stored in crate, and used in the test.        */
  if (m > 1) {
    cv_mem->cv_crate = SUNMAX(CRDOWN * cv_mem->cv_crate, del / cv_mem->cv_delp);
  }
  dcon = del * SUNMIN(ONE, cv_mem->cv_crate) / tol;

  if (dcon <= ONE) {
    /* If this is the first solve */
    if (m == 1) {
      cv_mem->cv_acnrm = del;
    } else {
      cv_mem->cv_acnrm = N_VWrmsNorm(ycor, ewt);
    }

    cv_mem->cv_acnrmcur = SUNTRUE;
    *reason = SNES_CONVERGED_FNORM_ABS; /* Placeholder for all convergences */
    cvls_petsc_mem->c_reason = *reason;
    return (0); /* Nonlinear system was solved successfully */
  }

  /* check if the iteration seems to be diverging */
  if ((m > 1) && (del > RDIV * cv_mem->cv_delp)) {
    *reason = SNES_DIVERGED_FUNCTION_DOMAIN;
    cvls_petsc_mem->c_reason = *reason;
    return (0);
  }

  /* Save norm of correction and loop again */
  cv_mem->cv_delp = del;


  /* Not yet converged */
  return (0);
}


/*
  Line search for CVODE
*/
PetscErrorCode  SNESLineSearchApply_CVODE(SNESLineSearch linesearch, void *ctx)
{
  PetscBool      changed_y, changed_w;
  PetscErrorCode ierr;
  Vec            X, F, Y, W;
  SNES           snes;
  PetscReal      gnorm, xnorm, ynorm, lambda;
  PetscBool      domainerror;

  /* convergence test data */
  PetscFunctionBegin;
  ierr = SNESLineSearchGetVecs(linesearch, &X, &F, &Y, &W, NULL);CHKERRQ(ierr);
  ierr = SNESLineSearchGetLambda(linesearch, &lambda);CHKERRQ(ierr);
  ierr = SNESLineSearchGetSNES(linesearch, &snes);CHKERRQ(ierr);
  ierr = SNESLineSearchSetReason(linesearch, SNES_LINESEARCH_SUCCEEDED);CHKERRQ(ierr);

  /* dummy info */
  SNESConvergedReason reason;
  void *cctx;
  PetscInt it;
  
  /* perform update */
  ierr = VecWAXPY(W,-lambda,Y,X);CHKERRQ(ierr);
  /* copy the solution over */

  
  ierr = VecCopy(W, X);CHKERRQ(ierr);



  /* printf("convergence reason: %d \n", reason); */
 
  /* THe custom line search skips a function calculation /if/ the function has already converged
     in which case we don't need to recalculate the function */
  ierr = CVodeConvergenceTest(snes, it, xnorm, gnorm, ynorm, &reason, cctx); CHKERRQ(ierr);

  if (!reason) {
      ierr = SNESComputeFunction(snes, X,F);CHKERRQ(ierr);
  }
  
  /* if it hasn't converged yet then calculate the gradient */





  PetscFunctionReturn(0);
}
