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
#include <sundials/sundials_types.h>

/* private macros */
#define ONE RCONST(1.0)
#define TWO RCONST(2.0)

#define CRDOWN RCONST(0.3)      /* convergence rate estimate */
#define RDIV RCONST(2.0)        /* declare divergence if del/delp > RDIV */
#define NLS_MAXCOR 3            /* Maximum number of corrector iterations */
/*****************************************************************************/
/*                                General functions                          */
/*****************************************************************************/

/*-----------------------------------------------------------------
  Wrapper to imitate what the linear system solver does in
  cvLsLinSys with what the Jacobian wrapper does in Petsc.

  Calculates a delayed J_{snes}(delta_x_res) = I - \gamma J_{cvode}(x(0) + delta_x_res)
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
  CVMNPETScMem cvls_petsc_mem_2;
  SNESGetApplicationContext(snes, (void **) &cvls_petsc_mem);
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

  
  CVodeGetCurrentState(cvls_petsc_mem->cvode_mem, &x_n);
  printf("jok : %d \n", cvls_petsc_mem->jok);
  /* printf("jok_2 : %d \n", cvls_petsc_mem_2->jok); */
  int steps;
  CVodeGetNumSteps(cvls_petsc_mem->cvode_mem, &steps);
  printf("num steps : %d \n", steps);
  N_VPrint_Petsc(x_n);


  CVodeGetCurrentGamma(cvls_petsc_mem->cvode_mem, &gamma);
  x_n_petsc = N_VGetVector_Petsc(x_n);
  int cvode_steps;

  cv_mem = (CVodeMem)(cvls_petsc_mem->cvode_mem);
  /* check jacobian needs to be updated */
  printf("jok (inside delayed snes) = ");
  printf("%d", cvls_petsc_mem->jok);
  if (cvls_petsc_mem->jok) {
    /* use saved copy of jacobian */
    (cvls_petsc_mem->jcur) = PETSC_FALSE;
    /* Overwrite linear system matrix with saved J
       Assuming different non zero structure */
    /* TODO: expose the NON zero structure usage */
    /* ierr = MatCopy(cvls_petsc_mem->savedJ, A, DIFFERENT_NONZERO_PATTERN); */
    A = cvls_petsc_mem->savedJ;
    CHKERRQ(ierr);
    printf("branch 1 -------- \n");
  } else {
    /* call jac() to update the function */
    (cvls_petsc_mem->jcur) = PETSC_TRUE;
    ierr = MatZeroEntries(A);
    printf("this is great\n");
    CHKERRQ(ierr);
    /* compute new jacobian matrix */
    ierr = cvls_petsc_mem->user_jac_func(t, x_n_petsc, A, cvls_petsc_mem->user_mem);
    CHKERRQ(ierr);

    /* Update saved jacobian copy */
    /* TODO: expose nonzero structure usage */
    /* I'm not sure this makes sense */
    /* TODO: fix savedJ not initalized */
    /* MatCopy(A, cvls_petsc_mem->savedJ, DIFFERENT_NONZERO_PATTERN); */
    cvls_petsc_mem->savedJ = A;
    MatView(A, 0);
    MatView(cvls_petsc_mem->savedJ, 0);
    printf("here 2");
    printf("branch 2 --------- \n ");
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
  cv_mem = (CVodeMem) cvls_petsc_mem->cvode_mem;
  /* needs changes */
  PetscReal gamma;
  CVodeGetCurrentGamma(cvls_petsc_mem->cvode_mem, &gamma);
  if (cv_mem->cv_gamrat != ONE) {
    VecScale(b, TWO / (ONE + cv_mem->cv_gamrat));
  }
  /* TODO Figure out how to add recoverable cases from cvode_ls 1677:25 here */
  /* we can do that later considering it's not as necessary */

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
PetscErrorCode CVSNESMNSetup(SNES snes, CVMNPETScMem cvmnmem, Mat Jac_mat, CVSNESJacFn func,void * user_mem, void *cvode_mem, Vec y, booleantype scalesol) {
  /* PetscFunctionBegin; */
    if (cvmnmem==NULL) {
        cvmnmem = (CVMNPETScMem)malloc(sizeof *cvmnmem);
    }
    CVodeMem cv_mem;

  KSP ksp;
  PC pc;
  PCType pc_type;
  cvmnmem->jok = PETSC_FALSE;
  printf("this is being called");
  /* TODO: check whether it needs to be updated */
  cvmnmem->jcur = PETSC_TRUE;  
  /* assign memory for the saved jacobian */
  MatDuplicate(Jac_mat, MAT_DO_NOT_COPY_VALUES, &cvmnmem->savedJ);
  /* we may not have to do this */
  /* assign memory for the vectors */
  VecDuplicate(y, &cvmnmem->ycur);
  VecDuplicate(y, &cvmnmem->fcur);

  /* whether to scale the solution after the solve or not */
  cvmnmem->scalesol = scalesol;
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
  /* set up memory for saved J */
  

  /* If the Preconditioner is not LU or Cholesky, return no support exit code */
  if (!(strcmp(pc_type, PCLU) || strcmp(pc_type, PCCHOLESKY))) {
    PetscFunctionReturn(PETSC_ERR_SUP);
  }

  KSPSetPreSolve(ksp, cvLSPreSolveKSP, (void *)cvmnmem);
  KSPSetPostSolve(ksp, cvLSPostSolveKSP, (void *)cvmnmem);

  /* pass the wrapped jacobian function on to SNES */
 /* SNESSetJacobian(snes, Jac_mat, Jac_mat, CVDelayedJSNES, cvmnmem); */
  /* TODO: make this the only approach */
  /* SNESSetApplicationContext(snes, &cvmnmem); */
  /* SNESGetApplicationContext(snes, (void **) &cvmnmem); */
  /* assign memory pointers */
  cvmnmem->user_mem = user_mem;
  cvmnmem->user_jac_func = func;
  printf("setup done for modified newton \n");
  /* Set the jacobian function */
  /* PetscFunctionReturn(0); */
}

/* convergence test written with for SNES as done by CVODE */
/* Well if this is going to be bad might as well  */
/* I'm going to assume the relative tolerances aren't different in different
   directions. note that xnorm, fnorm and
   gnorm are Petsc functions so it makes our lives easier */
PetscErrorCode CVodeConvergenceTest(SNES snes, PetscInt it, PetscReal xnorm,
                                    PetscReal gnorm, PetscReal fnorm,
                                    SNESConvergedReason *reason, void *cctx) {
    /* we don't use the norms because they're not weighted   */


    
}