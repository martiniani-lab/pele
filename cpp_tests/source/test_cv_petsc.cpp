/*-----------------------------------------------------------------
 * NOTE: This test is being used as a development environment
 * for c code. preferably use C libraries as much as possible
 *-----------------------------------------------------------------
 * acknowledgement: some code reused from other petsc examples
 *-----------------------------------------------------------------
 * Example/Test to compare function evaluation performance between
 * sundials dense and a comparable petsc implementation using
 * cholesky for a system defined by
 * $$
 * x' = - \grad{V(x)}
 * $$
 *
 * where
 * $$
 * V(x) = - (a-x[0])^2 + b(x[1]-x[0]^2)^2
 * $$
 * This is the Rosenbrock function (not to be confused with
 * rosenbrock ODE methods). The key idea of this test is to
 * identify the attractor (There is only one,the global minimum)
 * corresponding to a point.
 * The key idea is not to solve an optimization problem
 * (for which there are better methods check Nocedal and Wright)
 * but to ensure the PETSc version works just as well to identify
 * the attractor for a slightly non trivial problem where the Jacobian
 * is symmetric.
 *
 * Note: no non convex regions here, maybe a better example might help
 * but all examples I know are slightly non trivial
 * -----------------------------------------------------------------
 * SUNDIALS Copyright Start
 * Copyright (c) 2002-2020, Lawrence Livermore National Security
 * and Southern Methodist University.
 * All rights reserved.
 *
 * See the top-level LICENSE and NOTICE files for details.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * SUNDIALS Copyright End
 *---------------------------------------------------------------*/

#include <csignal>
#include <cstddef>
#include <functional>
#include <math.h>
#include <new>
#include <petscdm.h>
#include <petscdmda.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "cvode/cvode_ls.h"
#include "nvector/nvector_petsc.h"
#include "pele/base_potential.hpp"
#include "pele/cell_lists.hpp"
#include "pele/harmonic.hpp"
#include "pele/inversepower.hpp"

// #include "pele/petsc_interface.hpp"
// #include "pele/rosenbrock.hpp"
#include <memory>

#include "petscksp.h"
#include "petscmat.h"
#include "petscpc.h"
#include "petscpctypes.h"
#include "petscsnes.h"
#include "petscsys.h"
#include "petscsystypes.h"
#include "petscvec.h"
#include "petscviewer.h"
#include "sundials/sundials_linearsolver.h"
#include "sundials/sundials_matrix.h"
#include "sundials/sundials_nonlinearsolver.h"
#include "sundials/sundials_nvector.h"
#include "sundials/sundials_types.h"
// #include <pele/array.hpp>
// #include <pele/cvode.hpp>
// #include <pele/lj.hpp>
#include "pele/sunnonlinsol_petscsnes.h"
#include <cvode/cvode.h>
#include <nvector/nvector_petsc.h>
#include <nvector/nvector_serial.h>
#include <pele/cvode_modified_newton.h>
#include <petsctao.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sunmatrix/sunmatrix_dense.h>
// #include <sunnonlinsol/sunnonlinsol_petscsnes.h>
#include <unistd.h>

/* gtest */
#include <gtest/gtest.h>

/* Precision specific formatting macros */
#if defined(SUNDIALS_EXTENDED_PRECISION)
#define GSYM "Lg"
#define ESYM "Le"
#define FSYM "Lf"
#else
#define GSYM "g"
#define ESYM "e"
#define FSYM "f"
#endif

/* Precision specific math function macros */
#if defined(SUNDIALS_DOUBLE_PRECISION)
#define SIN(x) (sin((x)))
#define COS(x) (cos((x)))
#define SQRT(x) (sqrt((x)))
#elif defined(SUNDIALS_SINGLE_PRECISION)
#define SIN(x) (sinf((x)))
#define COS(x) (cosf((x)))
#define SQRT(x) (sqrtf((x)))
#elif defined(SUNDIALS_EXTENDED_PRECISION)
#define SIN(x) (sinl((x)))
#define COS(x) (cosl((x)))
#define SQRT(x) (sqrtl((x)))
#endif

/*!
 * This exists because strcmp doesn't follow
 * the true false convention and returns 0
 * if the strings are the same
 */
#define same_string(x, y) (strcmp(x, y) == 0)
/* Problem Constants */
#define PI RCONST(3.141592653589793238462643383279502884197169)
#define ZERO RCONST(0.0)
#define ONE RCONST(1.0)
#define TWO RCONST(2.0)
#define HUNDRED RCONST(100.0)
#define MAXSTEPS 5

/* User-defined data structure */
typedef struct UserData_ {
  realtype a; /* particle velocity */
  realtype b; // rosenbrock location

  realtype rtol; /* integration tolerances */
  realtype atol;

  int tstop; /* use tstop mode */

  /* TODO: fix this within petsc */
  int njac_petsc; /* jacobian evaluations for the petsc solver */

} * UserData;

/* Functions provided to CVODE */
static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data);
static int Jac(realtype t, N_Vector y, N_Vector fy, SUNMatrix J,
               void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

/* Utility functions */
static int InitUserData(UserData udata);
static int PrintStats(void *cvode_mem);
static int PrintStats_petsc(void *cvode_mem, void *user_data);
static int check_retval(void *returnvalue, const char *funcname, int opt);

int rosenbrock_minus_gradient_petsc(PetscReal t, N_Vector x, N_Vector xdot,
                                    void *user_data);
PetscErrorCode rosenbrock_minus_Jac_petsc(PetscReal t, Vec x, Mat J,
                                          void *user_data);

/**
 * This runs a comparison between CVODE one step with sundials dense and
 * otherwise
 */
TEST(CVDP, CVM) {
  int retval;            /* reusable return flag       */
  int out = 0;           /* output counter             */
  int totalout = 0;      /* output counter             */
  realtype t = ZERO;     /* current integration time   */
  realtype dtout = ZERO; /* output spacing             */
  realtype tout = 10000; /* next output time           */
  realtype ec = ZERO;    /* constraint error           */
  UserData udata = NULL; /* user data structure        */

  void *cvode_mem = NULL;    /* CVODE memory         */
  N_Vector y = NULL;         /* solution vector      */
  realtype *ydata = NULL;    /* solution vector data */
  N_Vector e = NULL;         /* error vector         */
  SUNMatrix A = NULL;        /* Jacobian matrix      */
  SUNLinearSolver LS = NULL; /* linear solver        */


  /* minimum identification */
  double norm2;       /* norm of the funtion */
  double wnorm;       /* norm weighted by length  for identification */
  double minimum_tol; /* tolerance with which the minimum is identified */

  minimum_tol = 1e-6;   /* minimum tolerance */
  double maxsteps = MAXSTEPS; /* maximum number of steps to run the solver for */

  udata = (UserData)malloc(sizeof *udata);
  retval = InitUserData(udata);

  /* Create serial vector to store the solution */
  y = N_VNew_Serial(2);

  /* Set initial contion */
  ydata = N_VGetArrayPointer(y);
  ydata[0] = ONE;
  ydata[1] = ZERO;
  

  /* Create serial vector to store the solution error */
  e = N_VClone(y);

  /* Set initial error */
  N_VConst(ZERO, e);

  /* Create CVODE memory */
  cvode_mem = CVodeCreate(CV_BDF);

  /* Initialize CVODE */
  retval = CVodeInit(cvode_mem, f, t, y);

  /* Attach user-defined data structure to CVODE */
  retval = CVodeSetUserData(cvode_mem, udata);

  /* Set integration tolerances */
  retval = CVodeSStolerances(cvode_mem, udata->rtol, udata->atol);

  /* Create dense SUNMatrix for use in linear solves */
  A = SUNDenseMatrix(2, 2);

  /* Create dense SUNLinearSolver object */
  LS = SUNLinSol_Dense(y, A);

  /* Attach the matrix and linear solver to CVODE */
  retval = CVodeSetLinearSolver(cvode_mem, LS, A);

  /* Set a user-supplied Jacobian function */
  retval = CVodeSetJacFn(cvode_mem, Jac);

  /* Set max steps between outputs */
  retval = CVodeSetMaxNumSteps(cvode_mem, 100000);

  N_Vector grad;
  grad = N_VClone(y);

  // break out if close to a minimum
  for (int nstep = 0; nstep < maxsteps; ++nstep) {
    // take a step
      std::cout << "--------- Step:" << nstep << "\n";

    retval = CVode(cvode_mem, tout, y, &t, CV_ONE_STEP);
    // obtain  the gradient
    std::cout << "gradient :"
              << "\n";

    N_VPrint_Serial(grad);
    std::cout << "y:"
              << "\n";
    N_VPrint_Serial(y);
    CVodeGetDky(cvode_mem, t, 1, grad);
    norm2 = N_VDotProd(grad, grad);
    /* calculate norm accounting for length */
    wnorm = sqrt(norm2 / N_VGetLength(y));
    double gamma;
    CVodeGetCurrentGamma(cvode_mem, &gamma);
    printf("Gamma %d \n", gamma);
    std::cout << wnorm << "\n";
    /* break out when the minimum is found */
    if (wnorm < minimum_tol) {
      break;
    }
  }
  /* Output final solution and error to screen */

  /* Print some final statistics */
  PrintStats(cvode_mem);

  /* Free memory */
  N_VDestroy(y);
  SUNMatDestroy(A);
  SUNLinSolFree(LS);
  CVodeFree(&cvode_mem);
}

TEST(CVPet, CVM) {
  PetscInitializeNoArguments();
  /* initial setup */
  realtype tout = 10000; /* next output time           */
  int retval;

  /* Setup for Petsc solver */
  void *cvode_mem_PETSc = NULL; /* CVODE memory         */
  N_Vector y_nv_petsc = NULL; /* solution vector wrapper around petsc vector  */
  Vec y_vec;                  /* solution vector petsc */
  UserData udata_petsc = NULL; /* user data for the petsc version */
  Mat Jac_petsc;
  realtype t_petsc = ZERO; /* current integration time petsc  */

  void *cvode_mem = NULL;    /* CVODE memory         */
  N_Vector y = NULL;         /* solution vector      */
  realtype *ydata = NULL;    /* solution vector data */
  N_Vector e = NULL;         /* error vector         */
  SUNMatrix A = NULL;        /* Jacobian matrix      */
  SUNLinearSolver LS = NULL; /* linear solver        */
  UserData udata = NULL; /* user data structure        */
  realtype t = ZERO;     /* current integration time   */

  udata = (UserData)malloc(sizeof *udata);
  retval = InitUserData(udata);

  N_Vector y_diff;



  // ------------------- CVODE SETUP without petsc

  /* Create serial vector to store the solution */
  y = N_VNew_Serial(2);

  /* Create serial vector to store the solution error */
  e = N_VClone(y);

  y_diff = N_VClone(y);

/* Set initial contion */
  ydata = N_VGetArrayPointer(y);
  ydata[0] = ONE;
  ydata[1] = ZERO;

  /* Create CVODE memory */
  cvode_mem = CVodeCreate(CV_BDF);
  /* Initialize CVODE */
  retval = CVodeInit(cvode_mem, f, t, y);
  
    /* Attach user-defined data structure to CVODE */
  retval = CVodeSetUserData(cvode_mem, udata);


  /* Set integration tolerances */
  retval = CVodeSStolerances(cvode_mem, udata->rtol, udata->atol);

    /* Create dense SUNMatrix for use in linear solves */
  A = SUNDenseMatrix(2, 2);

  /* Create dense SUNLinearSolver object */
  LS = SUNLinSol_Dense(y, A);

  /* Attach the matrix and linear solver to CVODE */
  retval = CVodeSetLinearSolver(cvode_mem, LS, A);

  /* Set a user-supplied Jacobian function */
  retval = CVodeSetJacFn(cvode_mem, Jac);

  /* Set max steps between outputs */
  retval = CVodeSetMaxNumSteps(cvode_mem, 100000);
  N_Vector grad = N_VClone(y);

  // ------------------- end CVODE setup without petsc

  /* minimum identification */
  double norm;        /* norm of the funtion */
  double wnorm;       /* norm weighted by length  for identification */
  double minimum_tol; /* tolerance with which the minimum is identified */

  minimum_tol = 1e-6;   /* minimum tolerance */
  double maxsteps = MAXSTEPS; /* maximum number of steps to run the solver for */

  /* allocate memory */
  udata_petsc = (UserData)malloc(sizeof *udata_petsc);
  retval = InitUserData(udata_petsc);

  /* Create petsc vector to store the solution */
  int dim = 2;
  VecCreateSeq(PETSC_COMM_SELF, dim, &y_vec);
  y_nv_petsc = N_VMake_Petsc(y_vec);
  /* clone the gradient */
  N_Vector grad_petsc = N_VClone(y_nv_petsc);


  // /* -----------------------------------------------------------------------------
  //  * initialize vectors for storing memory
  //  * ---------------------------------------------------------------------------*/


  



  /* create matrix for jacobian calculation */
  MatCreateDense(PETSC_COMM_SELF, dim, dim, PETSC_DECIDE, PETSC_DECIDE, NULL,
                 &Jac_petsc);

  // MatCreateSeqSBAIJ(PETSC_COMM_SELF, 2 ,2 ,2 ,2,NULL,&Jac_petsc);
  std::cout << "heeloo 2"
            << "\n";
  /* Set initial condition */
  PetscScalar ydata_petsc[2];
  ydata_petsc[0] = ONE;
  ydata_petsc[1] = ZERO;

  /* set values petsc */
  PetscInt y_index[2] = {0, 1};
  VecSetValues(y_vec, dim, y_index, ydata_petsc, INSERT_VALUES);

    cvode_mem_PETSc = CVodeCreate(CV_BDF);
  retval = CVodeInit(cvode_mem_PETSc, rosenbrock_minus_gradient_petsc, t_petsc,
                     y_nv_petsc);

  retval = CVodeSetUserData(cvode_mem_PETSc, udata_petsc);
  retval =
      CVodeSStolerances(cvode_mem_PETSc, udata_petsc->rtol, udata_petsc->atol);

  PetscScalar *ydata_values_petsc;


  /* -----------------------------------------------------------------------------
   * Solver setup petsc
   * ---------------------------------------------------------------------------*/

  

  SNES snes;
  KSP ksp;
  SUNNonlinearSolver NLS;
  PC pc;
  SNESLineSearch ls;

  /* memory for CVODE solver using modified newton approach.
     The user data contexts is attached to this*/
  CVMNPETScMem cv_mn_petsc_mem = NULL;

  /* Initialize nonlinear solver */
  SNESCreate(PETSC_COMM_SELF, &snes);
  NLS = SUNNonlinSol_PetscSNES(y_nv_petsc, snes, cv_mn_petsc_mem);

  SNESGetKSP(snes, &ksp);
  SNESGetLineSearch(snes, &ls);

  SNESSetType(snes, SNESNEWTONLS);
  /* defaults to no line search*/
  /* turn off computation of norms in line search */
  KSPGetPC(ksp, &pc);

  PCSetType(pc, PCCHOLESKY);





  /* Attach setup to the SNES Solver */
  CVSNESMNSetup(snes, cv_mn_petsc_mem, Jac_petsc, rosenbrock_minus_Jac_petsc, udata_petsc, cvode_mem_PETSc, y_vec, 1);

  // Attach nonlinear solver to petsc
  CVodeSetNonlinearSolver(cvode_mem_PETSc, NLS);

  PetscReal * y_diff_vals;

  /* workspace for getting gradient out */

  /* -----------------------------------------------------------------------------
   * main run
   * ---------------------------------------------------------------------------*/
  // break out if close to a minimum
  for (int nstep = 0; nstep < maxsteps; ++nstep) {
      std::cout << "--------- Step:" << nstep << "\n";

    if (nstep==66) {
        printf("this is it");
        int blash =3;
    }
    /* TODO check whether a postsolve function can be provided to sundials */
    retval = CVode(cvode_mem_PETSc, tout, y_nv_petsc, &t_petsc, CV_ONE_STEP);
    retval = CVode(cvode_mem, tout, y, &t, CV_ONE_STEP);

    
    
    
    CVodeGetDky(cvode_mem_PETSc, t_petsc, 1, grad_petsc);
    CVodeGetDky(cvode_mem, t, 1, grad);
    /* calculate norm accounting for length */
    VecNorm(N_VGetVector_Petsc(grad_petsc), NORM_2, &norm);
    PetscInt length;
    VecGetSize(N_VGetVector_Petsc(grad_petsc), &length);
    wnorm = norm / sqrt(length);

    // std::cout << "gradient :"
    //           << "\n";
    // N_VPrint_Petsc(grad_petsc);
    std::cout << "y petsc:"
              << "\n";
    N_VPrint_Petsc(y_nv_petsc);


    // Block for calculatin vector differences
    VecGetArray(y_vec, &ydata_values_petsc);
    ydata = N_VGetArrayPointer(y);
    y_diff_vals = N_VGetArrayPointer(y_diff);
    y_diff_vals[0] = ydata[0] - ydata_values_petsc[0];
    y_diff_vals[1] = ydata[1] - ydata_values_petsc[1];
    VecRestoreArray(y_vec, &ydata_values_petsc);




    

    

    
    std::cout << "y:"
              << "\n";
    N_VPrint_Serial(y);

    std::cout << "----- difference y" << "\n";

    N_VPrint_Serial(y_diff);

    std::cout << "-------" << "\n";
    
    
    /* TODO check for ops cloning */
    double gamma;
    CVodeGetCurrentGamma(cvode_mem_PETSc, &gamma);
    printf("Gamma out %d \n", gamma);
    std::cout << wnorm << "\n";
    /* break out when the minimum is found */
    if (wnorm < minimum_tol) {
      break;
    }
  }


  /* Free memory */
  N_VDestroy(y);
  SUNMatDestroy(A);
  SUNLinSolFree(LS);
  CVodeFree(&cvode_mem);
  /* Print some final statistics */
  PrintStats_petsc(cvode_mem_PETSc, udata_petsc);
  PetscFinalize();
}

/* -----------------------------------------------------------------------------
 * Functions provided to CVODE for dense calculations
 * ---------------------------------------------------------------------------*/

/* Compute the right-hand side function, y' = f(t,y) */
static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data) {
  UserData udata = (UserData)user_data;
  realtype *ydata = N_VGetArrayPointer(y);
  realtype *fdata = N_VGetArrayPointer(ydot);

  /* extract member variables */
  realtype m_a = udata->a;
  realtype m_b = udata->b;

  /* rosenbrock calculation */
  /* 1. first calculate gradient */
  fdata[0] = 4 * m_b * ydata[0] * ydata[0] * ydata[0] -
             4 * m_b * ydata[0] * ydata[1] + 2 * m_a * ydata[0] - 2 * m_a;
  fdata[1] = 2 * m_b * (ydata[1] - ydata[0] * ydata[0]);
  /* 2. then take the negative */
  fdata[0] = -fdata[0];
  fdata[1] = -fdata[1];
  return (0);
}

/* Compute the Jacobian of the right-hand side function, J(t,y) = df/dy */
static int Jac(realtype t, N_Vector y, N_Vector fy, SUNMatrix J,
               void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
  UserData udata = (UserData)user_data;
  realtype *Jdata = SUNDenseMatrix_Data(J);
  realtype *ydata = N_VGetArrayPointer(y);

  /* extract member variables */
  realtype m_a = udata->a;
  realtype m_b = udata->b;

  /* rosenbrock jacobian */
  /* 1. first calculate jacobian */
  Jdata[0] = 2 + 8 * m_b * ydata[0] * ydata[0] -
             4 * m_b * (-ydata[0] * ydata[0] + ydata[1]);
  Jdata[1] = -4 * m_b * ydata[0];
  Jdata[2] = -4 * m_b * ydata[0];
  Jdata[3] = 2 * m_b;
  /* 2. then take the negative */
  Jdata[0] = -Jdata[0];
  Jdata[1] = -Jdata[1];
  Jdata[2] = -Jdata[2];
  Jdata[3] = -Jdata[3];

  return (0);
}

/* -----------------------------------------------------------------------------
 * Private helper functions
 * ---------------------------------------------------------------------------*/

static int InitUserData(UserData udata) {
  int arg_idx = 1;

  /* set default values */
  udata->a = ONE;
  udata->b = 0.1;

  udata->rtol = RCONST(1.0e-4);
  udata->atol = RCONST(1.0e-9);

  udata->njac_petsc = 0;

  return (0);
}

/* Print command line options */
static void InputHelp() {
  printf("\nCommand line options:\n");
  printf("  --alpha <vel>      : particle velocity\n");
  printf("  --orbits <orbits>  : number of orbits to perform\n");
  printf("  --rtol <rtol>      : relative tolerance\n");
  printf("  --atol <atol>      : absoltue tolerance\n");
  printf("  --proj <1 or 0>    : enable (1) / disable (0) projection\n");
  printf("  --projerr <1 or 0> : enable (1) / disable (0) error projection\n");
  printf("  --nout <nout>      : outputs per period\n");
  printf("  --tstop            : stop at output time (do not interpolate)\n");
  return;
}

/* Print final statistics */
static int PrintStats(void *cvode_mem) {
  int retval;
  long int nst, nfe, nsetups, nje, nni, ncfn, netf;

  retval = CVodeGetNumSteps(cvode_mem, &nst);
  check_retval(&retval, "CVodeGetNumSteps", 1);
  retval = CVodeGetNumRhsEvals(cvode_mem, &nfe);
  check_retval(&retval, "CVodeGetNumRhsEvals", 1);
  retval = CVodeGetNumLinSolvSetups(cvode_mem, &nsetups);
  check_retval(&retval, "CVodeGetNumLinSolvSetups", 1);
  retval = CVodeGetNumNonlinSolvIters(cvode_mem, &nni);
  check_retval(&retval, "CVodeGetNumNonlinSolvIters", 1);
  retval = CVodeGetNumNonlinSolvConvFails(cvode_mem, &ncfn);
  check_retval(&retval, "CVodeGetNumNonlinSolvConvFails", 1);

  retval = CVodeGetNumJacEvals(cvode_mem, &nje);
  check_retval(&retval, "CVodeGetNumJacEvals", 1);

  printf("\nIntegration Statistics:\n");

  printf("Number of steps taken = %-6ld\n", nst);
  printf("Number of function evaluations = %-6ld\n", nfe);

  printf("Number of linear solver setups = %-6ld\n", nsetups);
  printf("Number of Jacobian evaluations = %-6ld\n", nje);

  printf("Number of nonlinear solver iterations = %-6ld\n", nni);
  printf("Number of convergence failures = %-6ld\n", ncfn);

  return (0);
}

/* Print final statistics Petsc */
static int PrintStats_petsc(void *cvode_mem, void *user_data) {
  int retval;
  long int nst, nfe, nsetups, nje, nni, ncfn, netf;
  UserData udata;
  udata = (UserData)user_data;

  retval = CVodeGetNumSteps(cvode_mem, &nst);
  check_retval(&retval, "CVodeGetNumSteps", 1);
  retval = CVodeGetNumRhsEvals(cvode_mem, &nfe);
  check_retval(&retval, "CVodeGetNumRhsEvals", 1);
  retval = CVodeGetNumLinSolvSetups(cvode_mem, &nsetups);
  check_retval(&retval, "CVodeGetNumLinSolvSetups", 1);
  retval = CVodeGetNumNonlinSolvIters(cvode_mem, &nni);
  check_retval(&retval, "CVodeGetNumNonlinSolvIters", 1);
  retval = CVodeGetNumNonlinSolvConvFails(cvode_mem, &ncfn);
  check_retval(&retval, "CVodeGetNumNonlinSolvConvFails", 1);

  /* This has to be done because CVODE doesn't do it properly */
  /* TODO: change this in CVODE */
  nje = udata->njac_petsc;

  printf("\nIntegration Statistics:\n");

  printf("Number of steps taken = %-6ld\n", nst);
  printf("Number of function evaluations = %-6ld\n", nfe);

  printf("Number of linear solver setups = %-6ld\n", nsetups);
  printf("Number of Jacobian evaluations = %-6ld\n", nje);

  printf("Number of nonlinear solver iterations = %-6ld\n", nni);
  printf("Number of convergence failures = %-6ld\n", ncfn);

  return (0);
}

/* Check function return value */
static int check_retval(void *returnvalue, const char *funcname, int opt) {
  int *retval;

  /* Opt 0: Check if a NULL pointer was returned - no memory allocated */
  if (opt == 0 && returnvalue == NULL) {
    fprintf(stderr, "\nERROR: %s() returned a NULL pointer\n\n", funcname);
    return (1);
  }
  /* Opt 1: Check if retval < 0 */
  else if (opt == 1) {
    retval = (int *)returnvalue;
    if (*retval < 0) {
      fprintf(stderr, "\nERROR: %s() returned = %d\n\n", funcname, *retval);
      return (1);
    }
  }
  return (0);
}

/* -----------------------------------------------------------------------------
 * Functions for Petsc Implementation
 * ---------------------------------------------------------------------------*/

/* computes -\grad{f(t,x)} in the CVODE expected format. this is (- hessian) of
   the rosenbrock funtion t in this problem is a dummy variable to parametrize
   (note. user data should contain a pointer to the memory, since that is where
   The position is implemented)
   X here is NOT the Position.
   path */
PetscErrorCode rosenbrock_minus_Jac_petsc(PetscReal t, Vec x, Mat J,
                                          void *user_data) {
  /* Declarations */
  PetscErrorCode ierr;
  UserData data;
  double m_a;           /* member a */
  double m_b;           /* member b */
  PetscReal hessarr[4]; /* hessian data*/

  /* extract information from user data */
  data = (UserData)user_data;
  m_a = data->a;
  m_b = data->b;

  /* initalize jacobian to zero */
  MatZeroEntries(J);

  /* Illustration of how to deal with symmetric matrices */
  /* get the jacobian type */
  MatType Jac_type;
  MatGetType(J, &Jac_type);

  /* get read only array from vector */
  const double *x_arr;
  VecGetArrayRead(x, &x_arr);

  std::cout << Jac_type << "\n";
  std::cout << same_string(Jac_type, MATSBAIJ) << "\n";
  std::cout << same_string(Jac_type, MATSEQSBAIJ) << "\n";
  std::cout << same_string(Jac_type, MATMPISBAIJ) << "\n";
  std::cout << (same_string(Jac_type, MATSBAIJ) ||
                same_string(Jac_type, MATSEQSBAIJ) ||
                same_string(Jac_type, MATMPISBAIJ))
            << "\n";

  /* if the hessian is symmetric assume that the matrix is upper triangular */
  if (same_string(Jac_type, MATSBAIJ) || same_string(Jac_type, MATSEQSBAIJ) ||
      same_string(Jac_type, MATMPISBAIJ)) {
    hessarr[0] = 2 + 8 * m_b * x_arr[0] * x_arr[0] -
                 4 * m_b * (-x_arr[0] * x_arr[0] + x_arr[1]);
    hessarr[1] = -4 * m_b * x_arr[0];
    hessarr[2] = -4 * m_b * x_arr[0];
    // hessarr[2] = 0;             /* 0 since symmetric you should probably
    // remove */
    hessarr[3] = 2 * m_b;
  } else {
    hessarr[0] = 2 + 8 * m_b * x_arr[0] * x_arr[0] -
                 4 * m_b * (-x_arr[0] * x_arr[0] + x_arr[1]);
    hessarr[1] = -4 * m_b * x_arr[0];
    hessarr[2] = -4 * m_b * x_arr[0];
    hessarr[3] = 2 * m_b;
  }

  /* restore array */
  VecRestoreArrayRead(x, &x_arr);

  PetscInt idxm[] = {0, 1};
  MatSetValues(J, 2, idxm, 2, idxm, hessarr, INSERT_VALUES);
  MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);

  /* take the negative since J = -H */
  MatScale(J, -1.0);
  /* increase jacobian evaluations */
  data->njac_petsc += 1;
  return (0);
}

/* computes f(t,x) in the CVODE expected format. this is (- gradient) of the
   rosenbrock funtion Here t is a dummy variable to parametrize the path
   */
int rosenbrock_minus_gradient_petsc(PetscReal t, N_Vector x, N_Vector xdot,
                                    void *user_data) {
  /* declarations */
  PetscErrorCode ierr;
  Vec x_petsc = N_VGetVector_Petsc(x);
  Vec xdot_petsc = N_VGetVector_Petsc(xdot);
  UserData data;
  double m_a; /* member a */
  double m_b; /* member b */

  /* get read only array from vector */
  const double *x_arr;
  VecGetArrayRead(x_petsc, &x_arr);
  
  /* extract information from user data */
  data = (UserData)user_data;
  m_a = data->a;
  m_b = data->b;

  /* initalize to zero*/
  ierr = VecZeroEntries(xdot_petsc);
  CHKERRQ(ierr);
  /* First calculate the gradient \grad{V(x)}*/
  ierr = VecSetValue(xdot_petsc, 0,
                     4 * m_b * x_arr[0] * x_arr[0] * x_arr[0] -
                         4 * m_b * x_arr[0] * x_arr[1] + 2 * m_a * x_arr[0] -
                         2 * m_a,
                     INSERT_VALUES);
  CHKERRQ(ierr);
  ierr = VecSetValue(xdot_petsc, 1, 2 * m_b * (x_arr[1] - x_arr[0] * x_arr[0]),
                     INSERT_VALUES);
  CHKERRQ(ierr);

  /* restore read array */
  ierr = VecRestoreArrayRead(x_petsc, &x_arr);
  CHKERRQ(ierr);

  /* assemble the vector */
  ierr = VecAssemblyBegin(xdot_petsc);
  CHKERRQ(ierr);
  ierr = VecAssemblyEnd(xdot_petsc);
  CHKERRQ(ierr);

  /* scale the vector by -1. since xdot is -\grad{V(x)} */
  ierr = VecScale(xdot_petsc, -1.0);
  CHKERRQ(ierr);
  return (0);
}

/* -----------------------------------------------------------------------------
 * Sets up solver information
 * ---------------------------------------------------------------------------*/
