/*-----------------------------------------------------------------
 *
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

#include "nvector/nvector_petsc.h"
#include "pele/base_potential.hpp"
#include "pele/cell_lists.hpp"
#include "pele/harmonic.hpp"
#include "pele/inversepower.hpp"
#include <cstdio>
#include <iostream>
#include <math.h>
#include <petscdm.h>
#include <petscdmda.h>

// #include "pele/petsc_interface.hpp"
// #include "pele/rosenbrock.hpp"
#include "petscmat.h"
#include "petscsnes.h"
#include "petscsys.h"
#include "petscsystypes.h"
#include "petscvec.h"
#include "petscviewer.h"
#include "sundials/sundials_linearsolver.h"
#include "sundials/sundials_nvector.h"
#include "sundials/sundials_types.h"
#include <memory>
// #include <pele/array.hpp>
// #include <pele/cvode.hpp>
// #include <pele/lj.hpp>
#include <petsctao.h>

#include <cvode/cvode.h>
#include <nvector/nvector_petsc.h>
#include <nvector/nvector_serial.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunnonlinsol/sunnonlinsol_petscsnes.h>
#include <unistd.h>

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

/* Problem Constants */
#define PI RCONST(3.141592653589793238462643383279502884197169)
#define ZERO RCONST(0.0)
#define ONE RCONST(1.0)
#define TWO RCONST(2.0)
#define HUNDRED RCONST(100.0)

#include <gtest/gtest.h>

/* User-defined data structure */
typedef struct UserData_ {
  realtype a; /* particle velocity */
  realtype b; // rosenbrock location

  realtype rtol; /* integration tolerances */
  realtype atol;

  int tstop; /* use tstop mode */

} * UserData;

/* Functions provided to CVODE */
static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data);
static int Jac(realtype t, N_Vector y, N_Vector fy, SUNMatrix J,
               void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

/* Utility functions */
static int InitUserData(UserData udata);
static int PrintStats(void *cvode_mem);
static int check_retval(void *returnvalue, const char *funcname, int opt);

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
  realtype tout = 10000;  /* next output time           */
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
  
  
  minimum_tol = 1e-6;    /* minimum tolerance */
  double maxsteps = 400; /* maximum number of steps to run the solver for */  

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
    retval = CVode(cvode_mem, tout, y, &t, CV_ONE_STEP);
    // obtain  the gradient
    
    CVodeGetDky(cvode_mem, t, 1, grad);
    norm2 = N_VDotProd(grad, grad);
    /* calculate norm accounting for length */
    wnorm = sqrt(norm2 / N_VGetLength(y));
    N_VPrint(grad);
    double gamma;    
    CVodeGetCurrentGamma(cvode_mem, &gamma);
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

/* -----------------------------------------------------------------------------
 * Functions provided to CVODE
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
  udata->b = HUNDRED;

  udata->rtol = RCONST(1.0e-4);
  udata->atol = RCONST(1.0e-9);

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
