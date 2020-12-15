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
#include <iostream>
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
#include <memory>
// #include <pele/array.hpp>
// #include <pele/cvode.hpp>
// #include <pele/lj.hpp>
#include <petsctao.h>

#include <cvode/cvode.h>
#include <nvector/nvector_petsc.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sunnonlinsol/sunnonlinsol_petscsnes.h>

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

#include <gtest/gtest.h>

/* User-defined data structure */
typedef struct UserData_ {
  realtype alpha; /* particle velocity */

  int orbits;      /* number of orbits */
  realtype torbit; /* orbit time       */

  realtype rtol; /* integration tolerances */
  realtype atol;

  int proj;    /* enable/disable solution projection */
  int projerr; /* enable/disable error projection */

  int tstop; /* use tstop mode */
  int nout;  /* number of outputs per orbit */

} * UserData;

/* Functions provided to CVODE */
static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data);
static int Jac(realtype t, N_Vector y, N_Vector fy, SUNMatrix J,
               void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

/* Utility functions */
static int InitUserData(UserData udata);
static int PrintUserData(UserData udata);
static void InputHelp();
static int ComputeSolution(realtype t, N_Vector y, UserData udata);
static int ComputeError(realtype t, N_Vector y, N_Vector e, realtype *ec,
                        UserData udata);
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
  realtype tout = ZERO;  /* next output time           */
  realtype ec = ZERO;    /* constraint error           */
  UserData udata = NULL; /* user data structure        */

  void *cvode_mem = NULL;    /* CVODE memory         */
  N_Vector y = NULL;         /* solution vector      */
  realtype *ydata = NULL;    /* solution vector data */
  N_Vector e = NULL;         /* error vector         */
  SUNMatrix A = NULL;        /* Jacobian matrix      */
  SUNLinearSolver LS = NULL; /* linear solver        */

  udata = (UserData)malloc(sizeof *udata);
  retval = InitUserData(udata);
 

  /* Create serial vector to store the solution */
  y = N_VNew_Serial(2);
 

  /* Set initial contion */
  ydata    = N_VGetArrayPointer(y);
  ydata[0] = ONE;
  ydata[1] = ONE;

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

  /* Output problem setup */
  retval = PrintUserData(udata);

  /* Output initial condition */
  printf("\n     t            x              y");
  printf("             err x          err y       err constr\n");


  /* Integrate in time and periodically output the solution and error */
  if (udata->nout > 0)
  {
    totalout = udata->orbits * udata->nout;
    dtout    = udata->torbit / udata->nout;
  }
  else
  {
    totalout = 1;
    dtout    = udata->torbit * udata->orbits;
  }
  tout = dtout;

  for (out = 0; out < totalout; out++)
  {
    /* Stop at output time (do not interpolate output) */
    if (udata->tstop || udata->nout == 0)
    {
      retval = CVodeSetStopTime(cvode_mem, tout);
    }

    /* Advance in time */
    retval = CVode(cvode_mem, tout, y, &t, CV_NORMAL);

    /* Output solution and error */
    if (udata->nout > 0)
    {
      retval = ComputeError(t, y, e, &ec, udata);
    }

    /* Update output time */
    if (out < totalout - 1)
    {
      tout += dtout;
    }
    else
    {
      tout = udata->torbit * udata->orbits;
    }
  }

  /* Output final solution and error to screen */
  ComputeError(t, y, e, &ec, udata);

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

  fdata[0] = -(udata->alpha) * ydata[0];
  fdata[1] = -(udata->alpha) * ydata[0];

  return (0);
}

/* Compute the Jacobian of the right-hand side function, J(t,y) = df/dy */
static int Jac(realtype t, N_Vector y, N_Vector fy, SUNMatrix J,
               void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
  UserData udata = (UserData)user_data;
  realtype *Jdata = SUNDenseMatrix_Data(J);

  Jdata[0] = -(udata->alpha);
  Jdata[1] = ZERO;
  Jdata[2] = ZERO;
  Jdata[3] = -(udata->alpha);
  return (0);
}

/* -----------------------------------------------------------------------------
 * Private helper functions
 * ---------------------------------------------------------------------------*/

static int InitUserData(UserData udata) {
  int arg_idx = 1;

  /* set default values */
  udata->alpha = ONE;

  udata->orbits = 100;
  udata->torbit = (TWO * PI) / udata->alpha;

  udata->rtol = RCONST(1.0e-4);
  udata->atol = RCONST(1.0e-9);

  udata->proj = 1;
  udata->projerr = 0;

  udata->tstop = 0;
  udata->nout = 0;

  return (0);
}

static int PrintUserData(UserData udata) {
  if (udata == NULL)
    return (-1);

  printf("\nParticle traveling on the unit circle example\n");
  printf("---------------------------------------------\n");
  printf("alpha      = %0.4" ESYM "\n", udata->alpha);
  printf("num orbits = %d\n", udata->orbits);
  printf("---------------------------------------------\n");
  printf("rtol       = %" GSYM "\n", udata->rtol);
  printf("atol       = %" GSYM "\n", udata->atol);
  printf("proj sol   = %d\n", udata->proj);
  printf("proj err   = %d\n", udata->projerr);
  printf("nout       = %d\n", udata->nout);
  printf("tstop      = %d\n", udata->tstop);
  printf("---------------------------------------------\n");

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

/* Compute the analytical solution */
static int ComputeSolution(realtype t, N_Vector y, UserData udata) {
  realtype *ydata = N_VGetArrayPointer(y);
  ydata[0] = COS((udata->alpha) * t);
  ydata[1] = SIN((udata->alpha) * t);
  return (0);
}

/* Compute the error in the solution and constraint */
static int ComputeError(realtype t, N_Vector y, N_Vector e, realtype *ec,
                        UserData udata) {
  realtype *ydata = N_VGetArrayPointer(y);
  int retval;

  /* solution error */
  retval = ComputeSolution(t, e, udata);
  if (check_retval(&retval, "ComputeSolution", 1))
    return (1);
  N_VLinearSum(ONE, y, -ONE, e, e);

  /* constraint error */
  *ec = ydata[0] * ydata[0] + ydata[1] * ydata[1] - ONE;

  return (0);
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
  retval = CVodeGetNumErrTestFails(cvode_mem, &netf);
  check_retval(&retval, "CVodeGetNumErrTestFails", 1);
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
  printf("Number of error test failures = %-6ld\n", netf);

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
