/**
 * Checks for using TAO and SNES methods using pele
 */

#include "nvector/nvector_petsc.h"
#include "pele/base_potential.hpp"
#include "pele/cell_lists.hpp"
#include "pele/inversepower.hpp"
#include "pele/rosenbrock.hpp"

#include "petscmat.h"
#include "petscsnes.h"
#include "petscsys.h"
#include <petscksp.h>
#include "petscsystypes.h"
#include "petscvec.h"
#include "petscviewer.h"
#include "sundials/sundials_nvector.h"
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <pele/array.hpp>
#include <pele/cvode.hpp>
#include <pele/lj.hpp>
#include <pele/pot_utils_tao_snes.hpp>
#include <petsctao.h>
    
using namespace pele;

// Tao tests
typedef struct {
  PetscInt n;      /* dimension */
  PetscReal alpha; /* condition parameter */
  PetscBool chained;
} AppCtx;

PetscErrorCode FormFunctionGradient(Tao tao, Vec X, PetscReal *f, Vec G,
                                    void *ptr) {
  AppCtx *user = (AppCtx *)ptr;
  PetscInt i, nn = user->n / 2;
  PetscErrorCode ierr;
  PetscReal ff = 0, t1, t2, alpha = user->alpha;
  PetscScalar *g;
  const PetscScalar *x;

  PetscFunctionBeginUser;
  /* Get pointers to vector data */
  ierr = VecGetArrayRead(X, &x);
  CHKERRQ(ierr);
  ierr = VecGetArray(G, &g);
  CHKERRQ(ierr);

  /* Compute G(X) */
  if (user->chained) {
    g[0] = 0;
    for (i = 0; i < user->n - 1; i++) {
      t1 = x[i + 1] - x[i] * x[i];
      ff += PetscSqr(1 - x[i]) + alpha * t1 * t1;
      g[i] += -2 * (1 - x[i]) + 2 * alpha * t1 * (-2 * x[i]);
      g[i + 1] = 2 * alpha * t1;
    }
  } else {
    for (i = 0; i < nn; i++) {
      t1 = x[2 * i + 1] - x[2 * i] * x[2 * i];
      t2 = 1 - x[2 * i];
      ff += alpha * t1 * t1 + t2 * t2;
      g[2 * i] = -4 * alpha * t1 * x[2 * i] - 2.0 * t2;
      g[2 * i + 1] = 2 * alpha * t1;
    }
  }

  /* Restore vectors */
  ierr = VecRestoreArrayRead(X, &x);
  CHKERRQ(ierr);
  ierr = VecRestoreArray(G, &g);
  CHKERRQ(ierr);
  *f = ff;

  ierr = PetscLogFlops(15.0 * nn);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
PetscErrorCode FormHessian(Tao tao, Vec X, Mat H, Mat Hpre, void *ptr) {
  AppCtx *user = (AppCtx *)ptr;
  PetscErrorCode ierr;
  PetscInt i, ind[2];
  PetscReal alpha = user->alpha;
  PetscReal v[2][2];
  const PetscScalar *x;
  PetscBool assembled;

  PetscFunctionBeginUser;
  /* Zero existing matrix entries */
  ierr = MatAssembled(H, &assembled);
  CHKERRQ(ierr);
  if (assembled) {
    ierr = MatZeroEntries(H);
    CHKERRQ(ierr);
  }

  /* Get a pointer to vector data */
  ierr = VecGetArrayRead(X, &x);
  CHKERRQ(ierr);

  /* Compute H(X) entries */
  if (user->chained) {
    ierr = MatZeroEntries(H);
    CHKERRQ(ierr);
    for (i = 0; i < user->n - 1; i++) {
      PetscScalar t1 = x[i + 1] - x[i] * x[i];
      v[0][0] = 2 + 2 * alpha * (t1 * (-2) - 2 * x[i]);
      v[0][1] = 2 * alpha * (-2 * x[i]);
      v[1][0] = 2 * alpha * (-2 * x[i]);
      v[1][1] = 2 * alpha * t1;
      ind[0] = i;
      ind[1] = i + 1;
      ierr = MatSetValues(H, 2, ind, 2, ind, v[0], ADD_VALUES);
      CHKERRQ(ierr);
    }
  } else {
    for (i = 0; i < user->n / 2; i++) {
      v[1][1] = 2 * alpha;
      v[0][0] = -4 * alpha * (x[2 * i + 1] - 3 * x[2 * i] * x[2 * i]) + 2;
      v[1][0] = v[0][1] = -4.0 * alpha * x[2 * i];
      ind[0] = 2 * i;
      ind[1] = 2 * i + 1;
      ierr = MatSetValues(H, 2, ind, 2, ind, v[0], INSERT_VALUES);
      CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArrayRead(X, &x);
  CHKERRQ(ierr);

  /* Assemble matrix */
  ierr = MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);
  ierr = MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);
  ierr = PetscLogFlops(9.0 * user->n / 2.0);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

TEST(TaoNewton3, TaoNewtonstepswork) {
    PetscErrorCode ierr; /* used to check for functions returning nonzeros */
    PetscReal zero = 0.0;
    Vec x; /* solution vector */
    Mat H;
    Tao tao; /* Tao solver context */
    PetscBool flg, test_lmvm = PETSC_FALSE;
    PetscMPIInt size; /* number of processes running */
    KSP ksp;
    PC pc;
    Mat M;
    Vec in, out, out2;
    PetscReal mult_solve_dist;

    // initialize potential and starting coordinates
    // we're going to be using rosenbrock with petsc routines
    std::shared_ptr<BasePotential> pot_ = std::make_shared<RosenBrock>(1, 100);

    pot_sptr_wrapper wrapped_pot_;
    wrapped_pot_.potential_ = pot_;


    ierr = PetscInitializeNoArguments();

    ierr = VecCreateSeq(PETSC_COMM_SELF, 2, &x);
    ierr = MatCreateSeqAIJ(PETSC_COMM_SELF, 2, 2, 2, NULL, &H);

    ierr = VecSet(x, zero);

    // Initialize solver
    ierr = TaoCreate(PETSC_COMM_SELF, &tao);
    ierr = TaoSetType(tao,TAOLMVM);
  
    ierr = TaoSetInitialVector(tao, x);
  


    ierr = TaoSetObjectiveAndGradientRoutine(tao, TaoBasePotentialFunctionGradient,
                                             &wrapped_pot_);
    ierr = TaoSetHessianRoutine(tao, H, H, TaoBasePotentialHessian, &wrapped_pot_);
    TaoConvergedReason reason;
    ierr = TaoSolve(tao);

    PetscInt iter;
    TaoGetIterationNumber(tao, &iter);
    std::cout << "iterations:" << iter + 1 << "\n";

    ierr = TaoGetConvergedReason(tao, &reason);
    std::cout << reason << "\n";
    std::cout << TAO_CONVERGED_GATOL << "\n";
    PetscReal result;

    std::cout << "objective function value " << TaoGetObjective(tao, &result)
            << "\n";
  Vec result_coords;
  TaoGetSolutionVector(tao, &result_coords);
  std::cout << "result coordinates"
            << "\n";
  VecView(result_coords, PETSC_VIEWER_STDOUT_SELF);
}


TEST(TaoNewtonssas, TaoNewtonstepswork) {
    PetscErrorCode ierr; /* used to check for functions returning nonzeros */
    PetscReal zero = 0.0;
    Vec x; /* solution vector */
    Mat H;
    Tao tao; /* Tao solver context */
    PetscBool flg, test_lmvm = PETSC_FALSE;
    PetscMPIInt size; /* number of processes running */
    KSP ksp;
    PC pc;
    Mat M;
    Vec in, out, out2;
    PetscReal mult_solve_dist;
    MatMumpsSetIcntl(H,24,1);
    // initialize potential and starting coordinates
    // we're going to be using rosenbrock with petsc routines
    std::shared_ptr<BasePotential> pot_ = std::make_shared<RosenBrock>(1, 100);

    pot_sptr_wrapper wrapped_pot_;
    wrapped_pot_.potential_ = pot_;


    ierr = PetscInitializeNoArguments();

    ierr = VecCreateSeq(PETSC_COMM_SELF, 2, &x);
    ierr = MatCreateSeqAIJ(PETSC_COMM_SELF, 2, 2, 2, NULL, &H);

    ierr = VecSet(x, zero);

    // Initialize solver
    ierr = TaoCreate(PETSC_COMM_SELF, &tao);
    ierr = TaoSetType(tao,TAOLMVM);
  
    ierr = TaoSetInitialVector(tao, x);
  


    ierr = TaoSetObjectiveAndGradientRoutine(tao, TaoBasePotentialFunctionGradient,
                                             &wrapped_pot_);
    ierr = TaoSetHessianRoutine(tao, H, H, TaoBasePotentialHessian, &wrapped_pot_);
    TaoConvergedReason reason;
    ierr = TaoSolve(tao);

    PetscInt iter;
    TaoGetIterationNumber(tao, &iter);
    std::cout << "iterations:" << iter + 1 << "\n";

    ierr = TaoGetConvergedReason(tao, &reason);
    std::cout << reason << "\n";
    std::cout << TAO_CONVERGED_GATOL << "\n";
    PetscReal result;

    std::cout << "objective function value " << TaoGetObjective(tao, &result)
              << "\n";
    Vec result_coords;
    TaoGetSolutionVector(tao, &result_coords);
    std::cout << "result coordinates"
              << "\n";
    VecView(result_coords, PETSC_VIEWER_STDOUT_SELF);
}