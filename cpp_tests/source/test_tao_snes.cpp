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
#include <cstddef>
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

TEST(TaoNewton, TaoNewtonstepswork) {
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



/**
 * Null space computation using mumps
 * The null space is used for the gmres solver
 */
TEST(MUMPSNullSpace, MUMPSNullSpaceCalculated) {
    PetscErrorCode ierr; /* used to check for functions returning nonzeros */
    // PetscReal zero = 0.0;
    // Vec x; /* solution vector */
    // Mat H;
    // Tao tao; /* Tao solver context */
    // PetscBool flg, test_lmvm = PETSC_FALSE;
    // PetscMPIInt size; /* number of processes running */
    // KSP ksp;
    // PC pc;
    // Mat M;
    // Vec in, out, out2;
    // PetscReal mult_solve_dist;

    PetscInitializeNoArguments();
    // We generate a 4x4 matrix with a non obvious null space and try to remove it while using a linear solver

    double rows = 4;
    double mat_arr[] = {0, 0, 0, 0,
        0, 1, 1, 0,
                        0, 1, 1, 0,
                        0, 0, 0, 0}; // The eigenvalues of this matrix are 0, 2, 0, 0

    // The null space consists of 0, 1, 1, 0 which is what we shall use

    


    double vec_arr[] = {2, 2, 0, 1}; // this is b
    // The goal is to remove the null space while solving A x = b (should remove)


    


    
    Mat A;
    Vec b;
    // ierr = MatCreateSeqAIJ(PETSC_COMM_SELF, rows, rows, rows, NULL, &A);
    MatCreateSeqSBAIJ(PETSC_COMM_SELF,1,rows,rows,rows,NULL,&A);
    VecCreateSeq(PETSC_COMM_SELF, rows, &b);
    

    PetscInt vals[] = {0, 1, 2, 3};


    VecSetValues(b, rows, vals, vec_arr, INSERT_VALUES);
    MatSetValues(A, rows, vals, rows, vals, mat_arr, INSERT_VALUES);

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
    VecAssemblyBegin(b);
    VecAssemblyEnd(b);
    VecView(b, PETSC_VIEWER_STDOUT_SELF);
    MatView(A, PETSC_VIEWER_STDOUT_SELF);


    
    Mat F,work,V;
    PetscInt N;

    MatGetFactor(A,MATSOLVERMUMPS,MAT_FACTOR_CHOLESKY,&F);
    MatMumpsSetIcntl(F,24,1);
    MatMumpsSetIcntl(F,25,-1);
    MatCholeskyFactorSymbolic(F, A, NULL, NULL);
    MatCholeskyFactorNumeric(F, A, NULL);
        
    // MatLUFactorSymbolic(F,A,NULL,NULL,NULL);
    // MatLUFactorNumeric(F,A,NULL);
    MatMumpsGetInfog(F,28,&N);   /* this is the dimension of the null space */
    MatCreateDense(PETSC_COMM_SELF,rows,N,PETSC_DETERMINE,PETSC_DETERMINE,NULL,&V);    /* this will contain the null space in the columns */
    MatDuplicate(V,MAT_DO_NOT_COPY_VALUES,&work);
    MatMatSolve(F,work,V);
    MatView(V, PETSC_VIEWER_STDOUT_SELF);
    MatView(work, PETSC_VIEWER_STDOUT_SELF);
}

