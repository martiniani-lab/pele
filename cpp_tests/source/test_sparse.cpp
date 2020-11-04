/**
 * Tests for sparse methods for hessians in pele 
 */


#include "pele/array.hpp"
#include "pele/lj.hpp"
#include "pele/ngt.hpp"
#include "petscsys.h"
#include "petscsystypes.h"
#include <petscmat.h>
#include <petscvec.h>


#include <iostream>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>
#include <cmath>
#include <memory>


using pele::Array;




TEST(Sparse_energy_gradient_hessian, Energy_Gradient_Hessian_Works){
    auto lj = std::make_shared<pele::LJ> (1., 1.);

    // setup coords
    int natoms = 3;
    Array<double> x;
    x = Array<double>(3*natoms);
    x[0]  = 0.1;
    x[1]  = 0.2;
    x[2]  = 0.3;
    x[3]  = 0.44;
    x[4]  = 0.55;
    x[5]  = 1.66;
    x[6] = 0;
    x[7] = 0;
    x[8] = -3.;
    int N = 3*natoms;
    Array<double> g(N);
    Array<double> h(N*N);
    

    double e = lj->get_energy_gradient_hessian(x, g, h);  
    // set up sparse gradient hessian;
    PetscInitializeNoArguments();
    PetscInt blocksize = 1;
    PetscInt hessav = 9;
    Mat petsc_hess;
    Vec petsc_grad;
    PetscInt i_petsc[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    PetscInt j_petsc[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    PetscScalar sparse_hess_vals[N*N];
    MatCreateSeqSBAIJ(PETSC_COMM_SELF, blocksize, N, N, hessav, NULL, &petsc_hess);
    VecCreateSeq(PETSC_COMM_SELF, N, &petsc_grad);
    double e_sparse = lj->get_energy_gradient_hessian_sparse(x, petsc_grad, petsc_hess);
    MatType hesstype;
    MatGetType(petsc_hess, &hesstype);
    double val;
    // to view what the hessian looks like
    // MatView(petsc_hess,PETSC_VIEWER_STDOUT_WORLD);
    for (int i =0; i < N; ++i) {
        for (int j = i; j < N; ++j) {
            MatGetValue(petsc_hess, i, j, &val);
            ASSERT_NEAR(val, h[N*i+j], 1e-10);   
        }
    }
    for (int i=0; i < h.size(); ++i) {
        

    }
}


TEST(Sparse_energy_gradient, Energy_Gradient_Works){
    auto lj = std::make_shared<pele::LJ> (1., 1.);

    // setup coords
    int natoms = 3;
    Array<double> x;
    x = Array<double>(3*natoms);
    x[0]  = 0.1;
    x[1]  = 0.2;
    x[2]  = 0.3;
    x[3]  = 0.44;
    x[4]  = 0.55;
    x[5]  = 1.66;
    x[6] = 0;
    x[7] = 0;
    x[8] = -3.;
    int N = 3*natoms;
    Array<double> g(N);
    double e = lj->get_energy_gradient(x, g);   
    // set up sparse gradient hessian;
    PetscInitializeNoArguments();
    PetscInt blocksize = 1;
    PetscInt hessav = 2;
    Mat petsc_hess;
    Vec petsc_grad;
    VecCreateSeq(PETSC_COMM_SELF, N, &petsc_grad);
    double e_sparse = lj->get_energy_gradient_sparse(x, petsc_grad);

    PetscInt i_petsc[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    PetscScalar petsc_grad_vals[9];
    VecGetValues(petsc_grad, 9, i_petsc, petsc_grad_vals);
    
    ASSERT_NEAR(e, e_sparse, 1e-10);
    for (int i = 0; i < N; ++i) {
        ASSERT_NEAR(petsc_grad_vals[i], g[i], 1e-10);
    }
}