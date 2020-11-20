#include "pele/cvode.hpp"
#include "autodiff/reverse/eigen.hpp"
#include "cvode/cvode.h"
#include "cvode/cvode_ls.h"
#include "pele/base_potential.hpp"
#include "pele/optimizer.hpp"
#include "pele/debug.hpp"

// cvode imports
#include "petscerror.h"
#include "petscmat.h"
#include "petscsnes.h"
#include "petscsystypes.h"
#include "petscvec.h"
#include "petscviewer.h"
#include "sundials/sundials_linearsolver.h"
#include "sundials/sundials_nonlinearsolver.h"
#include "sundials/sundials_nvector.h"
#include "sunmatrix/sunmatrix_dense.h"
#include <nvector/nvector_petsc.h>
#include <cstddef>
#include <iostream>
#include <memory>
#include <petscsys.h>
#include <sys/types.h>
// #include <petscmat.h>

namespace pele {
CVODEBDFOptimizer::CVODEBDFOptimizer(std::shared_ptr<pele::BasePotential> potential,
                                     const pele::Array<double> x0,
                                     double tol,
                                     double rtol,
                                     double atol)
    : GradientOptimizer(potential, x0, tol),
      cvode_mem(CVodeCreate(CV_BDF)), // create cvode memory
      N_size(x0.size()),
      t0(0),
      tN(10000000.0)
{
    // dummy t0
    // initialize petsc
    PetscInitializeNoArguments();
    // this assumes that the number of non zeros aren't different
    blocksize = 1;
    hessav = 10;
    // matrix initialization
    
    MatCreateSeqSBAIJ(PETSC_COMM_SELF, blocksize, N_size, N_size, hessav, NULL, &petsc_jacobian);

    VecCreateSeq(PETSC_COMM_SELF, N_size, &petsc_grad);

    // wrap nvector into petsc array
    nvec_grad_petsc = N_VMake_Petsc(petsc_grad);
    VecDuplicate(petsc_grad, &residual);
    // SNES create
    SNESCreate(PETSC_COMM_SELF, &snes);
    


    // we need to pass the potential pointer on to the
    // Jacobian function as the application context
    // The potential pointer allows us to use the base potential class to calculate the Jacobian
    // of the gradient i.e the hessian
    udataptr = &udata;
    SNESSetJacobian(snes, petsc_jacobian, petsc_jacobian, negative_hessian_wrapper,udataptr);
    // this is where we are;
    
    

    
    
    
    
    
    double t0 = 0;
    std::cout << x0 << "\n";
    Array<double> x0copy = x0.copy();
    Vec x0_petsc;
    PetscVec_eq_pele(x0_petsc, x0);
    VecView(x0_petsc, PETSC_VIEWER_STDOUT_SELF);
    x0_N = N_VMake_Petsc(x0_petsc);
    // initialization of everything CVODE needs


    std::cout << "print " << "\n";
    N_VPrint_Petsc(x0_N);
    N_VPrint(x0_N);

    Vec s = N_VGetVector_Petsc(x0_N);

    VecView(s, PETSC_VIEWER_STDOUT_SELF);
        
    int ret = CVodeInit(cvode_mem, f, t0, x0_N);

    // initialize userdata
    udata.rtol = rtol;
    udata.atol = atol;
    udata.nfev = 0;
    udata.nhev = 0;
    // current_grad = N_VClone_Petsc(nvec_grad_petsc);
    udata.pot_ = potential_;
    udata.neq = x0.size();
    udata.stored_grad = Array<double>(x0.size(), 0);
    CVodeSStolerances(cvode_mem, udata.rtol, udata.atol);
    ret = CVodeSetUserData(cvode_mem, &udata);
    // Linear solver method (use for non petsc purposes)
    // A = SUNDenseMatrix(N_size, N_size);
    // LS = SUNLinSol_Dense(x0_N, A);
    // CVodeSetLinearSolver(cvode_mem, LS, A);
    // CVodeSetJacFn(cvode_mem, Jac);
    // Non linear solver
    // std::cout << "here 1" << "\n";
    CVodeSetNonlinearSolver(cvode_mem, NLS);
    // g_ = udata.stored_grad;
    CVodeSetMaxNumSteps(cvode_mem, 1000000);
    CVodeSetStopTime(cvode_mem, tN);
};



void CVODEBDFOptimizer::one_iteration() {
    /* advance solver just one internal step */
    Array<double> xold = x_;
    int flag = CVode(cvode_mem, tN, x0_N, &t0, CV_ONE_STEP);
    iter_number_ +=1;
    double t;
    CVodeGetCurrentTime(cvode_mem, &t);
    CVodeGetDky(cvode_mem, t, 1, current_grad);
    double norm2 = N_VDotProd(current_grad, current_grad);
    x_ = pele_eq_N_Vector(x0_N);
    rms_ = (sqrt(norm2/udata.neq));
    f_ = udata.stored_energy;
    nfev_ = udata.nfev;
    Array<double> step = xold-x_;
};

CVODEBDFOptimizer::~CVODEBDFOptimizer() {
    VecDestroy(&petsc_grad);
    MatDestroy(&petsc_jacobian);
    N_VDestroy(nvec_grad_petsc);
    N_VDestroy(current_grad);
    CVodeFree(&cvode_mem);
    PetscFinalize();
};


/**
 * Function assumes an N_Vector wrapped in PETSc
 */
int f(double t, N_Vector y, N_Vector ydot, void *user_data) {

    UserData udata = (UserData) user_data;
    // wrap local vector as a pele vector
    Array<double> x_pele = pele_eq_PetscVec(N_VGetVector_Petsc(y));
    Array<double> g;

    Vec ydot_petsc = N_VGetVector_Petsc(ydot);
    double energy = udata->pot_->get_energy_gradient_sparse(x_pele, ydot_petsc);
    double *func_data;
    VecGetArray(ydot_petsc, &func_data);
    udata->nfev += 1;
#pragma simd
    for (size_t i = 0; i < udata->neq; ++i) {
      func_data[i] = -func_data[i];
    }
    VecAssemblyBegin(ydot_petsc);
    VecAssemblyEnd(ydot_petsc);
    // func data reversed
    // udata->stored_grad = (g);
    udata->stored_energy = energy;
    return 0;
}

int Jac(realtype t, N_Vector y, N_Vector fy, SUNMatrix J,
        void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
    UserData udata = (UserData) user_data;

    pele::Array<double> yw = pele_eq_N_Vector(y);
    Array<double> g = Array<double>(yw.size());
    Array<double> h = Array<double>(yw.size()*yw.size());
    udata->pot_->get_energy_gradient_hessian(pele_eq_N_Vector(y), g, h);
    udata->nhev += 1;
    double * hessdata = SUNDenseMatrix_Data(J);
    for (size_t i=0; i<h.size(); ++i) {
        hessdata[i] = -h[i];
    }
    return 0;
};


/**
 * Negative hessian wrapper. since the Jacobian we get is negative of the hessian
 */
PetscErrorCode negative_hessian_wrapper(SNES NLS,Vec x,  Mat Amat, Mat Precon, void* user_data)
{
    PetscFunctionBeginUser;
    UserData udata = (UserData) user_data;
    // wrap petsc data into pele vec
    Array<double> x_pele = pele_eq_PetscVec(x);
    x_pele = pele_eq_PetscVec(x);
    // negative_hessian_sparse handles all matrix assembly;
    udata->pot_->get_negative_hessian_sparse(x_pele, Precon);
    PetscFunctionReturn(0);
};

} // namespace pele
