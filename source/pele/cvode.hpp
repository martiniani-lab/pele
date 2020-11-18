#ifndef _PELE_CVODE_OPT_H__
#define _PELE_CVODE_OPT_H__

#include "base_potential.hpp"
#include "array.hpp"
#include "debug.hpp"


#include <cstddef>
#include <iostream>
#include <vector>
#include <memory>
#include "base_potential.hpp"
#include "array.hpp"
#include "debug.hpp"

// Eigen linear algebra library
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include "eigen_interface.hpp"                    


// line search methods 
#include "more_thuente.hpp"
#include "linesearch.hpp"
#include "optimizer.hpp"
#include "nwpele.hpp"
#include "backtracking.hpp"
#include "bracketing.hpp"




#include <petscmat.h>
#include <sunnonlinsol/sunnonlinsol_petscsnes.h>
    
#include "petscvec.h"
#include "sundials/sundials_linearsolver.h"
#include "sundials/sundials_matrix.h"
#include "sundials/sundials_nonlinearsolver.h"
#include "sundials/sundials_nvector.h"
#include "cvode/cvode_proj.h"
    
#include <cvode/cvode.h>               /* access to CVODE                 */
#include <nvector/nvector_serial.h>    /* access to serial N_Vector       */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix       */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver */    

extern "C" {
#include "xsum.h"
}




namespace pele {

/**
 * user data passed to CVODE
 */
typedef struct UserData_
{

    double rtol; /* integration tolerances */
    double atol;
    size_t nfev;                // number of gradient(function) evaluations
    size_t nhev;                // number of hessian (jacobian) evaluations
    double stored_energy = 0;       // stored energy
    Array<double>    stored_grad;      // stored gradient. need to pass this on to
    std::shared_ptr<pele::BasePotential> pot_;
} * UserData;


// /**
//  * Data that helps determine the context in which petsc can calculate a hessian
//  * Contains
//  * 1. shared pointer to base potential
//  * 2. reference to coords array
//  */
// typedef struct hessdata_
// {
    
// } * hessdata;


/**
 * wrapper around get_energy_gradient_hessian_sparse that helps get the hessian matrix for petsc
 */
PetscErrorCode hessian_wrapper(SNES NLS,Vec x,  Mat Amat, Mat Precon, void* user_data);
/**
 * Not exactly an optimizer but solves for the differential equation $ dx/dt = - \grad{V(x)} $ to
 * arrive at the trajectory to the corresponding minimum
 */
class CVODEBDFOptimizer : public GradientOptimizer {
private:
    UserData_ udata;
    void *cvode_mem; /* CVODE memory         */
    size_t N_size;
    SUNMatrix A;
    SUNLinearSolver LS;
    SUNNonlinearSolver NLS;
    double t0;
    double tN;
    void * udataptr;
    N_Vector x0_N;
    Array<double> xold;
    // sparse calculation initializers
    Mat petsc_hess;
    Vec petsc_grad;
    N_Vector nvec_grad_petsc;
    Vec residual;
    PetscInt blocksize;
    // average number of non zeros per block for memory allocation purposes
    PetscInt hessav;

    SNES                 snes;
public:
    void one_iteration();
    // int f(realtype t, N_Vector y, N_Vector ydot, void *user_data);
    // static int Jac(realtype t, N_Vector y, N_Vector fy, SUNMatrix J,
    //                void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);
    CVODEBDFOptimizer(std::shared_ptr<pele::BasePotential> potential,
                      const pele::Array<double> x0,
                      double tol=1e-5,
                      double rtol=1e-4,
                      double atol=1e-4);
    inline int get_nhev() const { return udata.nhev;}

protected:
    double H02;

};

/**
 * creates a new N_vector array that wraps around the pele array data
 */
inline N_Vector N_Vector_eq_pele(pele::Array<double> x)
{
    N_Vector y;
    y = N_VNew_Serial(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        NV_Ith_S(y, i) = x[i];
    }
    // NV_DATA_S(y) = x.copy().data();
    return y;
}


/**
 * wraps the sundials data in a pele array and passes it on
 */
inline pele::Array<double> pele_eq_N_Vector(N_Vector x) {
    Array<double> y = Array<double>(N_VGetLength(x));
    for (size_t i = 0; i < y.size(); ++i) {
        y[i] = NV_Ith_S(x, i);
    }
    return pele::Array<double>(NV_DATA_S(x), N_VGetLength(x)).copy();
}



/**
 * gets the array data and wraps it into a pele Array.
 * Note: pele arrays are single processor only
 */
inline pele::Array<double> pele_eq_PetscVec(Vec x) {
    double *x_arr;
    VecGetArray(x, &x_arr);
    int length;
    VecGetLocalSize(x, &length);
    return pele::Array<double>(x_arr, length);
}

int f(realtype t, N_Vector y, N_Vector ydot, void *user_data);
int Jac(realtype t, N_Vector y, N_Vector fy, SUNMatrix J, void *user_data,
        N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);
static int Jac2(realtype t, N_Vector y, N_Vector fy, SUNMatrix J,
                void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);
static int f2(realtype t, N_Vector y, N_Vector ydot, void *user_data);

} // namespace pele
#endif