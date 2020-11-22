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
#include "nvector/nvector_petsc.h"
#include "optimizer.hpp"
#include "nwpele.hpp"
#include "backtracking.hpp"
#include "bracketing.hpp"




#include <petscmat.h>
#include <sunnonlinsol/sunnonlinsol_petscsnes.h>
    
#include "petscsys.h"
#include "petscsystypes.h"
#include "petscvec.h"
#include "sundials/sundials_linearsolver.h"
#include "sundials/sundials_matrix.h"
#include "sundials/sundials_nonlinearsolver.h"
#include "sundials/sundials_nvector.h"
#include "cvode/cvode_proj.h"
    
#include <cvode/cvode.h>               /* access to CVODE                 */
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
  
  Vec last_gradient;
  std::shared_ptr<pele::BasePotential> pot_;
  size_t neq;                   // number of equations set so that we don't have to keep setting this over and over again
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
    Mat petsc_jacobian;
    Vec petsc_grad;
    N_Vector current_grad;
       
    N_Vector nvec_grad_petsc;
    Vec residual;
    PetscInt blocksize;
    // average number of non zeros per block for memory allocation purposes
    PetscInt hessav;

    SNES  snes;
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
    ~CVODEBDFOptimizer();
    inline int get_nhev() const { return udata.nhev;}

protected:
    double H02;

  };

/**
 * creates a new N_vector_petsc array array that wraps around the pele array data
 */
inline N_Vector N_Vector_eq_pele(pele::Array<double> x)
{
    N_Vector y;
    Vec y_petsc;
    VecCreateSeq(PETSC_COMM_SELF,x.size(), &y_petsc);
    for (size_t i = 0; i < x.size(); ++i) {
        VecSetValue(y_petsc, i, x[i], INSERT_VALUES);
    }
    VecAssemblyBegin(y_petsc);
    VecAssemblyEnd(y_petsc);
    y = N_VMake_Petsc(y_petsc);
    return y;
}


/**
 * Creates a new pele array with data belonging to an N_Vector petsc arra
 */

inline pele::Array<double> pele_eq_N_Vector(N_Vector x) {
    Vec x_petsc = N_VGetVector_Petsc(x);
    double * x_data;
    VecGetArray(x_petsc, &x_data);
    return pele::Array<double>(x_data, N_VGetLength(x)).copy();
}



/**
 * gets the array data and wraps it into a pele Array.
 * Note: pele arrays are single processor only Note: does not
 * work if restore array is not called later
 */
inline pele::Array<double> pele_eq_PetscVec(Vec x) {
    double *x_arr;
    VecGetArray(x, &x_arr);
    int length;
    VecGetLocalSize(x, &length);
    return pele::Array<double>(x_arr, length);
}

/**
 * Creates a new Vec object with data from pele array (cannot wrap data due to
 * PETSc data encapsulation)
 * involves creation of new stuff
 */
inline void PetscVec_eq_pele(Vec & x_petsc, pele::Array<double> x) {
    VecCreateSeq(PETSC_COMM_SELF,x.size(), &x_petsc);
    for (auto i = 0; i < x.size(); ++i) {
      VecSetValue(x_petsc, i, x[i], INSERT_VALUES);
    }
    VecAssemblyBegin(x_petsc);
    VecAssemblyEnd(x_petsc);
  }

  int f(realtype t, N_Vector y, N_Vector ydot, void *user_data);
  int Jac(realtype t, N_Vector y, N_Vector fy, SUNMatrix J, void *user_data,
        N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);
static int Jac2(realtype t, N_Vector y, N_Vector fy, SUNMatrix J,
                void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);
static int f2(realtype t, N_Vector y, N_Vector ydot, void *user_data);
/**
 * wrapper around negative hessian which allows for faster computations
 */
PetscErrorCode negative_hessian_wrapper(SNES NLS,Vec x,  Mat Amat, Mat Precon, void* user_data);
} // namespace pele
#endif