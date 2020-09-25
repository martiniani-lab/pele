#ifndef _PELE_CVODE_OPT_H__
#define _PELE_CVODE_OPT_H__

#include "base_potential.h"
#include "array.h"
#include "debug.h"


#include <cstddef>
#include <iostream>
#include <vector>
#include <memory>
#include "base_potential.h"
#include "array.h"
#include "cvode/cvode_proj.h"
#include "debug.h"

// Eigen linear algebra library
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include "eigen_interface.h"                    


// line search methods 
#include "more_thuente.h"
#include "linesearch.h"
#include "optimizer.h"
#include "nwpele.h"
#include "backtracking.h"
#include "bracketing.h"
#include "sundials/sundials_linearsolver.h"
#include "sundials/sundials_matrix.h"
#include "sundials/sundials_nvector.h"
    
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
    double stored_energy;       // stored energy
    Array<double>    stored_grad;      // stored gradient. need to pass this on to 
    std::shared_ptr<pele::BasePotential> pot_;

} * UserData;

/**
 * Not exactly an optimizer but solves for the differential equation $ dx/dt = - \grad{V(x)} $ to
 * arrive at the trajectory to the corresponding minimum
 */
class CVODEBDFOptimizer : public GradientOptimizer {
private:
    UserData udata;
    void *cvode_mem; /* CVODE memory         */
    size_t N_size;
    SUNMatrix A;
    SUNLinearSolver LS;
    double t;
    double t1;
    N_Vector x0_N;
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
    NV_DATA_S(y) = x.data();
    return y;
}


/**
 * wraps the sundials data in a pele array and passes it on
 */
inline pele::Array<double> pele_eq_N_Vector(N_Vector x) {
    return pele::Array<double>(NV_DATA_S(x), N_VGetLength(x));
}

static int f(realtype t, N_Vector y, N_Vector ydot, void *user_data);
static int Jac(realtype t, N_Vector y, N_Vector fy, SUNMatrix J,
               void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

} // namespace pele
#endif