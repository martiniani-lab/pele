#include "pele/cvode.h"
#include "autodiff/reverse/eigen.hpp"
#include "cvode/cvode.h"
#include "cvode/cvode_ls.h"
#include "nvector/nvector_serial.h"
#include "pele/optimizer.h"
#include "pele/debug.h"
#include "sundials/sundials_linearsolver.h"
#include "sundials/sundials_nvector.h"
#include "sunmatrix/sunmatrix_dense.h"
#include <cstddef>
#include <iostream>
#include <memory>


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
      tN(10000.0)
{
    // dummy t0
    double t0 = 0;
    std::cout << x0 << "\n";
    Array<double> x0copy = x0.copy();
    x0_N = N_Vector_eq_pele(x0copy);
    if (check_flag((void *)cvode_mem, "CVodeCreate", 0)) return;
    // initialization of everything CVODE needs
    int ret = CVodeInit(cvode_mem, f, t0, x0_N);
    if (check_flag((void *)&ret, "CVodeInit", 0)) return;
    // initialize userdata
    udata.rtol = rtol;
    udata.atol = atol;
    udata.nfev = 0;
    udata.nhev = 0;
    udata.pot_ = potential_;
    udata.stored_grad = Array<double>(x0.size(), 0);
    CVodeSStolerances(cvode_mem, udata.rtol, udata.atol);
    if (check_flag((void *)&ret, "CVodeSStolerances", 0)) return;
    ret = CVodeSetUserData(cvode_mem, &udata);
    if (check_flag((void *)&ret, "CVodeuserdata", 0)) return;
    
    A = SUNDenseMatrix(N_size, N_size);
    if (check_flag((void *)A, "SUNDenseMatrix", 0)) return;
    LS = SUNLinSol_Dense(x0_N, A);
    if (check_flag((void *)LS, "SUNLinSol_Dense", 0)) return;
    
    CVodeSetLinearSolver(cvode_mem, LS, A);
    if (check_flag((void *)&ret, "CVodeSetLinearSolver", 1)) return;
    CVodeSetJacFn(cvode_mem, Jac);
    g_ = udata.stored_grad;
    CVodeSetMaxNumSteps(cvode_mem, 100000);
    CVodeSetStopTime(cvode_mem, tN);
};



void CVODEBDFOptimizer::one_iteration() {
    /* advance solver just one internal step */
    Array<double> xold = x_;
    int flag = CVode(cvode_mem, tN, x0_N, &t0, CV_ONE_STEP);
    if(check_flag(&flag, "CVode", 1)) return;
    iter_number_ +=1;
    x_ = pele_eq_N_Vector(x0_N);
    g_ = udata.stored_grad;
    rms_ = (norm(g_)/sqrt(x_.size()));
    f_ = udata.stored_energy;
    nfev_ = udata.nfev;
    Array<double> step = xold-x_;

};

int f(double t, N_Vector y, N_Vector ydot, void *user_data) {

    UserData udata = (UserData) user_data;
    pele::Array<double> yw = pele_eq_N_Vector(y);
    udata->nfev += 1;
    
    Array<double> g = Array<double>(yw.size());
    double energy = udata->pot_->get_energy_gradient(yw, g);
    double *fdata = NV_DATA_S(ydot);
    for (size_t i = 0; i < yw.size(); ++i) {
        fdata[i] = -g[i];
    }
    udata->stored_grad = (g.copy());
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
    double * hessdata = SUNDenseMatrix_Data(J);
    for (size_t i=0; i<h.size(); ++i) {
        hessdata[i] = -h[i];
    }
    return 0;
};
static int check_flag(void *flagvalue, const char *funcname, int opt) {
    int *errflag;

    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if (opt == 0 && flagvalue == NULL) {
        fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
                funcname);
        return(1); }

    /* Check if flag < 0 */
    else if (opt == 1) {
        errflag = (int *) flagvalue;
        if (*errflag < 0) {
            fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
                    funcname, *errflag);
                return(1); }}

        /* Check if function returned NULL pointer - no memory allocated */
        else if (opt == 2 && flagvalue == NULL) {
            fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
                    funcname);
            return(1);}
        return(0);
    }


} // namespace pele
