#include "pele/cvode.h"
#include "cvode/cvode.h"
#include "cvode/cvode_ls.h"
#include "pele/optimizer.h"
#include "pele/debug.h"
#include "sundials/sundials_linearsolver.h"
#include "sundials/sundials_nvector.h"
#include <iostream>


namespace pele {
CVODEBDFOptimizer::CVODEBDFOptimizer(std::shared_ptr<pele::BasePotential> potential,
                                     const pele::Array<double> x0,
                                     double tol,
                                     double rtol,
                                     double atol)
    : GradientOptimizer(potential, x0, tol),
      cvode_mem(CVodeCreate(CV_BDF)), // create cvode memory
      N_size(x0.size())
{
    std::cout << "creating cvode optimizer" << "\n";
    // dummy t0
    double t0 = 0;
    x0_N = N_Vector_eq_pele(x0);
    
    // initialization of everything CVODE needs
    CVodeInit(cvode_mem, f, t0, x0_N);
    
    // initialize userdata
    udata->rtol = rtol;
    udata->atol = atol;
    udata->nfev = 0;
    udata->nhev = 0;
    
    CVodeSetUserData(cvode_mem, udata);
    CVodeSStolerances(cvode_mem, udata->rtol, udata->atol);
    
    A = SUNDenseMatrix(N_size, N_size);
    LS = SUNLinSol_Dense(x0_N, A);
    
    CVodeSetLinearSolver(cvode_mem, LS, A);
    CVodeSetJacFn(cvode_mem, Jac);
};



void CVODEBDFOptimizer::one_iteration() {
    /* advance solver just one internal step */
    CVode(cvode_mem, t1, x0_N, &t, CV_ONE_STEP);
    iter_number_ +=1;
    // compute rms that allows solver to get out.
    rms_ = compute_pot_norm(pele_eq_N_Vector(x0_N));
};

int CVODEBDFOptimizer::f(double t, N_Vector y, N_Vector ydot, void *user_data) {
    pele::Array<double> yw = pele_eq_N_Vector(y);
    udata->nfev += 1;
    Array<double> g = Array<double>(yw.size());
    potential_->get_energy_gradient(yw, g);
}


} // namespace pele
