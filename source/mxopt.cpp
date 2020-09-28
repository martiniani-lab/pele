#include "pele/mxopt.h"
#include "Eigen/src/Core/Matrix.h"
#include "cvode/cvode.h"
#include "pele/array.h"
#include "pele/base_potential.h"
#include "pele/debug.h"
#include "pele/eigen_interface.h"
#include "pele/lsparameters.h"
#include "pele/optimizer.h"
#include <algorithm>
#include <complex>
#include <memory>
#include <iostream>
#include <limits>

namespace pele {

MixedOptimizer::MixedOptimizer( std::shared_ptr<pele::BasePotential> potential,
                                const pele::Array<double> x0,
                                double tol, int T, double step, double conv_tol, double conv_factor)
    : GradientOptimizer(potential, x0, tol),
      H0_(1e-10),
      cvode_mem(CVodeCreate(CV_ADAMS)), // create cvode memory
      N_size(x_.size()),
      t0(0),
      tN(100.0),
      rtol(1e-3),
      atol(1e-3),
      xold(x_.size()),
      gold(x_.size()),
      step(x_.size()),
      T_(T),
      conv_tol_(conv_tol),
      conv_factor_(conv_factor),
      line_search_method(this, step)
{
    // set precision of printing
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
    // dummy t0
    double t0 = 0;
    std::cout << x0 << "\n";
    Array<double> x0copy = x0.copy();
    x0_N = N_Vector_eq_pele(x0copy);
    // initialization of everything CVODE needs
    int ret = CVodeInit(cvode_mem, f, t0, x0_N);
    // initialize userdata
    udata.rtol = rtol;
    udata.atol = atol;
    udata.nfev = 0;
    udata.nhev = 0;
    udata.pot_ = potential_;
    udata.stored_grad = Array<double>(x0.size(), 0);
    CVodeSStolerances(cvode_mem, udata.rtol, udata.atol);
    ret = CVodeSetUserData(cvode_mem, &udata);
    
    A = SUNDenseMatrix(N_size, N_size);
    LS = SUNLinSol_Dense(x0_N, A);
    
    CVodeSetLinearSolver(cvode_mem, LS, A);
    CVodeSetJacFn(cvode_mem, Jac);
    g_ = udata.stored_grad;
    CVodeSetMaxNumSteps(cvode_mem, 100000);
    CVodeSetStopTime(cvode_mem, tN);
    inv_sqrt_size = 1 / sqrt(x_.size());
    std::cout << OPTIMIZER_DEBUG_LEVEL << "optimizer debug level \n";

    
#if OPTIMIZER_DEBUG_LEVEL >= 1
    std::cout << "Mixed optimizer constructed" << "T=" << T_ << "\n";
#endif
    std::cout << H0_ << "\n";
}
/**
 * Does one iteration of the optimization algorithm
 */
void MixedOptimizer::one_iteration() {
    if (!func_initialized_) {
        initialize_func_gradient();
    }
    
    // declare that the hessian hasn't been calculated for this iteration
    hessian_calculated =false;
    // make a copy of the position and gradient
    xold.assign(x_);
    gold.assign(g_);    
    // copy the gradient into step

    step.assign(g_);
    // does a convexity check every T iterations
    if (iter_number_ % T_ == 0) {
#if OPTIMIZER_DEBUG_LEVEL>=3
        std::cout << "checking convexity" << "\n";        
#endif        
        usephase1 = convexity_check();        
    }

    if (usephase1) {
#if OPTIMIZER_DEBUG_LEVEL>=3
        std::cout << " computing phase 1 step" << "\n";        
#endif
        
        compute_phase_1_step(step);
    } else {
        
#if OPTIMIZER_DEBUG_LEVEL>=3
        std::cout << " computing phase 2 step" << "\n";        
#endif        
        
        compute_phase_2_step(step);
        line_search_method.set_xold_gold_(xold, gold);
        line_search_method.set_g_f_ptr(g_);
        double stepnorm = line_search_method.line_search(x_, step);
    }

    // should think of using line search method within the phase steps

    
        
    
    
    // update inverse hessian estimate
    update_H0_(xold, gold, x_, g_);

    
#if OPTIMIZER_DEBUG_LEVEL>=2
    std::cout << "mixed optimizer: " << iter_number_
              << " E " << f_
              << " rms " << rms_
              << " nfev " << nfev_
              << " step norm " << stepnorm << std::endl;
#endif
    iter_number_ += 1;

}


void MixedOptimizer::update_H0_(Array<double> x_old,
                                Array<double> & g_old,
                                Array<double> x_new,
                                Array<double> & g_new)
{
    // update the lbfgs memory
    // This updates s_, y_, rho_, and H0_, and k_
    double ys = 0;
    double yy = 0;

    Array<double> y_(x_old.size());
    Array<double> s_(x_old.size());
    
#pragma simd reduction( + : ys, yy)
    for (size_t j2 = 0; j2 < x_.size(); ++j2){
        y_[j2] = g_new[j2] - g_old[j2];
        s_[j2] = x_new[j2] - x_old[j2];
        ys += y_[j2] * s_[j2];
        yy += y_[j2] * y_[j2];
    }

#if OPTIMIZER_DEBUG_LEVEL >= 3    
    std::cout << ys << " YS value \n";
    std::cout << yy << "YY value \n";
#endif
    if (ys == 0.) {
        ys = 1.;

#if OPTIMIZER_DEBUG_LEVEL >= 2
        std::cout << "warning: resetting YS to 1." << std::endl;
#endif

    }
    if (yy == 0.) {
        yy = 1.;

        
#if OPTIMIZER_DEBUG_LEVEL >= 2
        std::cout << "warning: resetting YY to 1." << std::endl;
#endif
    }

    
    H0_ = ys / yy;
    
    // increment k
}


/**
 * resets the minimizer for usage again 
 */
void MixedOptimizer::reset(pele::Array<double> &x0) {
    if (x0.size() != x_.size()){
        throw std::invalid_argument("The number of degrees of freedom (x0.size()) cannot change when calling reset()");
    }    
    iter_number_ = 0;
    nfev_ = 0;
    x_.assign(x0);
    initialize_func_gradient();
}

/**
 * checks convexity in the region and updates the convexity flag accordingly
 * convexity flag 0 -> heavily concave function: Use an differential equation solver with adaptive stepping (Check RK45? Check Euler)
 * a good scale for the problem is the inverse hessian.
 * convexity flag true -> concavity below a tolerance. can be solved with newton steps with help
 * convexity flag false -> convex function, use a newton method. possible good flags (Tr(Hess) = Tr(|Hess|))
 */


bool MixedOptimizer::convexity_check() {
    
    hessian = get_hessian();
    hessian_calculated = true;  // pass on the fact that the hessian has been calculated
    Eigen::VectorXd eigvals = hessian.eigenvalues().real();
    minimum = eigvals.minCoeff();
    double maximum = eigvals.maxCoeff();
    double convexity_estimate = std::abs(minimum/maximum);
    
    if (minimum<0 and convexity_estimate >= conv_tol_) {
        // note minimum is negative
#if OPTIMIZER_DEBUG_LEVEL >= 1
        std::cout << "minimum less than 0" << " convexity tolerance condition not satisfied \n";
#endif        
        minimum_less_than_zero =true; 
        return true;
    }
    else if (convexity_estimate < conv_tol_ and minimum < 0) {

#if OPTIMIZER_DEBUG_LEVEL >= 1
        std::cout << "minimum less than 0" << " convexity tolerance condition satisfied \n";
#endif                
        scale = 1.;
        minimum_less_than_zero = true;
        return false;
        }
        
        else {scale =1;
#if OPTIMIZER_DEBUG_LEVEL >= 1
            std::cout << "minimum greater than 0" << " convexity tolerance condition satisfied \n";
#endif                        
            minimum_less_than_zero = false;
            return false;}

}


    /**
     * Gets the hessian. involves a dense hessian for now. #TODO replace with a sparse hessian
     */
Eigen::MatrixXd MixedOptimizer::get_hessian() {
    Array<double> hess(xold.size()*xold.size());
    Array<double> grad(xold.size());
    
    double e = potential_->get_energy_gradient_hessian(x_, grad, hess); // preferably switch this to sparse Eigen
    
    Eigen::MatrixXd hess_dense(xold.size(), xold.size());
    udata.nhev +=1;
    hess_dense.setZero();
    for (size_t i = 0; i < xold.size(); ++i) {
        for (size_t j=0; j < xold.size(); ++j) {
            hess_dense(i, j) = hess[i + grad.size()*j];
        }
    }
    
    return hess_dense;
    
}

// /**
//  * Phase 1 The problem does not look convex, Try solving using with an adaptive differential equationish approach
//  */
// void MixedOptimizer::compute_phase_1_step(Array<double> step) {
//     // use a a scaled steepest descent step
//     step *= -std::abs(H0_);
// }


/**
 * Phase 1 The problem does not look convex, Try solving using with sundials
 */
void MixedOptimizer::compute_phase_1_step(Array<double> step) {
    // use a a scaled steepest descent step
    /* advance solver just one internal step */
    Array<double> xold = x_;
    int flag = CVode(cvode_mem, tN, x0_N, &t0, CV_ONE_STEP);
    iter_number_ +=1;
    x_ = pele_eq_N_Vector(x0_N);
    g_ = udata.stored_grad;
    rms_ = (norm(g_)/sqrt(x_.size()));
    f_ = udata.stored_energy;
    nfev_ = udata.nfev;
    step = xold-x_;
}




/**
 * Phase 2 The problem looks convex enough to switch to a newton method
 */
void MixedOptimizer::compute_phase_2_step(Array<double> step) {
    if (hessian_calculated == false) {
        // we can afford to perform convexity checks at every steps
        // assuming the rate limiting cost is the hessian which is calculated
        // in the convexity check anyway
        convexity_check();
    }
    // this can mess up accuracy if we aren't close to a minimum preferably switch
    // sparse
    
    if (minimum_less_than_zero) {
        hessian -= conv_factor_*minimum*Eigen::MatrixXd::Identity(x_.size(), x_.size());
    }

    Eigen::VectorXd r(step.size());
    Eigen::VectorXd q(step.size());
    q.setZero();
    eig_eq_pele(r, step);
    // negative sign to switch direction
    q = -scale*hessian.ldlt().solve(r);
    pele_eq_eig(step, q);
}
}




