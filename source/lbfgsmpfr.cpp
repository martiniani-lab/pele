#include "pele/lbfgsmpfr.h"
#include <memory>
#include <iostream>
#include <limits>

namespace pele {

LBFGSMPFR::LBFGSMPFR( std::shared_ptr<pele::BasePotential> potential, const pele::Array<double> x0,
                      double tol, int M, int precision)
    : GradientOptimizerMPFR(potential, x0, tol, precision),
      M_(M),
      max_f_rise_(1e-4),
      use_relative_f_(false),
      rho_(M_),
      H0_(0.1),
      k_(0),
      alpha(M_),
      xold(x_.size()),
      gold(x_.size()),
      step(x_.size()),
      exact_g_(x_.size()),
      exact_gold(x_.size())
{
    // set precision of printing
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);

    inv_sqrt_size = 1 / sqrt(x_.size());

    // allocate space for s_ and y_
    s_ = Array<mpreal>(x_.size() * M_);
    y_ = Array<mpreal>(x_.size() * M_);
}

    /**
     * Do one iteration iteration of the optimization algorithm
     */
    void LBFGSMPFR::one_iteration()
    {
        if (!func_initialized_) {
            initialize_func_gradient();
        }

        // make a copy of the position and gradient
        for (int i =0; i < x_.size(); ++i) {
            xold[i] = x_[i];
        }

        gold.assign(g_);

        for (int i = 0; i < xold.size(); ++i) {
            xsum_small_equal(&(exact_gold[i]), &(exact_g_[i]));
        }

        
        
        // get the stepsize and direction from the LBFGS algorithm
        compute_lbfgs_step(step);
        // std::cout << iter_number_ << "\n";
        // reduce the stepsize if necessary
        mpreal stepnorm = backtracking_linesearch(step);

        // update the LBFGS memeory
        update_memory(xold, exact_gold, x_, exact_g_);

        // print some status information
        if ((iprint_ > 0) && (iter_number_ % iprint_ == 0)){
            std::cout << "lbfgs: " << iter_number_
                      << " E " << f_
                      << " rms " << rms_
                      << " nfev " << nfev_
                      << " step norm " << stepnorm << std::endl;
        }
        iter_number_ += 1;
    }

void LBFGSMPFR::update_memory(Array<mpreal> x_old,
                              std::vector<xsum_small_accumulator> & exact_gold,
                              Array<mpreal> x_new,
                              std::vector<xsum_small_accumulator> & exact_gnew)
{
    // update the lbfgs memory
    // This updates s_, y_, rho_, and H0_, and k_
    int klocal = k_ % M_;
    mpreal ys = 0;
    mpreal yy = 0;
        
    // define a dummy accumulator to help with exactly
    // calculating y
    xsum_small_accumulator sacc_dummy;
        
    // std::cout << "gradient difference" << "\n";
#pragma simd reduction( + : ys, yy)
    for (size_t j2 = 0; j2 < x_.size(); ++j2){
        size_t ind_j2 = klocal * x_.size() + j2;
        xsum_small_subtract_acc_and_set(&(exact_gnew[j2]),
                                        &(exact_gold[j2]),
                                        &sacc_dummy);
        y_[ind_j2] = xsum_small_round(&sacc_dummy);
        s_[ind_j2] = x_new[j2] - x_old[j2];
        ys += y_[ind_j2] * s_[ind_j2];
        yy += y_[ind_j2] * y_[ind_j2];
    }
    // std::cout << "end gradient difference" << "\n";
    if (ys == 0.) {
        if (verbosity_ > 0) {
            std::cout << "warning: resetting YS to 1." << std::endl;
        }
        ys = 1.;
    }
    
    rho_[klocal] = 1. / ys;

    if (yy == 0.) {
        if (verbosity_ > 0) {
            std::cout << "warning: resetting YY to 1." << std::endl;
        }
        yy = 1.;
    }
    H0_ = ys / yy;

    // increment k
    k_ += 1;
}

void LBFGSMPFR::compute_lbfgs_step(Array<mpreal> step)
{
    if (k_ == 0){
        // take a conservative first step
        mpreal gnorm = norm(g_);
        if (gnorm > 1.) {
            gnorm = 1. / gnorm;
        }
        mpreal prefactor =  -gnorm * H0_;
#pragma simd
        for (size_t j2 = 0; j2 < x_.size(); ++j2){
            step[j2] = prefactor * g_[j2];
        }
        return;
    }

    // copy the gradient into step
    step.assign(g_);

    int jmin = std::max(0, k_ - M_);
    int jmax = k_;
    int i;

    alpha.assign(mpreal(0.0));
    // loop backwards through the memory
    for (int j = jmax - 1; j >= jmin; --j) {
        i = j % M_;
        mpreal alpha_tmp = 0;
#pragma simd reduction(+ : alpha_tmp)
        for (size_t j2 = 0; j2 < step.size(); ++j2){
            alpha_tmp += rho_[i] * s_[i * step.size() + j2] * step[j2];
        }
#pragma simd
        for (size_t j2 = 0; j2 < step.size(); ++j2){
            step[j2] -= alpha_tmp * y_[i * step.size() + j2];
        }
        alpha[i] = alpha_tmp;
    }
    // Assumption of the choice here makes a difference
    // scale the step size by H0, invert the step to point downhill
#pragma simd
    for (size_t j2 = 0; j2 < step.size(); ++j2){
        step[j2] *= -H0_;
    }

        // loop forwards through the memory
        for (int j = jmin; j < jmax; ++j) {
            i = j % M_;
            mpreal beta = 0;
#pragma simd reduction(+ : beta)
            for (size_t j2 = 0; j2 < step.size(); ++j2){
                beta -= rho_[i] * y_[i * step.size() + j2] * step[j2];  // -= due to inverted step
            }
            mpreal alpha_beta = alpha[i] - beta;
#pragma simd
            for (size_t j2 = 0; j2 < step.size(); ++j2){
                step[j2] -= s_[i * step.size() + j2] * alpha_beta;  // -= due to inverted step
            }
        }

    
}

mpreal LBFGSMPFR::backtracking_linesearch(Array<mpreal> step)
{
    mpreal fnew;

    // if the step is pointing uphill, invert it
    if (dot(step, g_) > 0.){
        if (verbosity_ > 1) {
            std::cout << "warning: step direction was uphill.  inverting" << std::endl;
        }
#pragma simd
        for (size_t j2 = 0; j2 < step.size(); ++j2){
            step[j2] *= -1;
        }
    }

    mpreal factor = 1.;
    mpreal stepnorm = compute_pot_norm(step);

    // make sure the step is no larger than maxstep_
    if (factor * stepnorm > maxstep_) {
        factor = maxstep_ / stepnorm;
    }

    int nred;
    int nred_max = 10;
    for (nred = 0; nred < nred_max; ++nred){
#pragma simd
        for (size_t j2 = 0; j2 < x_.size(); ++j2){
            x_[j2] = xold[j2] + factor * step[j2];
        }
        compute_func_gradient(x_, fnew, exact_g_);

        for (int i = 0; i < exact_g_.size(); ++i) {
            g_[i] = xsum_small_round(&(exact_g_[i]));
        }

        
        mpreal df = fnew - f_;
        if (use_relative_f_) {
            mpreal absf = 1e-100;
            if (f_ != 0) {
                absf = abs(f_);
            }
            df /= absf;
        }
        if (df < max_f_rise_){
            break;
        } else {
            factor *= 0.5;
            if (verbosity_ > 2) {
                std::cout
                    << "energy increased by " << df
                    << " to " << fnew
                    << " from " << f_
                    << " reducing step norm to " << factor * stepnorm
                    << " H0 " << H0_ << std::endl;
            }
        }
    }

    if (nred >= nred_max){
        if (verbosity_ > 0) {
            std::cout << "warning: the line search backtracked too many times" << std::endl;
        }
    }

    f_ = fnew;
    rms_ = norm(g_) * inv_sqrt_size;
    return stepnorm * factor;
}

void LBFGSMPFR::reset(pele::Array<double> &x0)
{
    if (x0.size() != x_.size()){
        throw std::invalid_argument("The number of degrees of freedom (x0.size()) cannot change when calling reset()");
    }
    k_ = 0;
    iter_number_ = 0;
    nfev_ = 0;
    for (int i =0; i < x_.size(); ++i) {
        x_[i] = x0[i];
    }
    initialize_func_gradient();
}

}
