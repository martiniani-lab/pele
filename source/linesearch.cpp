#include "pele/linesearch.hpp"
#include <memory>
#include <iostream>
#include <limits>


namespace pele {

double OldLineSearch::line_search(Array<double> &x, Array<double> step) {
    double fnew;
    x_ = x;
    f_= opt_->get_f();
    // This basically makes sure the f_ is assigned from linesearch and not
    // from before
    // Always point the step in the direction of the gradient:
    // May fail if the convergence is quadratic
    if (dot(step, g_) > 0.){
        if (verbosity_>1) {
            std::cout << "warning: step direction was uphill.  inverting" << std::endl;
        }
#pragma simd
        for (size_t j2 = 0; j2 < step.size(); ++j2){
            step[j2] *= -1;
        }
    }
    double factor = 1.;
    int nred;
    double stepnorm = opt_->compute_pot_norm(step);
    // make sure the stepnorm is no larger than maxstepnorm
    if (factor * stepnorm > max_stepnorm_) {
        factor = max_stepnorm_ / stepnorm;
    }
    for (nred = 0; nred < nred_max_; ++nred){
#pragma simd
        for (size_t j2 = 0; j2 < x_.size(); ++j2){
            x_[j2] = xold_[j2] + factor * step[j2];
        }
        opt_->compute_func_gradient(x_, fnew, g_);
        
        double df = fnew - f_;
        if (use_relative_f_) {
            double absf = 1e-100;
            if (f_ != 0) {
                absf = std::abs(f_);
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
                    << std::endl;
            }
        }
    }
    if (nred >= nred_max_){
        // possibly raise an error here
        if (verbosity_ > 0) {
            std::cout << "warning: the line search backtracked too many times" << std::endl;
        }
    }

    opt_->set_f(fnew);
    // rms_ = norm(g_) * inv_sqrt_size;
    opt_->set_rms(norm(g_)/sqrt(x_.size()));
    return stepnorm * factor;
};

}