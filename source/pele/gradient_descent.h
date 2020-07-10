#ifndef _PELE_GD_H__
#define _PELE_GD_H__

#include <vector>
#include <memory>
#include "base_potential.h"
#include "array.h"
#include "optimizer.h"

namespace pele{

/**
 * An implementation of the standard gradient descent optimization algorithm in c++.
 * The learning rate (step size) is set via maxstep. Additionally, we use a backtracking linesearch.
 */
class GradientDescent : public GradientOptimizer{
private:
    double max_f_rise_; /**< The maximum the function is allowed to rise in a
                         * given stbisecting does a better jobep.  This is the criterion for the
                         * backtracking line search.
                         */
    bool use_relative_f_; /**< If True, then max_f_rise is the relative
                          * maximum the function is allowed to rise during
                          * a step.
                          * (f_new - f_old) / abs(f_old) < max_f_rise
                          */

    Array<double> xold; //!< Coordinates before taking a step
    Array<double> step; //!< Step size and direction
    double inv_sqrt_size; //!< The inverse square root the the number of components

public:
    /**
     * Constructor
     */
    GradientDescent( std::shared_ptr<pele::BasePotential> potential, const pele::Array<double> x0,
            double tol = 1e-4)
        : GradientOptimizer(potential, x0, tol),
        max_f_rise_(1e-4),
        use_relative_f_(false),
        xold(x_.size()),
        step(x_.size())
    {
        // set precision of printing
        std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);

        inv_sqrt_size = 1 / sqrt(x_.size());
    }

    /**
     * Destructor
     */
    virtual ~GradientDescent() {}

    /**
     * Do one iteration iteration of the optimization algorithm
     */
    void one_iteration()
    {
        if (!func_initialized_) {
            initialize_func_gradient();
        }

        // make a copy of the position and gradient
        xold.assign(x_);

        // Gradient defines step direction
        step = g_.copy();

        // reduce the stepsize if necessary
        double stepnorm = backtracking_linesearch(step);

        // print some status information
        if ((iprint_ > 0) && (iter_number_ % iprint_ == 0)){
            std::cout << "lbgs: " << iter_number_
                << " E " << f_
                << " rms " << rms_
                << " nfev " << nfev_
                << " step norm " << stepnorm << std::endl;
        }
        iter_number_ += 1;
    }

    // functions for setting the parameters
    inline void set_max_f_rise(double max_f_rise) { max_f_rise_ = max_f_rise; }

    inline void set_use_relative_f(int use_relative_f)
    {
        use_relative_f_ = (bool) use_relative_f;
    }

    /**
     * reset the optimizer to start a new minimization from x0
     *
     */
    virtual void reset(pele::Array<double> &x0)
    {
        if (x0.size() != x_.size()){
            throw std::invalid_argument("The number of degrees of freedom (x0.size()) cannot change when calling reset()");
        }
        iter_number_ = 0;
        nfev_ = 0;
        x_.assign(x0);
        initialize_func_gradient();
    }

private:

    /**
     * Take the step and do a backtracking linesearch if necessary.
     * Apply the maximum step size constraint and ensure that the function
     * does not rise more than the allowed amount.
     *
     * Same as LBFGS.
     */
    double backtracking_linesearch(Array<double> step)
    {
        double fnew;
        double factor = 1.;
        double stepnorm = compute_pot_norm(step);

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
            compute_func_gradient(x_, fnew, g_);

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

        if (nred >= nred_max){
            // possibly raise an error here
            if (verbosity_ > 0) {
                std::cout << "warning: the line search backtracked too many times" << std::endl;
            }
        }

        f_ = fnew;
        rms_ = norm(g_) * inv_sqrt_size;
        return stepnorm * factor;
    }

};
}

#endif
