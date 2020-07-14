#ifndef _PELE_LINESEARCH_H__
#define _PELE_LINESEARCH_H__

/**
 * Should include line search methods for use in various optimizers
 * refer to Nocedal and Wright Chapter 3 for information.
 * preferably should be header only, 
 * 
 * 
 *
 */

#include <vector>
#include <memory>
#include "array.h"
#include "base_potential.h"
#include "optimizer.h"






// Eigen Linear Algebra library

// #include <Eigen/Dense>
// #include <Eigen/SparseCore>
// #include <Eigen/SparseCholesky>





namespace pele {


/**
 * Abstract base class for Line Search Methods in optimizers. All Line search methods
 * should ideally derive from this class
 */
class LineSearch
{

protected:
    // // stored energy and gradient at the computed stepsize
    Array<double> end_gradient;
    double end_energy;
    double stepnorm;
    // Base potential pointer
    double max_stepnorm_;
    // Optimizer class
    GradientOptimizer * opt_;
public:
    LineSearch(GradientOptimizer * opt, double max_stepnorm):
        opt_(opt),
        max_stepnorm_(max_stepnorm)
    {std::cout << "class constructed " << "\n";};
    virtual ~LineSearch () {std::cout << "destructor called" << "\n";};


    // * Computes the step size given a step and a position
    // * 
    virtual double line_search(Array<double> & x, Array<double> step){
        throw std::runtime_error("line_search must be overloaded in the LineSearch class");
    }

    // // variable settings
    inline void set_max_stepnorm_(double max_stepnorm_in) { max_stepnorm_ = max_stepnorm_in; }
};


/**
 * line search method as implemented in backtracking line search in pele originally for LBFGS: 
 * prevents the energy from rising above a certain value as compared to the usual backtracking linesearch method
 * which implements sufficient decrease condition
 * May also fail if the convergence is quadratic
 *
 * ALso it is assumed that any potential methods used by this would be defined from an interface to the optimizer
 * so that the evaluations can be tracked
 */

class OldLineSearch : public LineSearch
{
public:
    OldLineSearch( GradientOptimizer * opt, double maxstepnorm, 
                   double max_f_rise=1e-4, double nred_max = 10, bool use_relative_f = true) :
        LineSearch(opt, maxstepnorm),
        max_f_rise_(max_f_rise),
        nred_max_(nred_max),
        use_relative_f_(use_relative_f),
        verbosity_(opt->get_verbosity_())
    {
        
    };

private:
    // maximum the function can rise to
    double max_f_rise_;
    // maximum number of backtracking iterations
    int nred_max_;
    // whether to use relative f or absolute f in determining how much larger the function can go
    bool use_relative_f_;
    Array<double> xold;
    // verbosity, should be inherited from the optimizer
    int verbosity_;
    Array<double> x_;
    Array<double> g_;
    Array<double> gold_;
    Array<double> xold_;
    // Set an f pointer
    double * f_ptr_;
    double f_;
    // GradientOptimizer * gopt_ ; // hacky but I don't know any good way to specify opt is gradient optimizer
    // This ends up giving me a version of this with version

public:
    // I'm putting this here instead of putting it elsewhere since I ended up getting an undefined symbol error
    // which means there are some linker issues?
    double line_search(Array<double>  &x, Array<double> step) ;
    // This is slightly convoluted because we want to set the variables without setting them as args in the linesearch
    // method. 
    
    // helper function to assign xold and gold.
    
    // These should reflect the xold in the old version
    void set_xold_gold_(Array<double> &xold, Array<double>  &gold) {
        gold_=gold;
        xold_=xold;
    };
    // helper function to assign g_ and f
    void set_g_f_ptr(Array<double> &g) {g_ = g;};
};


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










#endif  /* End line search methods */