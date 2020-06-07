#ifndef _PELE_OPTIMIZER_MPFR_H__
#define _PELE_OPTIMIZER_MPFR_H__

#include "base_potential.h"
#include "array.h"
#include "xsum.h"
#include <vector>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <memory>
#include <mpreal.h>
using mpfr::mpreal;


namespace pele{

    /**
 * this defines the basic interface for optimizers.  All pele optimizers
 * should derive from this class.
 */
class OptimizerMPFR {
public:
    /**
     * virtual destructor
     */
    virtual ~OptimizerMPFR() {}

    virtual void one_iteration() = 0;

    /**
     * Run the optimization algorithm until the stop criterion is satisfied or
     * until the maximum number of iterations is reached
     */
    virtual void run() = 0;

    /**
     * Run the optimization algorithm for niter iterations or until the
     * stop criterion is satisfied
     */
    virtual void run(int const niter) = 0;

    /**
     * accessors
     */
    // inline virtual Array<double> get_x() const = 0;
    // inline virtual Array<double> get_g() const = 0;
    inline virtual double get_f() const = 0;
    inline virtual double get_rms() const = 0;
    inline virtual int get_nfev() const = 0;
    inline virtual int get_niter() const = 0;
    inline virtual bool success() = 0;
};

/**
 * This defines the basic interface for optimizers.  All pele optimizers
 * should derive from this class.
 */
class GradientOptimizerMPFR : public OptimizerMPFR {
protected :
    // input parameters
    /**
     * A pointer to the object that computes the function and gradient
     */
    std::shared_ptr<pele::BasePotential> potential_;

    mpreal tol_; /**< The tolerance for the rms gradient */
    mpreal maxstep_; /**< The maximum step size */

    int maxiter_; /**< The maximum number of iterations */
    int iprint_; /**< how often to print status information */
    int verbosity_; /**< How much information to print */

    int iter_number_; /**< The current iteration number */
    int nfev_; /**< The number of function evaluations */
    int precision_;

    // variables representing the state of the system
    Array<mpreal> x_; /**< The current coordinates */
    mpreal f_; /**< The current function value */
    Array<mpreal> g_; /**< The current gradient */
    mpreal rms_; /**< The root mean square of the gradient */
    const int digits;

    /**
     * This flag keeps track of whether the function and gradient have been
     * initialized.  This allows the initial function and gradient to be computed
     * outside of the constructor and also allows the function and gradient to
     * be passed rather than computed.  The downside is that it complicates the
     * logic because this flag must be checked at all places where the gradient,
     * function value, or rms can be first accessed.
     */
    bool func_initialized_;

public :
    GradientOptimizerMPFR(std::shared_ptr<pele::BasePotential> potential,
                          const pele::Array<double> x0, double tol=1e-4, int precision = 53) 
        : potential_((mpreal::set_default_prec(mpfr::digits2bits(precision)), potential)),
          digits(precision),
          tol_(tol),
          maxstep_(0.1),
          maxiter_(1000),
          iprint_(-1),
          verbosity_(0),
          iter_number_(0),
          nfev_(0),
          f_(0.),
          g_(x0.size()), 
          rms_(1e10),
          func_initialized_(false)
          
    {    
        x_ = Array<mpreal>(x0.size());
                                      
        for (int i =0; i < x0.size(); ++i) {
            x_[i] = x0[i];
        }
    }
        
    virtual ~GradientOptimizerMPFR() {}

    /**
     * Do one iteration iteration of the optimization algorithm
     */
    virtual void one_iteration() = 0;

    /**
     * Run the optimization algorithm until the stop criterion is satisfied or
     * until the maximum number of iterations is reached
     */
    void run(int const niter)
    {
        if (! func_initialized_){
            // note: this needs to be both here and in one_iteration
            initialize_func_gradient();
        }
        // iterate until the stop criterion is satisfied or maximum number of
        // iterations is reached
        for (int i = 0; i < niter; ++i) {
            if (stop_criterion_satisfied()) {
                break;
            }
            one_iteration();
        }
    }

    /**
     * Run the optimzation algorithm for niter iterations or until the
     * stop criterion is satisfied
     */
    void run()
    {
        run(maxiter_ - iter_number_);
    }

    /**
     * Set the initial func and gradient.  This can be used
     * to avoid one potential call
     */
    virtual void set_func_gradient(double f, Array<mpreal> grad)
    {
        if (grad.size() != g_.size()){
            throw std::invalid_argument("the gradient has the wrong size");
        }
        if (iter_number_ > 0){
            std::cout << "warning: setting f and grad after the first iteration.  this is dangerous.\n";
        }
        // copy the function and gradient
        f_ = f;
        g_.assign(grad);
        rms_ = norm(g_) / sqrt(g_.size());
        func_initialized_ = true;
    }

    inline virtual void reset(pele::Array<double> &x0)
    {
        throw std::runtime_error("GradientOptimizer::reset must be overloaded");
    }

    // functions for setting the parameters
    inline void set_tol(double tol) { tol_ = tol; }
    inline void set_maxstep(double maxstep) { maxstep_ = maxstep; }
    inline void set_max_iter(int max_iter) { maxiter_ = max_iter; }
    inline void set_iprint(int iprint) { iprint_ = iprint; }
    inline void set_verbosity(int verbosity) { verbosity_ = verbosity; }


    // functions for accessing the status of the optimizer
    inline Array<double> get_x()  {
        Array<double> x0(x_.size());
        for (int i=0; i < x_.size(); ++i) {
            x0[i] = x_[i].toDouble();
        }
        return x0; }

    inline Array<double> get_g()  {
        Array<double> g0(g_.size());
        for (int i=0; i < g_.size(); ++i) {
            g0[i] = g_[i].toDouble();
        }
        return g0; }
    inline double get_f() const { return f_.toDouble(); }
    inline double get_rms() const { return rms_.toDouble(); }
    inline int get_nfev() const { return nfev_; }
    inline int get_niter() const { return iter_number_; }
    inline int get_maxiter() const { return maxiter_; }
    inline double get_maxstep() { return maxstep_.toDouble(); }
    inline double get_tol() const {return tol_.toDouble();}
    inline bool success() { return stop_criterion_satisfied(); }

    /**
     * Return true if the termination condition is satisfied, false otherwise
     */
    virtual bool stop_criterion_satisfied()
    {
        if (! func_initialized_) {
            initialize_func_gradient();
        }
        return rms_ <= tol_;
    }

protected :

    /**
     * Compute the func and gradient of the objective function
     */
    void compute_func_gradient(Array<mpreal> x, mpreal & func,
                               Array<mpreal> & gradient)
    {
        nfev_ += 1;

        // pass the arrays to the potential
        Array<double> doublegrad(gradient.size());

        Array<double> x0(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            x0[i] = x[i].toDouble();
        }
        func = potential_->get_energy_gradient(x0, doublegrad);
        for (int i = 0; i < gradient.size(); ++i) {
            gradient[i] = doublegrad[i];
        }

    }

    void compute_func_gradient(Array<mpreal> x, mpreal & func,
                               std::vector<xsum_small_accumulator> & gradient)
    {
        nfev_ += 1;
        Array<double> x0(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            x0[i] = x[i].toDouble();
        }

        // pass the arrays to the potential
        func = potential_->get_energy_gradient(x0, gradient);
    }
    /**
     * compute the initial func and gradient
     */
    virtual void initialize_func_gradient()
    {
        // compute the func and gradient at the current locations
        // and store them
        compute_func_gradient(x_, f_, g_);
        rms_ = norm(g_) / sqrt(x_.size());
        func_initialized_ = true;
    }

    /**
     * Compute the norm defined by the potential
     */
    mpreal compute_pot_norm(Array<mpreal> x)
    {
        return norm(x);
    }

};
}

#endif
