#ifndef _PELE_GD_H__
#define _PELE_GD_H__

#include <vector>
#include <memory>
#include "base_potential.hpp"
#include "array.hpp"
#include "optimizer.hpp"


// line search methods 
#include "more_thuente.hpp"
#include "linesearch.hpp"
#include "optimizer.hpp"
#include "nwpele.hpp"
#include "backtracking.hpp"
#include "bracketing.hpp"

namespace pele{

/**
 * An implementation of the standard gradient descent optimization algorithm in c++.
 * The learning rate (step size) is set via initial_step. defaults to backtracking line search 
 */
class GradientDescent : public GradientOptimizer{
private:
    Array<double> xold; //!< Coordinates before taking a step
    Array<double> step; //!< Step direction
    double inv_sqrt_size; //!< The inverse square root the the number of components
    BacktrackingLineSearch line_search_method; // Line search method
public:
    /**
     * Constructor
     */
    GradientDescent( std::shared_ptr<pele::BasePotential> potential, const pele::Array<double> x0,
                     double tol = 1e-4, double stepsize = 1e-4)
        : GradientOptimizer(potential, x0, tol),
          xold(x_.size()),
          step(x_.size()),
          line_search_method(this, stepsize)
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
        //  really need to refactor and clean this up
        Array<double> gold = g_.copy();
        // Gradient defines step direction
        step = -g_.copy();
        
        // reduce the stepsize if necessary
        line_search_method.set_xold_gold_(xold, gold);
        line_search_method.set_g_f_ptr(g_);
        double stepnorm = line_search_method.line_search(x_, step);
        Array<double> gdiff = gold-g_;
        Array<double> xdiff = xold - x_;

        // inverse hessian estimate;
        double H0 = dot(gdiff, xdiff)/dot(gdiff, gdiff);
        // if (H0<line_search_method.get_initial_stpsize()) {
        //     std::cout << "warning initial step size larger than curvature estimate, decrease stepsize" << "\n";
        //     std::cout << H0 << "H0 value \n";
        //     std::cout << H0*g_ << "what the step should be";
        // }
        // print some status information
        if ((iprint_ > 0) && (iter_number_ % iprint_ == 0)){
            std::cout << "steepest descent: " << iter_number_
                      << " E " << f_
                      << " rms " << rms_
                      << " nfev " << nfev_
                            << " step norm " << stepnorm << std::endl;
              }
         iter_number_ += 1;
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
};
}

#endif
