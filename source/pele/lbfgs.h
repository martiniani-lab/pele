#ifndef _PELE_LBFGS_H__
#define _PELE_LBFGS_H__

#include <vector>
#include <memory>
#include "base_potential.h"
#include "array.h"




// Eigen linear algebra library
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include "eigen_interface.h"

// line search methods 
#include "lsparameters.h"
#include "more_thuente.h"
#include "linesearch.h"
#include "optimizer.h"
#include "nwpele.h"
#include "backtracking.h"
#include "bracketing.h"


extern "C" {
#include "xsum.h"
}

namespace pele{



/**
 * An implementation of the LBFGS optimization algorithm in c++.
 */
class LBFGS : public GradientOptimizer{
private:

    int M_; /**< The length of the LBFGS memory */
    double max_f_rise_; /**< The maximum the function is allowed to rise in a
                         * given step.  This is the criterion for the
                         * backtracking line search.
                         */
    bool use_relative_f_; /**< If True, then max_f_rise is the relative
                          * maximum the function is allowed to rise during
                          * a step.
                          * (f_new - f_old) / abs(f_old) < max_f_rise
                          */


    // places to store the lbfgs memory
    /** s_ stores the changes in position for the previous M steps */
    Array<double> s_;
    /** y_ stores the changes in gradient for the previous M steps */
    Array<double> y_;
    /** rho stores 1/dot(y_, s_) for the previous M steps */
    Array<double> rho_;
    
    
    /**
     * H0 is the initial estimate for the diagonal component of the inverse Hessian.
     * It is an input parameter, but the estimate is improved during the run.
     * H0 is a scalar, which means that we use the same value for all degrees of freedom.
     */
    double H0_;
    int t_;
    int k_; /**< Counter for how many times the memory has been updated */

    Array<double> alpha; //!< Alpha used when looping through LBFGS memory

    Array<double> xold; //!< Coordinates before taking a step
    Array<double> gold; //!< Gradient before taking a step
    std::vector<xsum_small_accumulator> exact_gold;
    std::vector<xsum_small_accumulator> exact_g_;
    Array<double> step; //!< Step size and direction
    double inv_sqrt_size; //!< The inverse square root the the number of components
    // Preconditioning
    int T_;      // number of steps after which the hessian is updated
    std::shared_ptr<Eigen::ColPivHouseholderQR<Eigen::MatrixXd>> solver;                  // solver for H x = b
    Eigen::VectorXd update_solver(Eigen::VectorXd r);                                     // updates the solver with the new hessian
    Eigen::MatrixXd get_hessian_sparse();
    // Calculates hess + delta I where delta makes the new eigenvalue positive
    Eigen::MatrixXd get_hessian_sparse_pos();
    Eigen::MatrixXd saved_hessian;
    BacktrackingLineSearch line_search_method;

public:
    /**
     * Constructor
     */
    LBFGS( std::shared_ptr<pele::BasePotential> potential, const pele::Array<double> x0,
           double tol = 1e-4, int M = 4, int T=1);

    /**
     * Destructor
     */
    virtual ~LBFGS() {}

    /**
     * Do one iteration iteration of the optimization algorithm
     */
    void one_iteration();

    // functions for setting the parameters
    inline void set_H0(double H0)
    {
        if (iter_number_ > 0){
            std::cout << "warning: setting H0 after the first iteration.\n";
        }
        H0_ = H0;
    }
    inline void set_max_f_rise(double max_f_rise) { max_f_rise_ = max_f_rise; }

    inline void set_use_relative_f(int use_relative_f)
    {
        use_relative_f_ = (bool) use_relative_f;
    }

    // functions for accessing the results
    inline double get_H0() const { return H0_; }

    /**
     * reset the lbfgs optimizer to start a new minimization from x0
     *
     * H0 is not reset because the current value of H0 is probably better than the input value.
     * You can use set_H0() to change H0.
     */
    virtual void reset(pele::Array<double> &x0);

private:

    /**
     * Add a step to the LBFGS Memory
     * This updates s_, y_, rho_, H0_, and k_
     */
    void update_memory(Array<double> x_old,
                       Array<double> & gold,
                       Array<double> x_new,
                       Array<double> & gnew);

    /**
     * Compute the LBFGS step from the memory
     */
    void compute_lbfgs_step(Array<double> step);

    /**
     * Take the step and do a backtracking linesearch if necessary.
     * Apply the maximum step size constraint and ensure that the function
     * does not rise more than the allowed amount.
     */
    double backtracking_linesearch(Array<double>);

    /** Take the step and do a backtracking linesearch if necessary.Take the step and do a backtracking linesearch if necessary.
     * Preconditions the array according to a double thing
     * Apply the maximum step size constraint and ensure that the function
     * does not rise more than the allowed amount.
     */
    void precondition(Array<double> step);
    // Does normal LBFGS without preconditioning
    void no_precondition(Array<double>step);
    double hessnorm;
    double minimum;
    double pf;
    // scale of the final vector
    double scale;
    
    // // line search implementing strong Wolfe conditioins.
    // // As implemented in Nocedal and Wright Chapter 3
    // // This helps us deal with step length issues
    // //returns linestep
    // double wolfe_linesearch(Array<double> step);
    // Interpolate 
    // void zoom(double &a1, double &a2);
    
};






} // end namespace

#endif
