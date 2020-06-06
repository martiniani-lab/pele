#ifndef _PELE_LBFGSMPFR_H__
#define _PELE_LBFGSMPFR_H__

#include <vector>
#include <memory>
#include "base_potential.h"
#include "array.h"
#include "optimizermpfr.h"
#include "/usr/include/mpreal.h"


extern "C" {
#include "xsum.h"
}

using mpfr::mpreal;
namespace pele{

    /**
 * An implementation of the LBFGS optimization algorithm in c++.  This
 * Implementation uses a backtracking linesearch.
 */
class LBFGSMPFR : public GradientOptimizerMPFR{
private:

    int M_; /**< The length of the LBFGS memory */
    mpreal max_f_rise_; /**< The maximum the function is allowed to rise in a
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
    Array<mpreal> s_;
    /** y_ stores the changes in gradient for the previous M steps */
    Array<mpreal> y_;
    /** rho stores 1/dot(y_, s_) for the previous M steps */
    Array<mpreal> rho_;
    /**
     * H0 is the initial estimate for the diagonal component of the inverse Hessian.
     * It is an input parameter, but the estimate is improved during the run.
     * H0 is a scalar, which means that we use the same value for all degrees of freedom.
     */
    mpreal H0_;
    int k_; /**< Counter for how many times the memory has been updated */

    Array<mpreal> alpha; //!< Alpha used when looping through LBFGS memory


    Array<mpreal> xold; //!< Coordinates before taking a step
    Array<mpreal> gold; //!< Gradient before taking a step
    std::vector<xsum_small_accumulator> exact_gold;
    std::vector<xsum_small_accumulator> exact_g_;
    Array<mpreal> step; //!< Step size and direction
    mpreal inv_sqrt_size; //!< The inverse square root the the number of components

public:
    /**
     * Constructor
     */
    LBFGSMPFR( std::shared_ptr<pele::BasePotential> potential, const pele::Array<double> x0,
               double tol = 1e-4, int M = 4, int precison=53);

    /**
     * Destructor
     */
    virtual ~LBFGSMPFR() {}

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
    inline double get_H0() const { return H0_.toDouble(); }

    /**
     * reset the lbfgs optimizer to start a new minimization from x0
     *
     * H0 is not reset because the current value of H0 is probably better than the input value.
     * You can use set_H0() to change H0.
     */
    void reset(pele::Array<double> &x0);

private:

    /**
     * Add a step to the LBFGS Memory
     * This updates s_, y_, rho_, H0_, and k_
     */
    void update_memory(
                       Array<mpreal> x_old,
                       std::vector<xsum_small_accumulator> & exact_gold,
                       Array<mpreal> x_new,
                       std::vector<xsum_small_accumulator> & exact_gnew);

    /**
     * Compute the LBFGS step from the memory
     */
    void compute_lbfgs_step(Array<mpreal> step);

    /**
     * Take the step and do a backtracking linesearch if necessary.
     * Apply the maximum step size constraint and ensure that the function
     * does not rise more than the allowed amount.
     */
    mpreal backtracking_linesearch(Array<mpreal> step);

};
}

#endif
