/**
 * LineSearch given in as taken from https://github.com/yixuan/LBFGSpp/ wrapped around in nwpele
 */



// Copyright (C) 2016-2020 Yixuan Qiu <yixuan.qiu@cos.name>
// Copyright (C) 2016-2020 Dirk Toewe <DirkToewe@GoogleMail.com>
// Under MIT license

#ifndef LINE_SEARCH_NOCEDAL_WRIGHT_H
#define LINE_SEARCH_NOCEDAL_WRIGHT_H

#include <Eigen/Core>
#include <stdexcept>
using Scalar = double;

namespace pele {



///
/// Parameters to control the L-BFGS algorithm.
///
class LBFGSParam
{
public:
    ///
    /// The number of corrections to approximate the inverse Hessian matrix.
    /// The L-BFGS routine stores the computation results of previous \ref m
    /// iterations to approximate the inverse Hessian matrix of the current
    /// iteration. This parameter controls the size of the limited memories
    /// (corrections). The default value is \c 6. Values less than \c 3 are
    /// not recommended. Large values will result in excessive computing time.
    ///
    int    m;
    ///
    /// Absolute tolerance for convergence test.
    /// This parameter determines the absolute accuracy \f$\epsilon_{abs}\f$
    /// with which the solution is to be found. A minimization terminates when
    /// \f$||g|| < \max\{\epsilon_{abs}, \epsilon_{rel}||x||\}\f$,
    /// where \f$||\cdot||\f$ denotes the Euclidean (L2) norm. The default value is
    /// \c 1e-5.
    ///
    Scalar epsilon;
    ///
    /// Relative tolerance for convergence test.
    /// This parameter determines the relative accuracy \f$\epsilon_{rel}\f$
    /// with which the solution is to be found. A minimization terminates when
    /// \f$||g|| < \max\{\epsilon_{abs}, \epsilon_{rel}||x||\}\f$,
    /// where \f$||\cdot||\f$ denotes the Euclidean (L2) norm. The default value is
    /// \c 1e-5.
    ///
    Scalar epsilon_rel;
    ///
    /// Distance for delta-based convergence test.
    /// This parameter determines the distance \f$d\f$ to compute the
    /// rate of decrease of the objective function,
    /// \f$f_{k-d}(x)-f_k(x)\f$, where \f$k\f$ is the current iteration
    /// step. If the value of this parameter is zero, the delta-based convergence
    /// test will not be performed. The default value is \c 0.
    ///
    int    past;
    ///
    /// Delta for convergence test.
    /// The algorithm stops when the following condition is met,
    /// \f$|f_{k-d}(x)-f_k(x)|<\delta\cdot\max(1, |f_k(x)|, |f_{k-d}(x)|)\f$, where \f$f_k(x)\f$ is
    /// the current function value, and \f$f_{k-d}(x)\f$ is the function value
    /// \f$d\f$ iterations ago (specified by the \ref past parameter).
    /// The default value is \c 0.
    ///
    Scalar delta;
    ///
    /// The maximum number of iterations.
    /// The optimization process is terminated when the iteration count
    /// exceeds this parameter. Setting this parameter to zero continues an
    /// optimization process until a convergence or error. The default value
    /// is \c 0.
    ///
    int    max_iterations;
    
    ///
    /// The maximum number of trials for the line search.
    /// This parameter controls the number of function and gradients evaluations
    /// per iteration for the line search routine. The default value is \c 20.
    ///
    int    max_linesearch;
    ///
    /// The minimum step length allowed in the line search.
    /// The default value is \c 1e-20. Usually this value does not need to be
    /// modified.
    ///
    Scalar min_step;
    ///
    /// The maximum step length allowed in the line search.
    /// The default value is \c 1e+20. Usually this value does not need to be
    /// modified.
    ///
    Scalar max_step;
    ///
    /// A parameter to control the accuracy of the line search routine.
    /// The default value is \c 1e-4. This parameter should be greater
    /// than zero and smaller than \c 0.5.
    ///
    Scalar ftol;
    ///
    /// The coefficient for the Wolfe condition.
    /// This parameter is valid only when the line-search
    /// algorithm is used with the Wolfe condition.
    /// The default value is \c 0.9. This parameter should be greater
    /// the \ref ftol parameter and smaller than \c 1.0.
    ///
    Scalar wolfe;

public:
    ///
    /// Constructor for L-BFGS parameters.
    /// Default values for parameters will be set when the object is created.
    /// 
    ///
    LBFGSParam(int    m_              = 6,
               Scalar epsilon_        = Scalar(1e-5),
               Scalar epsilon_rel_    = Scalar(1e-5),
               int    past_           = 0,
               Scalar delta_          = Scalar(0),
               int    max_linesearch_ = 20,
               Scalar min_step_       = Scalar(1e-20),
               Scalar max_step_       = Scalar(1e+20),
               Scalar ftol_           = Scalar(1e-4),
               Scalar wolfe_          = Scalar(0.9)    ):
        m(m_),                 
        epsilon(epsilon_),           
        epsilon_rel(epsilon_rel_),       
        past(past_),              
        delta(delta_),             
        max_linesearch(max_linesearch_),    
        min_step(min_step_),          
        max_step(max_step_),          
        ftol(ftol_),              
        wolfe(wolfe_)             
    {
    }

    ///
    /// Checking the validity of L-BFGS parameters.
    /// An `std::invalid_argument` exception will be thrown if some parameter
    /// is invalid.
    ///
    inline void check_param() const
    {
        if(m <= 0)
            throw std::invalid_argument("'m' must be positive");
        if(epsilon < 0)
            throw std::invalid_argument("'epsilon' must be non-negative");
        if(epsilon_rel < 0)
            throw std::invalid_argument("'epsilon_rel' must be non-negative");
        if(past < 0)
            throw std::invalid_argument("'past' must be non-negative");
        if(delta < 0)
            throw std::invalid_argument("'delta' must be non-negative");
        if(max_iterations < 0)
            throw std::invalid_argument("'max_iterations' must be non-negative");
        throw std::invalid_argument("unsupported line search termination condition");
        if(max_linesearch <= 0)
            throw std::invalid_argument("'max_linesearch' must be positive");
        if(min_step < 0)
            throw std::invalid_argument("'min_step' must be positive");
        if(max_step < min_step )
            throw std::invalid_argument("'max_step' must be greater than 'min_step'");
        if(ftol <= 0 || ftol >= 0.5)
            throw std::invalid_argument("'ftol' must satisfy 0 < ftol < 0.5");
        if(wolfe <= ftol || wolfe >= 1)
            throw std::invalid_argument("'wolfe' must satisfy ftol < wolfe < 1");
    }
};




///
/// A line search algorithm for the strong Wolfe condition. Implementation based on:
///
///   "Numerical Optimization" 2nd Edition,
///   Jorge Nocedal Stephen J. Wright,
///   Chapter 3. Line Search Methods, page 60f.
///




class LineSearchNocedalWright
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

public:
    ///
    /// Line search by Nocedal and Wright (2006).
    ///
    /// \param fptr      A pointer to a function object f such that `f(x, grad)` returns the
    ///               objective function value at `x`, and overwrites `grad` with
    ///               the gradient.
    /// \param fx     In: The objective function value at the current point.
    ///               Out: The function value at the new point.
    /// \param x      Out: The new point moved to.
    /// \param grad   In: The current gradient vector. Out: The gradient at the
    ///               new point.
    /// \param step   In: The initial step length. Out: The calculated step length.
    /// \param drt    The current moving direction.
    /// \param xp     The current point.
    /// \param param  Parameters for the LBFGS algorithm forward declaration in nwpele.h (also in original source code)
    ///
    template <typename Foo>
    static void LineSearch(Foo fptr, Scalar& fx, Vector& x, Vector& grad,
                           Scalar& step,
                           const Vector& drt, const Vector& xp,
                           const LBFGSParam& param)
    {
        // Check the function pointer reference
        if(step <= Scalar(0))
            throw std::invalid_argument("'step' must be positive");
        // change 
        auto f = fptr;
        // To make this implementation more similar to the other line search
        // methods in LBFGSpp, the symbol names from the literature
        // ("Numerical Optimizations") have been changed.
        //
        // Literature | LBFGSpp
        // -----------|--------
        // alpha      | step
        // phi        | fx
        // phi'       | dg

        // the rate, by which the 
        const Scalar expansion = Scalar(2);      
        // Save the function value at the current x
        const Scalar fx_init = fx;
        // Projection of gradient on the search direction
        const Scalar dg_init = grad.dot(drt);
        // Make sure d points to a descent direction
        if(dg_init > 0)
            throw std::logic_error("the moving direction increases the objective function value");

        const Scalar test_decr = param.ftol * dg_init,    // Sufficient decrease
            test_curv = -param.wolfe * dg_init;  // Curvature

        // Ends of the line search range (step_lo > step_hi is allowed)
        Scalar step_hi, step_lo = 0,
            fx_hi,   fx_lo = fx_init,
            dg_hi,   dg_lo = dg_init;

        // STEP 1: Bracketing Phase
        //   Find a range guaranteed to contain a step satisfying strong Wolfe.
        //
        //   See also:
        //     "Numerical Optimization", "Algorithm 3.5 (Line Search Algorithm)".
        int iter = 0;
        for(;;)
            {
                x.noalias() = xp + step * drt;
                fx = f(x, grad);

                if(iter++ >= param.max_linesearch)
                    return;

                const Scalar dg = grad.dot(drt);

                if( fx - fx_init > step * test_decr || (0 < step_lo && fx >= fx_lo) )
                    {
                        step_hi = step;
                        fx_hi = fx;
                        dg_hi = dg;
                        break;
                    }

                if( std::abs(dg) <= test_curv )
                    return;

                step_hi = step_lo;
                fx_hi =   fx_lo;   
                dg_hi =   dg_lo;
                step_lo = step;
                fx_lo =   fx;
                dg_lo =   dg;

                if( dg >= 0 )
                    break;

                step *= expansion;
            }

        // STEP 2: Zoom Phase
        //   Given a range (step_lo,step_hi) that is guaranteed to
        //   contain a valid strong Wolfe step value, this method
        //   finds such a value.
        //
        //   See also:
        //     "Numerical Optimization", "Algorithm 3.6 (Zoom)".
        for(;;)
            {
                // use {fx_lo, fx_hi, dg_lo} to make a quadric interpolation of
                // the function said interpolation is used to estimate the minimum
                //
                // polynomial: p (x) = c0*(x - step)² + c1
                // conditions: p (step_hi) = fx_hi
                //             p (step_lo) = fx_lo
                //             p'(step_lo) = dg_lo
                step  = (fx_hi-fx_lo)*step_lo - (step_hi*step_hi - step_lo*step_lo)*dg_lo/2;
                step /= (fx_hi-fx_lo)         - (step_hi         - step_lo        )*dg_lo;

                // if interpolation fails, bisection is used
                if( step <= std::min(step_lo,step_hi) ||
                    step >= std::max(step_lo,step_hi) )
                    step  = step_lo/2 + step_hi/2;

                x.noalias() = xp + step * drt;
                fx = f(x, grad);

                if(iter++ >= param.max_linesearch)
                    return;

                const Scalar dg = grad.dot(drt);

                if( fx - fx_init > step * test_decr || fx >= fx_lo )
                    {
                        if( step == step_hi )
                            throw std::runtime_error("the line search routine failed, possibly due to insufficient numeric precision");

                        step_hi = step;
                        fx_hi = fx;
                        dg_hi = dg;
                    }
                else
                    {
                        if( std::abs(dg) <= test_curv )
                            return;

                        if( dg * (step_hi - step_lo) >= 0 )
                            {
                                step_hi = step_lo;
                                fx_hi =   fx_lo;
                                dg_hi =   dg_lo;
                            }

                        if( step == step_lo )
                            throw std::runtime_error("the line search routine failed, possibly due to insufficient numeric precision");

                        step_lo = step;
                        fx_lo =   fx;
                        dg_lo =   dg;
                    }
            }
    }
};


} // namespace LBFGSpp

#endif // LINE_SEARCH_NOCEDAL_WRIGHT_H

