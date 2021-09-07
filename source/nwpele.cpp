#include "pele/nwpele.hpp"






namespace pele {

double NocedalWrightLineSearch::line_search(Array<double> &x, Array<double> step) {

    eig_eq_pele(xoldvec, xold_);
    eig_eq_pele(gradvec, gold_);
    
    eig_eq_pele(step_direction, step);
    // std::cout << step << "\n";
    Scalar f = opt_->get_f();
    stpsize = 1;
    // force a unit step direction

    LSFunc(f, xvec, gradvec, stpsize, step_direction, xoldvec, params);
    
    pele_eq_eig(x, xvec);
    pele_eq_eig(g_, gradvec);
    pele_eq_eig(step, step_direction);
    opt_->set_f(f);
    opt_->set_rms(norm(g_)/sqrt(x.size()));
    // std::cout << stpsize << " ----------------------------------------- final step size \n";
    // std::cout << stpsize*opt_->compute_pot_norm(step) << "\n";
    return stpsize*opt_->compute_pot_norm(step);
};


double NocedalWrightLineSearch::func_grad_wrapper(Vector &x, Vector &grad) {
    pele_eq_eig(xdum, x);
    pele_eq_eig(gdum, grad);
    double f;
    opt_->compute_func_gradient(xdum, f, gdum);
    eig_eq_pele(grad, gdum);
    return f;
}



void NocedalWrightLineSearch::LSFunc(Scalar& fx, Vector& x, Vector& grad,
                                     Scalar& step,
                                     const Vector& drt, const Vector& xp,
                                     const LBFGSParam& param)
{
    // Check the function pointer reference
    if(step <= Scalar(0))
        throw std::invalid_argument("'step' must be positive");
    // change 
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
    // internal steps generating 
    
    
    // STEP 1: Bracketing Phase
    //   Find a range guaranteed to contain a step satisfying strong Wolfe.
    //
    //   See also:
    //     "Numerical Optimization", "Algorithm 3.5 (Line Search Algorithm)".
    int iter = 0;
    for(;;)
        {   

            x.noalias() = xp + step * drt;
            fx = func_grad_wrapper(x, grad);
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
                {   return;}
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


    bool debug=false;
    if (debug==true) {
        // writes line search data to files by iteration number
        Scalar ival;
        Scalar fdebug;
        Vector xdebug(x.size());
        Vector graddebug(x.size());
        std::cout << "line search function is searching along  " << "\n";
        
        std::cout << "linesearch function" << " ------- \n";
        for (int blah=0; blah < 1000; ++blah) {
            ival = blah*0.1;
            xdebug.noalias() = xp + ival*drt;
            fdebug = func_grad_wrapper(xdebug, graddebug);
            std::cout << fdebug << ", ";
        }
        std::cout << "\n---------" << "\n";
        std::cout << "norm of the internal step is " << drt.norm() << "\n";
                
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
            // // use {fx_lo, fx_hi, dg_lo} to make a quadric interpolation of
            // // the function said interpolation is used to estimate the minimum
            // //
            // // polynomial: p (x) = c0*(x - step)² + c1
            // // conditions: p (step_hi) = fx_hi
            // //             p (step_lo) = fx_lo
            // //             p'(step_lo) = dg_lo
            // step  = (fx_hi-fx_lo)*step_lo - (step_hi*step_hi - step_lo*step_lo)*dg_lo/2;
            // step /= (fx_hi-fx_lo)         - (step_hi         - step_lo        )*dg_lo;
            // // if interpolation fails, bisection is used
            // minstepval = std::min(step_lo,step_hi);
            // maxstepval = std::max(step_lo,step_hi);
            // stpdiff = maxstepval-minstepval;
            
            // if( step <= std::min(step_lo,step_hi) ||
                
            //     step >= std::max(step_lo,step_hi) // ||
                
            //     // step-minstepval < 1e-3*stpdiff    ||
                
            //     // maxstepval -step <1e-3*stpdiff 
            //     )
                
            //     {std::cout << "bisection used" << "\n";
            //         step  = step_lo/2 + step_hi/2;}

            step = (step_hi+ step_lo)/2;
            x.noalias() = xp + step * drt;
            fx = func_grad_wrapper(x, grad);
            if(iter++ >= param.max_linesearch)
                {   // Scalar ival;
                    std::cout << grad << "drt \n";
                    std::cout << drt.norm() << "norm \n";
                    std::cout << xp << " starting configuration for error \n";
                    std::cout << drt << "drt \n";
                    // std::cout << "linesearch function" << " ------- \n";
                    // for (int blah; blah < 1000; ++blah) {
                    //     ival = blah*0.1;
                    //     x.noalias() = xp + ival*drt;
                    //     fx = func_grad_wrapper(x, grad);
                    //     std::cout << fx << ", ";
                    // }

                    // std::cout << "------ ---" << "\n";

                    std::cout << opt_->get_tol() << "\n";
                    throw std::logic_error("line search number exceeded ");
                    return;}
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
                        {    
                            return;}

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





}  // pele
