#include "pele/backtracking.h"




namespace pele {

double BacktrackingLineSearch::line_search(Array<double> &x, Array<double> step) {

    eig_eq_pele(xoldvec, xold_);
    eig_eq_pele(gradvec, gold_);
    
    eig_eq_pele(step_direction, step);
    // std::cout << step << "\n";
    Scalar f = opt_->get_f();
    stpsize = 0.01;
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

/**
 * wrapper for the function; should define prolly as an extra in GradientOptimizer
 */
double BacktrackingLineSearch::func_grad_wrapper(Vector &x, Vector &grad) {
    pele_eq_eig(xdum, x);
    pele_eq_eig(gdum, grad);
    double f;
    opt_->compute_func_gradient(xdum, f, gdum);
    eig_eq_pele(grad, gdum);
    return f;
}


/**
 * Line Search function
 */
void BacktrackingLineSearch::LSFunc(Scalar& fx, Vector& x, Vector& grad,
                                    Scalar& step,
                                        const Vector& drt, const Vector& xp,
                                        const LBFGSParam& param) {


        // Decreasing and increasing factors
        const Scalar dec = 0.5;
        const Scalar inc = 2.1;

        // Check the value of step
if(step <= Scalar(0))
            std::invalid_argument("'step' must be positive");

        // Save the function value at the current x
        const Scalar fx_init = fx;
        // Projection of gradient on the search direction
        const Scalar dg_init = grad.dot(drt);
        // Make sure d points to a descent direction
        if(dg_init > 0)
            std::logic_error("the moving direction increases the objective function value");
        const Scalar test_decr = param.ftol * dg_init;
        Scalar width;

        int iter;
        for(iter = 0; iter < param.max_linesearch; iter++)
            {
                // x_{k+1} = x_k + step * d_k
                x.noalias() = xp + step * drt;
                // Evaluate this candidate
                fx = func_grad_wrapper(x, grad);

                if(fx > fx_init + step * test_decr)
                    {
                        width = dec;
                    } else {
                    // Armijo condition is met
                    if(param.linesearch == LINESEARCH_BACKTRACKING_ARMIJO)
                        break;
                    const Scalar dg = grad.dot(drt);
                    if(dg < param.wolfe * dg_init)
                        {
                            width = inc;
                        } else {
                        // Regular Wolfe condition is met
                        if(param.linesearch == LINESEARCH_BACKTRACKING_WOLFE)
                            break;
                        if(dg > -param.wolfe * dg_init)
                            {
                                width = dec;
                            } else {
                            // Strong Wolfe condition is met
                            break;
                        }
                    }
                }

                if(iter >= param.max_linesearch)
                    throw std::runtime_error("the line search routine reached the maximum number of iterations");

                if(step < param.min_step)
                    throw std::runtime_error("the line search step became smaller than the minimum value allowed");

                if(step > param.max_step)
                    throw std::runtime_error("the line search step became larger than the maximum value allowed");

                step *= width;
            }

}





}  // pele