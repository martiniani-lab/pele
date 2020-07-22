#include "pele/lbfgs.h"
#include <memory>
#include <iostream>
#include <limits>

namespace pele {

LBFGS::LBFGS( std::shared_ptr<pele::BasePotential> potential, const pele::Array<double> x0,
              double tol, int M, int T)
    : GradientOptimizer(potential, x0, tol),
      M_(M),
      max_f_rise_(1e-4),
      use_relative_f_(false),
      rho_(M_),
      H0_(0.1),
      k_(0),
      alpha(M_),
      xold(x_.size()),
      gold(x_.size()),
      step(x_.size()),
      exact_g_(x_.size()),
      exact_gold(x_.size()),
      T_(T),
      line_search_method(this, 0.1)
{
    // set precision of printing
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);

    inv_sqrt_size = 1 / sqrt(x_.size());

    // allocate space for s_ and y_
    s_ = Array<double>(x_.size() * M_);
    y_ = Array<double>(x_.size() * M_);
    std::cout << T << " set T \n";
    std::cout << x0 << "-------- x0" << "\n";
}

/**
 * Do one iteration iteration of the optimization algorithm
 */
void LBFGS::one_iteration() {
    if (!func_initialized_) {
        initialize_func_gradient();
    }
    // make a copy of the position and gradient
    xold.assign(x_);
    gold.assign(g_);
    compute_lbfgs_step(step);
    // Line search method
    line_search_method.set_xold_gold_(xold, gold);
    line_search_method.set_g_f_ptr(g_);
    double stepnorm = line_search_method.line_search(x_, step);
    // double stepnorm = 1;
    // Line search method 2

    // double stepnorm = backtracking_linesearch(step);

    // step taken 

    
    update_memory(xold, gold, x_, g_);
    
    if (false){
        std::cout << "lbfgs: " << iter_number_
                  << " E " << f_
                  << " rms " << rms_
                  << " nfev " << nfev_
                  << " step norm " << stepnorm << std::endl;
    }
    iter_number_ += 1;
}

void LBFGS::update_memory(Array<double> x_old,
                          Array<double> & g_old,
                          Array<double> x_new,
                          Array<double> & g_new)
{
    // update the lbfgs memory
    // This updates s_, y_, rho_, and H0_, and k_
    int klocal = k_ % M_;
    double ys = 0;
    double yy = 0;
    
    // define a dummy accumulator to help with exactly
    // calculating y
    // xsum_small_accumulator sacc_dummy;
    // std::cout << "gradient difference" << "\n";
#pragma simd reduction( + : ys, yy)
    for (size_t j2 = 0; j2 < x_.size(); ++j2){
        size_t ind_j2 = klocal * x_.size() + j2;
        y_[ind_j2] = g_new[j2] - g_old[j2];
        s_[ind_j2] = x_new[j2] - x_old[j2];
        ys += y_[ind_j2] * s_[ind_j2];
        yy += y_[ind_j2] * y_[ind_j2];
    }
    
    // std::cout << "end gradient difference" << "\n";
    if (ys == 0.) {
        if (verbosity_ > 0) {
            std::cout << "warning: resetting YS to 1." << std::endl;
        }
        ys = 1.;
    }
    rho_[klocal] = 1. / ys;
    if (yy == 0.) {
        if (verbosity_ > 0) {
            std::cout << "warning: resetting YY to 1." << std::endl;
        }
        yy = 1.;
    }
    std::cout << ys << "ys  \n";
    std::cout << yy << "yy  \n";
    // std::cout << dot(g_new-g_old, x_new-x_old) << "\n";

    H0_ = ys / yy;
    // increment k
    k_ += 1;
}



void LBFGS::compute_lbfgs_step(Array<double> step)
{
//     if (k_ == 0){
//         // take a conservative first step
//         double gnorm = norm(g_);
//         if (gnorm > 1.) {
//             gnorm = 1. / gnorm;
//         }
//         // 0.05 is a conservative factor that does not mess up anything
//         double prefactor =  -gnorm * (maxstep_);
// #pragma simd
//         for (size_t j2 = 0; j2 < x_.size(); ++j2){
//             step[j2] = prefactor * g_[j2];
//         }
//         return;
//     }

    // copy the gradient into step
    step.assign(g_);

    int jmin = std::max(0, k_ - M_);
    int jmax = k_;
    int i;

    alpha.assign(0.0);
    // loop backwards through the memory
    for (int j = jmax - 1; j >= jmin; --j) {
        i = j % M_;
        double alpha_tmp = 0;
#pragma simd reduction(+ : alpha_tmp)
        for (size_t j2 = 0; j2 < step.size(); ++j2){
            alpha_tmp += rho_[i] * s_[i * step.size() + j2] * step[j2];
        }
#pragma simd
        for (size_t j2 = 0; j2 < step.size(); ++j2){
            step[j2] -= alpha_tmp * y_[i * step.size() + j2];
        }
        alpha[i] = alpha_tmp;
    }
    
    no_precondition(step);
    std::cout << dot(step, g_) << " step val before the iterations \n";
    // loop forwards through the memory
    for (int j = jmin; j < jmax; ++j) {
            i = j % M_;
        double beta = 0;
#pragma simd reduction(+ : beta)
        for (size_t j2 = 0; j2 < step.size(); ++j2) {
            beta -= rho_[i] * y_[i * step.size() + j2] * step[j2];  // -= due to inverted step
        }
        double alpha_beta = alpha[i] - beta;
#pragma simd
        for (size_t j2 = 0; j2 < step.size(); ++j2) {
            step[j2] -= s_[i * step.size() + j2] * alpha_beta;  // -= due to inverted step
        }
    }
    std::cout << dot(step, gold) << " step val after the iterations \n";
    


}
    



void LBFGS::precondition(Array<double> step) {
    // Solver updates
    // Preconditioning steps
    Eigen::VectorXd r(step.size());
    r.setZero();
    for (int i =0; i < step.size(); ++i) {
        r[i] = step[i];
    }
    Eigen::VectorXd q(step.size());
    q.setZero();
    Eigen::VectorXd gg(step.size());
    eig_eq_pele(gg, g_);
    if ((iter_number_)%T_ ==0) {
        std::cout << (r.dot(gg)) << "r dot grad val \n";
        q = update_solver(r);
        std::cout << "this is inside" << "\n";
        std::cout << (q.dot(gg)) << "q dot grad val \n";
    }
    else {
        q = saved_hessian.colPivHouseholderQr().solve(r);
    }
    // q = solver->solve(r);
    for (int i =0; i < step.size(); ++i) {
        step[i] = -q[i];
    }

}


Eigen::VectorXd LBFGS::update_solver(Eigen::VectorXd r) {
    // Eigen::ColPivHouseholderQR<Eigen::MatrixXd> solve;
    saved_hessian = get_hessian_sparse_pos();
    std::cout << saved_hessian.eigenvalues() << " eigenvalues \n";
    std::cout << saved_hessian.inverse().eigenvalues() << "\n";
    return saved_hessian.fullPivHouseholderQr().solve(r);
    // std::cout << hess_sparse << "\n";
}

Eigen::MatrixXd LBFGS::get_hessian_sparse() {
    Array<double> hess(xold.size()*xold.size());
    Array<double> grad(xold.size());
    double e = potential_->get_energy_gradient_hessian(x_, grad, hess);
    Eigen::MatrixXd hess_dense(xold.size(), xold.size());
    // Note to future: Autodiff can be annoying with memory leaks
    hess_dense.setZero();
    for (int i = 0; i < xold.size(); ++i) {
        for (int j=0; j < xold.size(); ++j) {
            hess_dense(i, j) = hess[i + grad.size()*j];
        }
    }
    // std::cout << hess_dense << "\n dense hessian \n";
    return hess_dense;
}


Eigen::MatrixXd LBFGS::get_hessian_sparse_pos() {
    Eigen::MatrixXd hess = get_hessian_sparse();
    Eigen::VectorXd eigvals = hess.eigenvalues().real();
    double minimum = eigvals.minCoeff();
    double maximum = eigvals.maxCoeff();
    std::cout << minimum << " minimum \n";
    std::cout << maximum << " maximum \n";
    std::cout << minimum/maximum << " minimum/maximum\n";




    if (minimum<0 and minimum/maximum<-1e-2) {
        // hardcoded but can set globally if necessary
        // controls fraction of the PSD factor added to hessian
        double pf = 3;
        // note minimum is negative
        return hess - pf*minimum*Eigen::MatrixXd::Identity(x_.size(), x_.size());
    }
    else if (minimum/maximum < 1e-2) {
        double extra = eigvals.mean();
        double pf = 3;
        std::cout << "we are here" << "\n";
        return hess + pf*extra*Eigen::MatrixXd::Identity(x_.size(), x_.size());
    }
    else {return hess;}

    
}





void LBFGS::no_precondition(Array<double> step) {
    // scale the step size by H0, invert the step to point downhill
#pragma simd
    for (size_t j2 = 0; j2 < step.size(); ++j2){
        // std::cout << step[j2] * -H0_ << "\n";
        step[j2] *= - H0_;
    }
    std::cout << H0_ << "H0_ H0 \n";

}






double LBFGS::backtracking_linesearch(Array<double> step)
{
    double fnew;
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
        std::cout << "warning: step direction is wrong" << "\n";
    }
    
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
        compute_func_gradient(x_, fnew, exact_g_);

        for (int i = 0; i < exact_g_.size(); ++i) {
            g_[i] = xsum_small_round(&(exact_g_[i]));
        }
        
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
                    << " H0 " << H0_ << std::endl;
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





// double LBFGS::zoom(double size1, size2, Array<double> step, double phialphaj,
//                    double phialpha0, double gradphialpha0) {
//     for (int nred; i < nred_max; ++i) {
//         // bisecting search can be replaced with 
//         // better interpolation if necessary
//         // interpolation begin
//         double sizej = (size1 + size2)/2;
//         // TODO: set an interpolation safeguard;
//         double phialphaj = potential_->get_energy(xold + sizej*step);
//         if (phialphaj > phialpha0 + c1*phialpha0) {
            
//         }
//     }
// }
// double LBFGS::wolfe_linesearch(Array<double> step) {
//     Array<double> gradphialphi;      // 
//     Array<double> gradphialpha0;   // bounded step
//     double phialpha0;
//     double phialphai;
//     double nredmax = 10;
//     //  compute the gradient at the original point
//     compute_func_gradient(xold,
//                           phialpha0,
//                           gradphialpha0);
//     double alphai = maxstep_;
//     double phialphaold = phialpha0;
//     Array<double> gradphialphaold = gradphialpha0;
//     double phipalpha0 = dot(step, gradphialpha0);
//     // the result we want
//     double alphastar;
//     for (int nred=1; i < nred_max; ++i) {
//         compute_func_gradient(xold + alphai*step, phialphai, gradphialphi);
//         if ((phialphai>phialpha0 + c1*alphai*phipalpha0) or
//             ((phialphai >= 0 ) and i > 1)) {
//             alphastar = zoom(alphaold, alphai, step, phialpha0, phipalpha0);
//             break;
//         }

//         if(norm(gradphialphi) <= c2*norm(phialpha0)) {
//             alphastar = alphai;
//         }

//         if (dot(gradphialphi, step)>=0) {
//             alphastar = zoom(alphai, maxstep_, step, phialpha0, phipalpha0);
//         }
//     }
// }









void LBFGS::reset(pele::Array<double> &x0) {
    if (x0.size() != x_.size()){
        throw std::invalid_argument("The number of degrees of freedom (x0.size()) cannot change when calling reset()");
        }
        k_ = 0;
        iter_number_ = 0;
        nfev_ = 0;
        x_.assign(x0);
        initialize_func_gradient(); }

}


