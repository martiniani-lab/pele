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
      T_(T)
{
    // set precision of printing
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);

    inv_sqrt_size = 1 / sqrt(x_.size());

    // allocate space for s_ and y_
    s_ = Array<double>(x_.size() * M_);
    y_ = Array<double>(x_.size() * M_);
    std::cout << T << " set T \n";
}

/**
 * Do one iteration iteration of the optimization algorithm
 */
    void LBFGS::one_iteration()
    {   
        if (!func_initialized_) {
            initialize_func_gradient();
        }

        // make a copy of the position and gradient
        xold.assign(x_);
        gold.assign(g_);

        for (int i = 0; i < xold.size(); ++i) {
            xsum_small_equal(&(exact_gold[i]), &(exact_g_[i]));
        }
        // get the stepsize and direction from the LBFGS algorithm
        compute_lbfgs_step(step);
        // std::cout << iter_number_ << "\n";
        // reduce the stepsize if necessary
        double stepnorm = backtracking_linesearch(step);

        // update the LBFGS memeory
        update_memory(xold, exact_gold, x_, exact_g_);

        // print some status information
        if ((iprint_ > 0) && (iter_number_ % iprint_ == 0)){
            std::cout << "lbfgs: " << iter_number_
                      << " E " << f_
                      << " rms " << rms_
                      << " nfev " << nfev_
                      << " step norm " << stepnorm << std::endl;
        }
        iter_number_ += 1;
    }

void LBFGS::update_memory(
                          Array<double> x_old,
                              std::vector<xsum_small_accumulator> & exact_gold,
                              Array<double> x_new,
                          std::vector<xsum_small_accumulator> & exact_gnew)
{
    // update the lbfgs memory
    // This updates s_, y_, rho_, and H0_, and k_
    int klocal = k_ % M_;
    double ys = 0;
    double yy = 0;
        
    // define a dummy accumulator to help with exactly
    // calculating y
    xsum_small_accumulator sacc_dummy;
        
    // std::cout << "gradient difference" << "\n";
#pragma simd reduction( + : ys, yy)
    for (size_t j2 = 0; j2 < x_.size(); ++j2){
        size_t ind_j2 = klocal * x_.size() + j2;
        xsum_small_subtract_acc_and_set(&(exact_gnew[j2]),
                                        &(exact_gold[j2]),
                                        &sacc_dummy);
        y_[ind_j2] = xsum_small_round(&sacc_dummy);
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
    H0_ = ys / yy;

    // increment k
    k_ += 1;
}





void LBFGS::compute_lbfgs_step(Array<double> step)
{
    if (k_ == 0){
        // take a conservative first step
        double gnorm = norm(g_);
        if (gnorm > 1.) {
            gnorm = 1. / gnorm;
        }
        // 0.05 is a conservative factor that does not mess up anything
        double prefactor =  -gnorm * (0.03);
#pragma simd
        for (size_t j2 = 0; j2 < x_.size(); ++j2){
            step[j2] = prefactor * g_[j2];
        }

        return;
    }

    // copy the gradient into step
    step.assign(g_);

    int jmin = std::max(0, k_ - M_);
    int jmax = k_;
    int i;

    alpha.assign(0.0);
//     // loop backwards through the memory
//     for (int j = jmax - 1; j >= jmin; --j) {
//         i = j % M_;
//         double alpha_tmp = 0;
// #pragma simd reduction(+ : alpha_tmp)
//         for (size_t j2 = 0; j2 < step.size(); ++j2){
//             alpha_tmp += rho_[i] * s_[i * step.size() + j2] * step[j2];
//         }
// #pragma simd
//         for (size_t j2 = 0; j2 < step.size(); ++j2){
//             step[j2] -= alpha_tmp * y_[i * step.size() + j2];
//         }
//         alpha[i] = alpha_tmp;
//     }

    // Preconditioning step
    // we're using iter_number -1 to get a better initialization


    // Eigen::VectorXd r(step.size());

    // for (int i =0; i < step.size(); ++i) {
    //     r[i] = step[i];
    // }

    // Eigen::VectorXd q;

    // q = solver->solve(r);
    // // negative sign for descent direction
    // for (int i =0; i < step.size(); ++i) {
    //     step[i] = -q[i];
    // }
    precondition(step);
    
    
    // #pragma simd
    //     for (size_t j2 = 0; j2 < step.size(); ++j2){
    //         step[j2] *= -H0_;
    //     }

    
    
    // loop forwards through the memory
//     for (int j = jmin; j < jmax; ++j) {
//         i = j % M_;
//         double beta = 0;
// #pragma simd reduction(+ : beta)
//         for (size_t j2 = 0; j2 < step.size(); ++j2) {
//             beta -= rho_[i] * y_[i * step.size() + j2] * step[j2];  // -= due to inverted step
//         }
//         double alpha_beta = alpha[i] - beta;
// #pragma simd
//         for (size_t j2 = 0; j2 < step.size(); ++j2) {
//             step[j2] -= s_[i * step.size() + j2] * alpha_beta;  // -= due to inverted step
//         }
//     }
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
    if ((iter_number_-1)%T_ ==0) {
        q = update_solver(r);
    }
    // q = solver->solve(r);
    for (int i =0; i < step.size(); ++i) {
        step[i] = q[i];
    }
}

Eigen::VectorXd LBFGS::update_solver(Eigen::VectorXd r) {
    // Eigen::ColPivHouseholderQR<Eigen::MatrixXd> solve;
    Eigen::MatrixXd hess_sparse;
    hess_sparse.setZero();
    hess_sparse = get_hessian_sparse();
    return hess_sparse.colPivHouseholderQr().solve(r);
    // std::cout << hess_sparse << "\n";
}

Eigen::MatrixXd LBFGS::get_hessian_sparse() {
    Array<double> hess(xold.size()*xold.size());
    Array<double> grad(xold.size());
    double e = potential_->get_energy_gradient_hessian(x_, grad, hess);
    Eigen::MatrixXd hess_dense(xold.size(), xold.size());
    // Note to future: Eigen can be annoying with memory leaks
    hess_dense.setZero();
    for (int i = 0; i < xold.size(); ++i) {
        for (int j=0; j < xold.size(); ++j) {
            hess_dense(i, j) = hess[i + grad.size()*j];
        }
    }
    // std::cout << hess_dense << "\n dense hessian \n";
    return hess_dense;
}


void LBFGS::no_precondition(Array<double> step) {
    // scale the step size by H0, invert the step to point downhill
#pragma simd
    for (size_t j2 = 0; j2 < step.size(); ++j2){
        // std::cout << step[j2] * -H0_ << "\n";
        step[j2] *= -H0_;
    }
}






double LBFGS::backtracking_linesearch(Array<double> step)
{
    double fnew;

    // if the step is pointing uphill, invert it
        if (dot(step, g_) > 0.){
            if (verbosity_ > 1) {
                std::cout << "warning: step direction was uphill.  inverting" << std::endl;
            }
#pragma simd
        for (size_t j2 = 0; j2 < step.size(); ++j2){
            step[j2] *= -1;
        }
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

void LBFGS::reset(pele::Array<double> &x0)
{
    if (x0.size() != x_.size()){
        throw std::invalid_argument("The number of degrees of freedom (x0.size()) cannot change when calling reset()");
    }
    k_ = 0;
    iter_number_ = 0;
    nfev_ = 0;
    x_.assign(x0);
    initialize_func_gradient();
}
}
