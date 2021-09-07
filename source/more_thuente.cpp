#include "pele/more_thuente.hpp"



namespace pele
{
int MoreThuente::cvsrch(vector_t &x, scalar_t f,
                        vector_t &g, scalar_t &stp, vector_t &s) {
    // we rewrite this from MIN-LAPACK and some MATLAB code
    int nfev = 0;
    // internal values
    int info = 0;
    int infoc = 1;
    scalar_t stpmax = 1e15;
    scalar_t stpmin = 0.0;
    desdir = false;
    // std::cout << dot(g, s) << " dot(g, s) \n";
    // if (dot(g, s) >= 0) {
    //     // we''re here?
    //     for (int i = 0; i<s.size(); i++) {
    //         s[i] = -s[i];
    //     }
    //     desdir = true;
    // }
    // std::cout << dot(g, s) << " dot(g, s) \n";
    scalar_t dginit = dot(g, s);
    std::cout << dginit << "\n";

    if (dginit >= 0.0) {
        throw std::logic_error("the moving direction increases the objective function value");
    }

    bool brackt = false;
    bool stage1 = true;
    scalar_t finit = f;
    scalar_t dgtest = ftol * dginit;
    scalar_t width = stpmax - stpmin;
    scalar_t width1 = 2 * width;
    vector_t wa = x;
    scalar_t stx = 0.0;
    scalar_t fx = finit;
    scalar_t dgx = dginit;
    scalar_t sty = 0.0;
    scalar_t fy = finit;
    scalar_t dgy = dginit;
    scalar_t stmin;
    scalar_t stmax;
    
    while (true) {
        // make sure we stay in the interval when setting min/max-step-width
        if (brackt) {
            stmin = std::min<scalar_t>(stx, sty);
            stmax = std::max<scalar_t>(stx, sty);
        } else {
            stmin = stx;
            stmax = stp + xtrapf * (stp - stx);
        }

        // Force the step to be within the bounds stpmax and stpmin.
        stp = std::max<scalar_t>(stp, stpmin);
        stp = std::min<scalar_t>(stp, stpmax);
        // Oops, let us return the last reliable values
        // DEBUG
        // std::cout << (brackt && ((stp <= stmin) || (stp >= stmax))) << " case 1 \n";
        // std::cout << (nfev >= maxfev-1) << " case 2\n";
        // std::cout << nfev << "---- nfeev inside\n";
        // std::cout << maxfev-1 << " --- maxfeev inside\n";
        // std::cout << infoc << " infoc value \n";
        // std::cout << (brackt && ((stmax - stmin) <= (xtol * stmax))) << " case 3 \n";

        if ((brackt && ((stp <= stmin) || (stp >= stmax))) ||
            (nfev >= maxfev - 1) || (infoc == 0) ||
            (brackt && ((stmax - stmin) <= (xtol * stmax)))) {
            stp = stx;
        }

        // test new point

        for (size_t i=0; i < x.size(); ++i) {
            x[i] = wa[i] + stp * s[i];
        }


        // f = function(x);
        // function.Gradient(x, &g);
        opt_->compute_func_gradient(x, f, g);
        nfev++;
        scalar_t dg = dot(g, s);
        scalar_t ftest1 = finit + stp * dgtest;
        // all possible convergence tests
        if ((brackt & ((stp <= stmin) | (stp >= stmax))) | (infoc == 0)) info = 6;

        if ((stp == stpmax) & (f <= ftest1) & (dg <= dgtest)) info = 5;

        if ((stp == stpmin) & ((f > ftest1) | (dg >= dgtest))) info = 4;

        if (nfev >= maxfev) info = 3;

        if (brackt & (stmax - stmin <= xtol * stmax)) info = 2;

        if ((f <= ftest1) & (fabs(dg) <= gtol * (-dginit))) info = 1;

        // terminate when convergence reached
        if (info != 0) return -1;

        if (stage1 & (f <= ftest1) &
            (dg >= std::min<scalar_t>(ftol, gtol) * dginit))
            stage1 = false;

        if (stage1 & (f <= fx) & (f > ftest1)) {
            scalar_t fm = f - stp * dgtest;
            scalar_t fxm = fx - stx * dgtest;
            scalar_t fym = fy - sty * dgtest;
            scalar_t dgm = dg - dgtest;
            scalar_t dgxm = dgx - dgtest;
            scalar_t dgym = dgy - dgtest;

            cstep(stx, fxm, dgxm, sty, fym, dgym, stp, fm, dgm, brackt, stmin,
                  stmax, infoc);

            fx = fxm + stx * dgtest;
            fy = fym + sty * dgtest;
            dgx = dgxm + dgtest;
            dgy = dgym + dgtest;
        } else {
            // this is ugly and some variables should be moved to the class scope
            cstep(stx, fx, dgx, sty, fy, dgy, stp, f, dg, brackt, stmin, stmax,
                  infoc);
        }

        if (brackt) {
            if (fabs(sty - stx) >= 0.66 * width1) stp = stx + 0.5 * (sty - stx);
            width1 = width;
            width = fabs(sty - stx);
        }
    }
        
    return 0.;
}


int MoreThuente::cstep(scalar_t &stx, scalar_t &fx, scalar_t &dx, scalar_t &sty,
                       scalar_t &fy, scalar_t &dy, scalar_t &stp, scalar_t &fp,
          scalar_t &dp, bool &brackt, scalar_t &stpmin,
          scalar_t &stpmax, int &info) {
    info = 0;
    bool bound = false;

    // Check the input parameters for errors.
    if ((brackt & ((stp <= std::min<scalar_t>(stx, sty)) |
                   (stp >= std::max<scalar_t>(stx, sty)))) |
        (dx * (stp - stx) >= 0.0) | (stpmax < stpmin)) {
        return -1;
    }

    scalar_t sgnd = dp * (dx / fabs(dx));

    scalar_t stpf = 0;
    scalar_t stpc = 0;
    scalar_t stpq = 0;

    if (fp > fx) {
        info = 1;
        bound = true;
        scalar_t theta = 3. * (fx - fp) / (stp - stx) + dx + dp;
        scalar_t s = std::max<scalar_t>(theta, std::max<scalar_t>(dx, dp));
        scalar_t gamma =
            s * sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s));
        if (stp < stx) gamma = -gamma;
        scalar_t p = (gamma - dx) + theta;
        scalar_t q = ((gamma - dx) + gamma) + dp;
        scalar_t r = p / q;
        stpc = stx + r * (stp - stx);
        stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2.) * (stp - stx);
        if (fabs(stpc - stx) < fabs(stpq - stx))
            stpf = stpc;
        else
            stpf = stpc + (stpq - stpc) / 2;
        brackt = true;
    } else if (sgnd < 0.0) {
        info = 2;
        bound = false;
        scalar_t theta = 3 * (fx - fp) / (stp - stx) + dx + dp;
        scalar_t s = std::max<scalar_t>(theta, std::max<scalar_t>(dx, dp));
        scalar_t gamma = s * sqrt((theta / s) * (theta / s) - (dx / s) * (dp / s));
        if (stp > stx) gamma = -gamma;
        scalar_t p = (gamma - dp) + theta;
        scalar_t q = ((gamma - dp) + gamma) + dx;
        scalar_t r = p / q;
        stpc = stp + r * (stx - stp);
        stpq = stp + (dp / (dp - dx)) * (stx - stp);
        if (fabs(stpc - stp) > fabs(stpq - stp))
            stpf = stpc;
        else
            stpf = stpq;
        brackt = true;
    } else if (fabs(dp) < fabs(dx)) {
        info = 3;
        bound = 1;
        scalar_t theta = 3 * (fx - fp) / (stp - stx) + dx + dp;
        scalar_t s = std::max<scalar_t>(theta, std::max<scalar_t>(dx, dp));
        scalar_t gamma = s * sqrt(std::max<scalar_t>(static_cast<scalar_t>(0.),
                                                     (theta / s) * (theta / s) -
                                                     (dx / s) * (dp / s)));
        if (stp > stx) gamma = -gamma;
        scalar_t p = (gamma - dp) + theta;
        scalar_t q = (gamma + (dx - dp)) + gamma;
        scalar_t r = p / q;
        if ((r < 0.0) & (gamma != 0.0)) {
            stpc = stp + r * (stx - stp);
        } else if (stp > stx) {
            stpc = stpmax;
        } else {
            stpc = stpmin;
        }
        stpq = stp + (dp / (dp - dx)) * (stx - stp);
        if (brackt) {
            if (fabs(stp - stpc) < fabs(stp - stpq)) {
                stpf = stpc;
            } else {
                stpf = stpq;
            }
        } else {
            if (fabs(stp - stpc) > fabs(stp - stpq)) {
                stpf = stpc;
            } else {
                    stpf = stpq;
                }
            }
        } else {
            info = 4;
            bound = false;
            if (brackt) {
                scalar_t theta = 3 * (fp - fy) / (sty - stp) + dy + dp;
                scalar_t s = std::max<scalar_t>(theta, std::max<scalar_t>(dy, dp));
                scalar_t gamma =
                    s * sqrt((theta / s) * (theta / s) - (dy / s) * (dp / s));
                if (stp > sty) gamma = -gamma;

                scalar_t p = (gamma - dp) + theta;
                scalar_t q = ((gamma - dp) + gamma) + dy;
                scalar_t r = p / q;
                stpc = stp + r * (sty - stp);
                stpf = stpc;
            } else if (stp > stx)
                stpf = stpmax;
            else {
                stpf = stpmin;
            }
        }

        if (fp > fx) {
            sty = stp;
            fy = fp;
            dy = dp;
        } else {
            if (sgnd < 0.0) {
                sty = stx;
                fy = fx;
                dy = dx;
            }

            stx = stp;
            fx = fp;
            dx = dp;
        }
        stpf = std::min<scalar_t>(stpmax, stpf);
        stpf = std::max<scalar_t>(stpmin, stpf);
        stp = stpf;

        if (brackt & bound) {
            if (sty > stx) {
                stp = std::min<scalar_t>(
                                         stx + static_cast<scalar_t>(0.66) * (sty - stx), stp);
            } else {
                stp = std::max<scalar_t>(
                                         stx + static_cast<scalar_t>(0.66) * (sty - stx), stp);
            }
        }

        return 0;
}


/**
 * Interfacing function for general line search methods in pele
 */
    double MoreThuente::line_search(Array<double> &x, Array<double> step)  {
    // assumes xold is set TODO : rewrite interface
    double alpha = MoreThuente::search(xold_, step);
    x_= x;
    for (size_t i = 0; i < xold_.size(); ++i) {
        x_[i] = xold_[i] + alpha*step[i];
    }
    double fend;
    opt_->compute_func_gradient(x_, fend, g_);
    opt_->set_f(fend);
    opt_->set_rms(norm(g_)/sqrt(g_.size()));
    return alpha*opt_->compute_pot_norm(step);
}
}



