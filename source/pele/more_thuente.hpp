// The MIT License (MIT)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Modified from https://github.com/PatWie/CppNumericalSolvers
// Changes involve removing hardcoded options and interfacing with the pele LBFGS
// and reorganizing the functions
// note:: functions are no longer static

// see also ../more_thuente.cpp since the copied/modified functions are defined there


// Copyright 2020, https://github.com/PatWie/CppNumericalSolvers
//
#ifndef INCLUDE_CPPOPTLIB_LINESEARCH_MORE_THUENTE_H_
#define INCLUDE_CPPOPTLIB_LINESEARCH_MORE_THUENTE_H_

#include <cmath>
#include <iostream>
#include <Eigen/Dense>
#include "linesearch.hpp"
#include "eigen_interface.hpp"

namespace pele {

// to keep consistency
using scalar_t = double;
using vector_t = Array<double>;




class MoreThuente : public LineSearch {
public:
    MoreThuente(GradientOptimizer * opt, double alpha_max = 1e15, double alpha_min=1e-15,
                double x_tol = 1e-15, double f_tol=1e-10, double g_tol=1e-4, double x_trapf = 4, double max_fev = 40,double alphainit = 1.0) :
        LineSearch(opt, alpha_max),
        xtol(x_tol),
        ftol(f_tol),
        gtol(g_tol),
        xtrapf(x_trapf),
        maxfev(max_fev),
        alphamin(alpha_min),
        alphamax(alpha_max),
        alpha_init(alphainit)
        
    {};

    ~MoreThuente() {};
    // interfaced line search the line search assumes relevant variables are changed in the optimizer
    double line_search(Array<double> &x, Array<double> step);
    

    // interfacing methods lbfgs
public:
    void inline set_xold_gold_(Array<double> &xold, Array<double>  &gold) {
        gold_=gold;
        xold_=xold;
    };
    // helper function to assign g_ and f
    void inline set_g_f_ptr(Array<double> &g) {g_ = g;};

   
    // input variables
private:
    // info about the More Thuente Line search

    scalar_t alpha_init;

    // tolerance variables determine how close to zero we can be
    scalar_t xtol;
    scalar_t ftol;
    scalar_t gtol;
    // stuff to interface with LBFGS TODO : get rid and make a uniform interface
    Array<double> xold_;
    Array<double> gold_;
    Array<double> g_;
    Array<double> x_;
    // stepmin and stepmax variables
    scalar_t alphamin;
    scalar_t alphamax;
    
    // figure this out
    scalar_t xtrapf;
    int maxfev;
    bool desdir;

    // methods 
public:
    inline scalar_t search(const vector_t &x, const vector_t &search_direction) {
        // Assumed step width.
        scalar_t ak = alpha_init;
        Array<double> g(x.size());
        scalar_t fval;
        opt_->compute_func_gradient(x, fval, g);
        vector_t s = search_direction;
        vector_t xx = x.copy();
        cvsrch(xx, fval, g, ak, s);
        return ak;
    }
    /**
     * @brief use MoreThuente Rule for (strong) Wolfe conditiions
     * @details [long description
     *
     * @param search_direction search direction for next update step
     * @param function handle to problem
     *
     * @return step-width
     */
    int cvsrch(vector_t &x, scalar_t f,
               vector_t &g, scalar_t &stp, vector_t &s);

    int cstep(scalar_t &stx, scalar_t &fx, scalar_t &dx, scalar_t &sty,
              scalar_t &fy, scalar_t &dy, scalar_t &stp, scalar_t &fp,
              scalar_t &dp, bool &brackt, scalar_t &stpmin,
              scalar_t &stpmax, int &info);
    
};




};  // namespace pele

#endif  // INCLUDE_CPPOPTLIB_LINESEARCH_MORE_THUENTE_H_