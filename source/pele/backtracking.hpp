/**
 * backtracking LineSearch given in as taken from https://github.com/yixuan/LBFGSpp/ wrapped around in nwpele
 */

// Copyright (C) 2016-2020 Yixuan Qiu <yixuan.qiu@cos.name>
// Under MIT license

#ifndef PELE_LINE_SEARCH_BACKTRACKING_H
#define PELE_LINE_SEARCH_BACKTRACKING_H
#include "optimizer.hpp"
#include "linesearch.hpp"
#include "lsparameters.hpp"
#include "eigen_interface.hpp"
#include "debug.hpp"





using Scalar=double;
typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;


namespace pele {
class BacktrackingLineSearch : LineSearch {
public:
    BacktrackingLineSearch(GradientOptimizer *opt,
                           Scalar inital_stpsize_        = Scalar(1.0),
                           Scalar max_step_       = Scalar(10.0),
                           int    m_              = 6,
                           Scalar epsilon_        = Scalar(1e-5),
                           Scalar epsilon_rel_    = Scalar(1e-5),
                           int    past_           = 0,
                           Scalar delta_          = Scalar(0),
                           int    max_linesearch_ = 40,
                           Scalar min_step_       = 0,
                           Scalar ftol_           = Scalar(1e-4),
                           Scalar wolfe_          = Scalar(0.9),
                           Scalar lstype_         = LINESEARCH_BACKTRACKING_ARMIJO) :
        LineSearch(opt, max_step_),
        params(m_,
               epsilon_,
               epsilon_rel_,
               past_,
               delta_,
               max_linesearch_,
               min_step_,
               max_step_,
               ftol_,
               wolfe_,
               lstype_),
        xsize(opt->get_x().size()),
        xdum(opt->get_x().copy()),
        gdum(xsize),
        xvec(xsize),
        gradvec(xsize),
        initial_stpsize(inital_stpsize_),
        step_direction(xsize),
        xoldvec(xsize)
    {
        std::cout << inital_stpsize_ << "stpsize \n";
            std::cout << max_step_ << "max step \n";
    };
    virtual ~BacktrackingLineSearch() {};
    inline void set_xold_gold_(Array<double> &xold, Array<double>  &gold) {
        gold_=gold;
        xold_=xold;
    };
    // helper function to assign g_ and f
    inline void set_g_f_ptr(Array<double> &g) {g_ = g;};
    double line_search(Array<double> & x, Array<double> step);
    inline double get_initial_stpsize() {return initial_stpsize;};
        
private:
    LBFGSParam params;
    int xsize;
    double func_grad_wrapper(Vector &x, Vector &grad);
    // These should reflect the xold in the old version
    void LSFunc(Scalar& fx, Vector& x, Vector& grad,
                Scalar& step,
                const Vector& drt, const Vector& xp,
                const LBFGSParam& param);

private:
    Array<double> g_;
    Array<double> xold_;
    Array<double> gold_;
    Array<double> xdum;
    Array<double> gdum;
    Vector xvec;
    Vector gradvec;
    Scalar initial_stpsize;
    Vector step_direction;
    Vector xoldvec;
};
}  // pele

#endif  // end #ifndef PELE_LINE_SEARCH_BACKTRACKING_H