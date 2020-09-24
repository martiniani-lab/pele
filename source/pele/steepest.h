#ifndef _PELE_STEEPEST_DESCENT_H__
#define _PELE_STEEPEST_DESCENT_H__


/**
 * A faster steepest descent implementation in pele. generally slow but works on ill conditioned problems
 */

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
#include "more_thuente.h"
#include "linesearch.h"
#include "optimizer.h"
#include "nwpele.h"
#include "backtracking.h"
#include "bracketing.h"



namespace pele {

/**
 * An implementation of the steepest descent optimization algorithm in c++.
 */


class Steepest_Descent : public GradientOptimizer
{
private:
    double m_initial_step;        // initial step size for steepest descent
    std::shared_ptr<LineSearch>  line_search_method;
    
public:
    Steepest_Descent( std::shared_ptr<pele::BasePotential> potential, const pele::Array<double> x0,
                      double tol = 1e-4, double initial_step=1)     :
        GradientOptimizer(potential, x0, tol),
        m_initial_step(initial_step)
        line_search_method(std::make_shared<BacktrackingLineSearch> )
    {};
    virtual ~Steepest_Descent() {};

};





}  // pele




#endif  // end _PELE_STEEPEST_DESCENT_H__