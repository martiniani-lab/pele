#include "pele/petsc_ts.hpp"



namespace pele {
PETSCTSOptimizer::PETSCTSOptimizer(
    std::shared_ptr<pele::BasePotential> potential,
    const pele::Array<double> x0, double tol, double rtol, double atol)
    : GradientOptimizer(potential, x0, tol),
      N_size(x0.size()),
      t0(0),
      tN(10000000.0)
{
}
    
}