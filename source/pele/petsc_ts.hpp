#ifndef _PELE_CVODE_OPT_H__
#define _PELE_CVODE_OPT_H__
/**
 * "optimizer" based on PETSc's TS solver. Not technically an optimizer, but
 * should help us identify the path of steepest descent.
 */

#include "base_potential.hpp"
#include "array.hpp"
#include "debug.hpp"


#include "petscsnes.h"
#include "petscsys.h"
#include "petscsystypes.h"
#include "petscvec.h"
#include <petscksp.h>


namespace pele {
class PETSCTSOptimizer : public GradientOptimizer {
private:
    size_t N_size;
    double t0;
    double tN;
    // sparse calculation initializers
    Mat petsc_jacobian;
    Vec petsc_grad;
    Vec residual;
    PetscInt blocksize;
    // average number of non zeros per block for memory allocation purposes
    PetscInt hessav;
    SNES  snes;
public:
    void one_iteration();
    PETSCTSOptimizer(std::shared_ptr<pele::BasePotential> potential,
                     const pele::Array<double> x0,
                      double tol=1e-5,
                      double rtol=1e-4,
                      double atol=1e-4);
    ~PETSCTSOptimizer();

protected:
    
};
}







#endif