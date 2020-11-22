/**
 * Extra wrappers for use by PETSC/TAO from the potential
 */






#include "pele/base_potential.hpp"
#include <petsctao.h>
#include "pele/array.hpp"
#include "petscvec.h"
#include "petscviewer.h"

using namespace pele;

/**
 * wrapper workaround for passing a potential since passing a naked pointer
 * doesn't seem to work as well.
 */
typedef struct {
    std::shared_ptr<pele::BasePotential>  potential_;
} pot_sptr_wrapper;


/**
 * @brief get_energy_gradient wrapper for the TAO optimizers. calculates
 * energy(f) and gradient(G)
 * @input Tao: optimizer
 *        X : coordinates of the input
 *        f : energy
 *        G : gradient
 *        ptr: pointer to a wrapper of base_potential
 */
inline PetscErrorCode TaoBasePotentialFunctionGradient(Tao tao,Vec X,PetscReal *f, Vec G,void *ptr)
{   
    pot_sptr_wrapper            *user = (pot_sptr_wrapper *) ptr;
    PetscFunctionBeginUser;
    *f = user->potential_->get_energy_gradient_petsc(X, G);
    PetscFunctionReturn(0); 
}



/**
 * @brief      Calculates the hessian for BasePotential
 *
 * @details    Function that wraps get_negative_hessian_sparse for use by the TAO optimizer.
 *             This means we get the negative hessian and then take the negative of it to get the actual one
 *             (kind of annoying but we're probably going to add an extra method to get the positive hessian)
 *             Also this routine does not calculate an extra preconditioner
 *
 * @param      Tao: The optimizer.
 *             X  : The coordinates of the particles.
 *             H : The hessian
 *             Hpre: The hessian preconditioner
 *             ptr: pointer to a wrapper of base_potential
 *
 * @return     H, Hpre 
 */
inline PetscErrorCode TaoBasePotentialHessian(Tao tao,Vec X,Mat H, Mat Hpre, void *ptr)
{
    PetscErrorCode ierr;
    pot_sptr_wrapper            *user = (pot_sptr_wrapper *) ptr;
    PetscFunctionBeginUser;
    // get -hessian
    user->potential_->get_hessian_petsc(X, H);
    PetscFunctionReturn(0);
}


