#ifndef POTENTIAL_WRAPPER
#define POTENTIAL_WRAPPER

#include "pele/array.hpp"
#include "pele/base_potential.hpp"

typedef GetEnergyFunction = std::function


/*
 * @brief Wraps get_energy, get_energy_gradient, and
 * get_energy_gradient_hessians into a single Potential class for use with pele
 * optimizers
 *
 */
class PotentialWrapper : public pele::BasePotential {



};

#endif  // !POTENTIAL_WRAPPER