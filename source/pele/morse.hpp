#ifndef _PELE_MORSE_H_
#define _PELE_MORSE_H_

#include "simple_pairwise_potential.hpp"
#include "simple_pairwise_ilist.hpp"
#include "atomlist_potential.hpp"
#include "distance.hpp"
#include "frozen_atoms.hpp"
#include "base_interaction.hpp"
#include <cmath>
#include <memory>

namespace pele
{

/**
 * Define a pairwise interaction for morse with a cutoff.  The
 * potential goes is continuous but not smooth.
 */
struct morse_interaction : BaseInteraction {
    double const _A;
    double const _rho;
    double const _r0;
    morse_interaction(double rho, double r0, double A)
        : _A(A), _rho(rho), _r0(r0)
    {}

    /* calculate energy from distance squared */
    double inline energy(double r2, const double radius_sum) const
    {
        double r = std::sqrt(r2);
        double c = std::exp(-_rho * (r - _r0));
        return _A * c * (c - 2.);
    }

    /* calculate energy and gradient from distance squared, gradient is in g/|rij| */
    double inline energy_gradient(double r2, double *gij, const double radius_sum) const
    {
        double r = std::sqrt(r2);
        double c = std::exp(-_rho * (r - _r0));
        *gij = 2.0 * _A * c * _rho * (c - 1.0) / r;
        return _A * c * (c - 2.0);
    }

    double inline energy_gradient_hessian(double r2, double *gij, double *hij, const double radius_sum) const
    {
        double r = std::sqrt(r2);
        double c = std::exp(-_rho * (r - _r0));
        //double A_rho_2_c =
        *gij = 2.0 * _A * c * _rho * (c - 1.0) / r;
        *hij = 2.0 * _A * c * _rho * _rho * (2.0 * c - 1.0);
        return _A * c * (c - 2.0);
    }
};

/**
 * Pairwise Lennard-Jones potential with smooth cutoff
 */
class Morse: public SimplePairwisePotential<morse_interaction>
{
public:
    Morse(double rho, double r0, double A)
        : SimplePairwisePotential<morse_interaction>(
                std::make_shared<morse_interaction>(rho, r0, A) )
    {}
};
}
#endif
