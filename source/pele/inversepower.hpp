#ifndef _PELE_INVERSEPOWER_H
#define _PELE_INVERSEPOWER_H

#include <memory>

#include "atomlist_potential.hpp"
#include "cell_list_potential.hpp"
#include "distance.hpp"
#include "meta_pow.hpp"
#include "simple_pairwise_ilist.hpp"
#include "simple_pairwise_potential.hpp"
#include "base_interaction.hpp"

namespace pele {

/**
 * Pairwise interaction for Inverse power potential eps/pow * (1 - r/r0)^pow
 * the radii array allows for polydispersity
 * The most common exponents are:
 * pow=2   -> Hookean interaction
 * pow=2.5 -> Hertzian interaction
 *
 * Comments about this implementation:
 *
 * This implementation is using STL pow for all exponents. Performance wise this may not be
 * the fastest possible implementation, for instance using exp(pow*log) could be faster
 * (twice as fast according to some blogs).
 *
 * Maybe we should consider a function that calls exp(pow*log) though, this should be carefully benchmarked
 * though as my guess is that the improvement is going to be marginal and will depend on the
 * architecture (how well pow, exp and log can be optimized on a given architecture).
 *
 * See below for meta pow implementations for integer and half-integer exponents.
 *
 * If you have any experience with pow please suggest any better solution and/or provide a
 * faster implementation.
 */

struct InversePower_interaction : BaseInteraction {
    double const _eps;
    double const _pow;

    InversePower_interaction(double pow, double eps)
        : _eps(eps),
          _pow(pow)
    {}

    /* calculate energy from distance squared */
    double energy(double r2, const double radius_sum) const
    {
        double E;
        if (r2 >= radius_sum * radius_sum) {
            E = 0.;
        }
        else {
            // Sqrt moved into else, based on previous comment by CPG.
            const double r = std::sqrt(r2);
            E = std::pow((1 -r/radius_sum), _pow) * _eps/_pow;
        }
        return E;
    }

    /* calculate energy and gradient from distance squared, gradient is in -(dv/drij)/|rij| */
    double energy_gradient(double r2, double *gij, const double radius_sum) const
    {
        double E;
        if (r2 >= radius_sum * radius_sum) {
            E = 0.;
            *gij = 0.;
        }
        else {
            const double r = std::sqrt(r2);
            const double factor = std::pow((1 -r/radius_sum), _pow) * _eps;
            E =  factor / _pow;
            *gij =  - factor / ((r-radius_sum)*r);
        }
        return E;
    }

    double inline energy_gradient_hessian(double r2, double *gij, double *hij, const double radius_sum) const
    {
        double E;
        if (r2 >= radius_sum * radius_sum) {
            E = 0.;
            *gij = 0;
            *hij=0;
        }
        else {
            const double r = std::sqrt(r2);
            const double factor = std::pow((1 -r/radius_sum), _pow) * _eps;
            const double denom = 1.0 / (r-radius_sum);
            E =  factor / _pow;
            *gij =  - factor * denom / r ;
            *hij = (_pow-1) * factor * denom * denom;
        }
        return E;
    }
};

template<int POW>
struct InverseIntPower_interaction : BaseInteraction {
    double const _eps;
    double const _pow;

    InverseIntPower_interaction(double eps)
        : _eps(eps),
          _pow(POW)
    {}

    /* calculate energy from distance squared */
    double energy(double r2, const double radius_sum) const
    {
        double E;
        if (r2 >= radius_sum * radius_sum) {
            E = 0.;
        }
        else {
            const double r = std::sqrt(r2);
            E = pos_int_pow<POW>(1 -r/radius_sum) * _eps/_pow;
        }
        return E;
    }

    /* calculate energy and gradient from distance squared, gradient is in -(dv/drij)/|rij| */
    double energy_gradient(double r2, double *gij, const double radius_sum) const
    {
        double E;
        if (r2 >= radius_sum * radius_sum) {
            E = 0.;
            *gij = 0;
        }
        else {
            const double r = std::sqrt(r2);
            const double factor = pos_int_pow<POW>(1 -r/radius_sum) * _eps;
            E =  factor / _pow;
            *gij =  - factor / ((r-radius_sum)*r);
        }
        return E;
    }

    double inline energy_gradient_hessian(double r2, double *gij, double *hij, const double radius_sum) const
    {
        double E;
        if (r2 >= radius_sum * radius_sum) {
            E = 0.;
            *gij = 0;
            *hij=0;
        }
        else {
            const double r = std::sqrt(r2);
            const double factor = pos_int_pow<POW>(1 -r/radius_sum) * _eps;
            const double denom = 1.0 / (r-radius_sum);
            E =  factor / _pow;
            *gij =  - factor * denom / r ;
            *hij = (_pow-1) * factor * denom * denom;
        }
        return E;
    }
};

template<int POW2>
struct InverseHalfIntPower_interaction : BaseInteraction {
    double const _eps;
    double const _pow;

    InverseHalfIntPower_interaction(double eps)
        : _eps(eps),
          _pow(0.5 * POW2)
    {}

    /* calculate energy from distance squared */
    double energy(double r2, const double radius_sum) const
    {
        double E;
        if (r2 >= radius_sum * radius_sum) {
            E = 0.;
        }
        else {
            const double r = std::sqrt(r2);
            E = pos_half_int_pow<POW2>(1 -r/radius_sum) * _eps/_pow;
        }
        return E;
    }

    /* calculate energy and gradient from distance squared, gradient is in -(dv/drij)/|rij| */
    double energy_gradient(double r2, double *gij, const double radius_sum) const
    {
        double E;
        if (r2 >= radius_sum * radius_sum) {
            E = 0.;
            *gij = 0;
        }
        else {
            const double r = std::sqrt(r2);
            const double factor = pos_half_int_pow<POW2>(1 -r/radius_sum) * _eps;
            E =  factor / _pow;
            *gij =  - factor / ((r-radius_sum)*r);
        }
        return E;
    }

    double inline energy_gradient_hessian(double r2, double *gij, double *hij, const double radius_sum) const
    {
        double E;
        if (r2 >= radius_sum * radius_sum) {
            E = 0.;
            *gij = 0;
            *hij = 0;
        }
        else {
            const double r = std::sqrt(r2);
            const double factor = pos_half_int_pow<POW2>(1 -r/radius_sum) * _eps;
            const double denom = 1.0 / (r-radius_sum);
            E =  factor / _pow;
            *gij =  - factor * denom / r ;
            *hij = (_pow-1) * factor * denom * denom;
        }
        return E;
    }
};


//
// combine the components (interaction, looping method, distance function) into
// defined classes
//

/**
 * Pairwise Inverse Power potential
 */
template <size_t ndim>
class InversePower : public SimplePairwisePotential< InversePower_interaction, cartesian_distance<ndim> > {
public:
    InversePower(double pow, double eps, pele::Array<double> const radii)
        : SimplePairwisePotential< InversePower_interaction, cartesian_distance<ndim> >(
                std::make_shared<InversePower_interaction>(pow, eps),
                radii,
                std::make_shared<cartesian_distance<ndim> >()
          )
    {}
};

template <size_t ndim>
class InversePowerPeriodic : public SimplePairwisePotential< InversePower_interaction, periodic_distance<ndim> > {
public:
    InversePowerPeriodic(double pow, double eps, pele::Array<double> const radii, pele::Array<double> const boxvec)
        : SimplePairwisePotential< InversePower_interaction, periodic_distance<ndim> >(
                std::make_shared<InversePower_interaction>(pow, eps),
                radii,
                std::make_shared<periodic_distance<ndim> >(boxvec)
          )
    {}
};

/**
 * Integer powers
 */

template <size_t ndim, int POW>
class InverseIntPower : public SimplePairwisePotential< InverseIntPower_interaction<POW>, cartesian_distance<ndim> > {
public:
    InverseIntPower(double eps, pele::Array<double> const radii)
        : SimplePairwisePotential< InverseIntPower_interaction<POW>, cartesian_distance<ndim> >(
                std::make_shared<InverseIntPower_interaction<POW> >(eps),
                radii,
                std::make_shared<cartesian_distance<ndim> >()
          )
    {}
};

template <size_t ndim, int POW>
class InverseIntPowerPeriodic : public SimplePairwisePotential< InverseIntPower_interaction<POW>, periodic_distance<ndim> > {
public:
    InverseIntPowerPeriodic(double eps, pele::Array<double> const radii, pele::Array<double> const boxvec)
        : SimplePairwisePotential< InverseIntPower_interaction<POW>, periodic_distance<ndim> >(
                std::make_shared<InverseIntPower_interaction<POW> >(eps),
                radii,
                std::make_shared<periodic_distance<ndim> >(boxvec)
          )
    {}
};

/**
 * Half-integer powers
 */

template <size_t ndim, int POW2>
class InverseHalfIntPower : public SimplePairwisePotential< InverseHalfIntPower_interaction<POW2>, cartesian_distance<ndim> > {
public:
    InverseHalfIntPower(double eps, pele::Array<double> const radii)
        : SimplePairwisePotential< InverseHalfIntPower_interaction<POW2>, cartesian_distance<ndim> >(
                std::make_shared<InverseHalfIntPower_interaction<POW2> >(eps),
                radii,
                std::make_shared<cartesian_distance<ndim> >()
          )
    {}
};

template <size_t ndim, int POW2>
class InverseHalfIntPowerPeriodic : public SimplePairwisePotential< InverseHalfIntPower_interaction<POW2>, periodic_distance<ndim> > {
public:
    InverseHalfIntPowerPeriodic(double eps, pele::Array<double> const radii, pele::Array<double> const boxvec)
        : SimplePairwisePotential< InverseHalfIntPower_interaction<POW2>, periodic_distance<ndim> >(
                std::make_shared<InverseHalfIntPower_interaction<POW2> >(eps),
                radii,
                std::make_shared<periodic_distance<ndim> >(boxvec)
          )
    {}
};


/*
template <size_t ndim>
class InversePowerCellLists : public CellListPotential< InversePower_interaction, cartesian_distance<ndim> > {
public:
    InversePowerCellLists(double pow, double eps,
            pele::Array<double> const radii, pele::Array<double> const boxvec,
            const double rcut,
            const double ncellx_scale=1.0)
        : CellListPotential< InversePower_interaction, cartesian_distance<ndim> >(
                std::make_shared<InversePower_interaction>(pow, eps),
                std::make_shared<cartesian_distance<ndim> >(),
                boxvec, rcut, ncellx_scale, radii)
    {}
};
*/

template <size_t ndim>
class InversePowerPeriodicCellLists : public CellListPotential< InversePower_interaction, periodic_distance<ndim> > {
public:
    InversePowerPeriodicCellLists(double pow, double eps,
            pele::Array<double> const radii, pele::Array<double> const boxvec,
            const double ncellx_scale=1.0)
        : CellListPotential< InversePower_interaction, periodic_distance<ndim> >(
                std::make_shared<InversePower_interaction>(pow, eps),
                std::make_shared<periodic_distance<ndim> >(boxvec),
                boxvec,
				2.0* (*std::max_element(radii.begin(), radii.end())), // rcut,
				ncellx_scale, radii)
    {}
};

template <size_t ndim, int POW>
class InverseIntPowerPeriodicCellLists : public CellListPotential< InverseIntPower_interaction<POW>, periodic_distance<ndim> > {
public:
    InverseIntPowerPeriodicCellLists(double eps, pele::Array<double> const radii,
            pele::Array<double> const boxvec, const double ncellx_scale=1.0)
        : CellListPotential< InverseIntPower_interaction<POW>, periodic_distance<ndim> >(
                std::make_shared<InverseIntPower_interaction<POW>>(eps),
                std::make_shared<periodic_distance<ndim> >(boxvec),
                boxvec,
				2.0* (*std::max_element(radii.begin(), radii.end())), // rcut,
				ncellx_scale, radii)
    {}
};

template <size_t ndim, int POW2>
class InverseHalfIntPowerPeriodicCellLists : public CellListPotential< InverseHalfIntPower_interaction<POW2>, periodic_distance<ndim> > {
public:
    InverseHalfIntPowerPeriodicCellLists(double eps, pele::Array<double> const radii,
            pele::Array<double> const boxvec, const double ncellx_scale=1.0)
        : CellListPotential< InverseHalfIntPower_interaction<POW2>, periodic_distance<ndim> >(
                std::make_shared<InverseHalfIntPower_interaction<POW2>>(eps),
                std::make_shared<periodic_distance<ndim> >(boxvec),
                boxvec,
				2.0* (*std::max_element(radii.begin(), radii.end())), // rcut,
				ncellx_scale, radii)
    {}
};

} //namespace pele

#endif //#ifndef _PELE_INVERSEPOWER_H
