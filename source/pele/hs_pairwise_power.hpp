#ifndef _PELE_HS_IP_H
#define _PELE_HS_IP_H

#include <algorithm>
#include <memory>

#include "atomlist_potential.hpp"
#include "base_interaction.hpp"
#include "cell_list_potential.hpp"
#include "distance.hpp"
#include "meta_pow.hpp"
#include "simple_pairwise_ilist.hpp"
#include "simple_pairwise_potential.hpp"

namespace pele {

/**
 * Pairwise power law potential with a hard core,
 * potential form is the same as in inversepower.hpp
 * but with a hard core to make basin volume calculations faster.
 * The current version is heavily optimized for and only for half integer powers
 * POW2 = 2*power and is expected to be odd
 */
template <int POW2>
struct HS_halfint_pairwise_power_interaction : BaseInteraction {
  const double _pow;
  const double _eps;
  const double _sca;
  const double _infty;

  HS_halfint_pairwise_power_interaction(const double eps, const double sca)
      : _pow(0.5 * POW2), _eps(eps), _sca(sca), _infty(std::pow(10.0, 50)) {}

  /* calculate energy from distance squared, r0 is the hard core distance, r is
   * the distance between the centres */
  double inline energy(const double r2, const double dij) const {
    const double r02 = dij * dij;
    if (r2 <= r02) {
      return _infty;
    }
    const double coff =
        dij * (1.0 + _sca);  // distance at which the soft cores are at contact
    if (r2 > coff * coff) {
      return 0;
    }
    const double r = std::sqrt(r2);
    return pos_half_int_pow<POW2>(1 - r / dij) * _eps / _pow;
  }

  /* calculate energy and gradient from distance squared, gradient is in
   * g/|rij|, dij is the hard core distance, r is the distance between
   * the centres */
  double inline energy_gradient(const double r2, double *const gij,
                                const double dij) const {
    const double r02 = dij * dij;
    if (r2 <= r02) {
      *gij = _infty;
      return _infty;
    }
    const double coff =
        dij * (1.0 + _sca);  // distance at which the soft cores are at contact
    if (r2 > coff * coff) {
      *gij = 0.;
      return 0.;
    }
    const double r = std::sqrt(r2);
    const double factor = pos_half_int_pow<POW2>(1 - r / dij) * _eps;
    *gij = -factor / ((r - dij) * r);
    return factor / _pow;
  }

  double inline energy_gradient_hessian(const double r2, double *const gij,
                                        double *const hij,
                                        const double dij) const {
    const double r02 = dij * dij;
    if (r2 <= r02) {
      *gij = _infty;
      *hij = _infty;
      return _infty;
    }
    const double coff =
        dij * (1.0 + _sca);  // distance at which the soft cores are at contact
    if (r2 > coff * coff) {
      *gij = 0.;
      *hij = 0.;
      return 0.;
    }
    const double r = std::sqrt(r2);
    const double factor = pos_half_int_pow<POW2>(1 - r / dij) * _eps;
    const double denom = 1.0 / (r - dij);

    *gij = -factor * denom / r;
    *hij = (_pow - 1) * factor * denom * denom;
    return factor / _pow;
  }
};

/**
 * Half-integer powers
 */

template <size_t ndim, int POW2>
class HSHalfIntPairwisePower : public SimplePairwisePotential<
                                   HS_halfint_pairwise_power_interaction<POW2>,
                                   cartesian_distance<ndim>> {
 public:
  HSHalfIntPairwisePower(double eps, pele::Array<double> const radii,
                         bool exact_sum = false, double non_additivity = 0.0)
      : SimplePairwisePotential<HS_halfint_pairwise_power_interaction<POW2>,
                                cartesian_distance<ndim>>(
            std::make_shared<HS_halfint_pairwise_power_interaction<POW2>>(eps),
            radii, std::make_shared<cartesian_distance<ndim>>(), 0.0, exact_sum,
            non_additivity) {}
};

template <size_t ndim, int POW2>
class HSHalfIntPairwisePowerPeriodic
    : public SimplePairwisePotential<
          HS_halfint_pairwise_power_interaction<POW2>,
          periodic_distance<ndim>> {
 public:
  HSHalfIntPairwisePowerPeriodic(double eps, pele::Array<double> const radii,
                                 pele::Array<double> const boxvec,
                                 bool exact_sum = false,
                                 double non_additivity = 0.0)
      : SimplePairwisePotential<HS_halfint_pairwise_power_interaction<POW2>,
                                periodic_distance<ndim>>(
            std::make_shared<HS_halfint_pairwise_power_interaction<POW2>>(eps),
            radii, std::make_shared<periodic_distance<ndim>>(boxvec), 0.0,
            exact_sum, non_additivity) {}
};
template <size_t ndim, int POW2>
class HSHalfIntPairwisePowerPeriodicCellLists
    : public CellListPotential<HS_halfint_pairwise_power_interaction<POW2>,
                               periodic_distance<ndim>> {
 public:
  HSHalfIntPairwisePowerPeriodicCellLists(double eps,
                                          pele::Array<double> const radii,
                                          pele::Array<double> const boxvec,
                                          const double ncellx_scale = 1.0,
                                          bool exact_sum = false,
                                          double non_additivity = 0.0)
      : CellListPotential<HS_halfint_pairwise_power_interaction<POW2>,
                          periodic_distance<ndim>>(
            std::make_shared<HS_halfint_pairwise_power_interaction<POW2>>(eps),
            std::make_shared<periodic_distance<ndim>>(boxvec), boxvec,
            2.0 * (*std::max_element(radii.begin(), radii.end())),  // rcut,
            ncellx_scale, radii, 0.0, true, exact_sum, non_additivity) {}
};

}  // namespace pele

#endif  // _PELE_HS_IP_H