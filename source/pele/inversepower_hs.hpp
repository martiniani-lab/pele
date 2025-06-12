#ifndef _PELE_INVERSEPOWER_HS_H
#define _PELE_INVERSEPOWER_HS_H

#include <iostream>
#include <memory>
#include <cmath>
#include <limits>

#include "atomlist_potential.hpp"
#include "base_interaction.hpp"
#include "cell_list_potential.hpp"
#include "distance.hpp"
#include "meta_pow.hpp"
#include "simple_pairwise_ilist.hpp"
#include "simple_pairwise_potential.hpp"

namespace pele {

/**
 * Pairwise interaction for Inverse power potential with a hard sphere core
 * V = eps/pow * (1 - r/d_c)^pow for d_hs < r < d_c
 * V = infinity for r <= d_hs
 * V = 0 for r >= d_c
 *
 * d_hs is the hard sphere diameter (sum of radii)
 * d_c = d_hs * (1 + sca) is the cutoff distance
 * sca is the soft shell thickness parameter
 *
 * This is an adaption of the InversePower potential to include a hard-sphere
 * core, similar to the HS_WCA potential.
 */

struct InversePowerHS_interaction : BaseInteraction {
  double const _eps;
  double const _pow;
  double const _sca;
  double const _infty;

  InversePowerHS_interaction(double pow, double eps, double sca)
      : _eps(eps), _pow(pow), _sca(sca), _infty(std::numeric_limits<double>::infinity()) {}

  /* calculate energy from distance squared */
  double energy(double r2, const double dij_hs) const {
    const double d_hs2 = dij_hs * dij_hs;
    if (r2 <= d_hs2) {
      return _infty;
    }
    const double d_c = dij_hs * (1.0 + _sca);
    if (r2 >= d_c * d_c) {
      return 0.0;
    }
    const double r = std::sqrt(r2);
    return std::pow((1.0 - r / d_c), _pow) * _eps / _pow;
  }

  /* calculate energy and gradient from distance squared, gradient is in
   * -(dv/drij)/|rij| */
  double energy_gradient(double r2, double *gij,
                         const double dij_hs) const {
    const double d_hs2 = dij_hs * dij_hs;
    if (r2 <= d_hs2) {
      *gij = _infty;
      return _infty;
    }
    const double d_c = dij_hs * (1.0 + _sca);
    if (r2 >= d_c * d_c) {
      *gij = 0.0;
      return 0.0;
    }
    const double r = std::sqrt(r2);
    const double factor = std::pow((1.0 - r / d_c), _pow) * _eps;
    *gij = -factor / ((r - d_c) * r);
    return factor / _pow;
  }

  double inline energy_gradient_hessian(double r2, double *gij, double *hij,
                                        const double dij_hs) const {
    const double d_hs2 = dij_hs * dij_hs;
    if (r2 <= d_hs2) {
      *gij = _infty;
      *hij = _infty;
      return _infty;
    }
    const double d_c = dij_hs * (1.0 + _sca);
    if (r2 >= d_c * d_c) {
      *gij = 0.0;
      *hij = 0.0;
      return 0.0;
    }
    const double r = std::sqrt(r2);
    const double factor = std::pow((1.0 - r / d_c), _pow) * _eps;
    const double denom = 1.0 / (r - d_c);
    *gij = -factor * denom / r;
    *hij = (_pow - 1.0) * factor * denom * denom;
    return factor / _pow;
  }
};

template <int POW>
struct InverseIntPowerHS_interaction : BaseInteraction {
  double const _eps;
  double const _pow;
  double const _sca;
  double const _infty;

  InverseIntPowerHS_interaction(double eps, double sca)
      : _eps(eps), _pow(POW), _sca(sca), _infty(std::numeric_limits<double>::infinity()) {}

  /* calculate energy from distance squared */
  double energy(double r2, const double dij_hs) const {
    const double d_hs2 = dij_hs * dij_hs;
    if (r2 <= d_hs2) {
      return _infty;
    }
    const double d_c = dij_hs * (1.0 + _sca);
    if (r2 >= d_c * d_c) {
      return 0.0;
    }
    const double r = std::sqrt(r2);
    return pos_int_pow<POW>(1.0 - r / d_c) * _eps / _pow;
  }

  /* calculate energy and gradient from distance squared, gradient is in
   * -(dv/drij)/|rij| */
  double energy_gradient(double r2, double *gij,
                         const double dij_hs) const {
    const double d_hs2 = dij_hs * dij_hs;
    if (r2 <= d_hs2) {
      *gij = _infty;
      return _infty;
    }
    const double d_c = dij_hs * (1.0 + _sca);
    if (r2 >= d_c * d_c) {
      *gij = 0.0;
      return 0.0;
    }
    const double r = std::sqrt(r2);
    const double factor = pos_int_pow<POW>(1.0 - r / d_c) * _eps;
    *gij = -factor / ((r - d_c) * r);
    return factor / _pow;
  }

  double inline energy_gradient_hessian(double r2, double *gij, double *hij,
                                        const double dij_hs) const {
    const double d_hs2 = dij_hs * dij_hs;
    if (r2 <= d_hs2) {
      *gij = _infty;
      *hij = _infty;
      return _infty;
    }
    const double d_c = dij_hs * (1.0 + _sca);
    if (r2 >= d_c * d_c) {
      *gij = 0.0;
      *hij = 0.0;
      return 0.0;
    }
    const double r = std::sqrt(r2);
    const double factor = pos_int_pow<POW>(1.0 - r / d_c) * _eps;
    const double denom = 1.0 / (r - d_c);
    *gij = -factor * denom / r;
    *hij = (_pow - 1.0) * factor * denom * denom;
    return factor / _pow;
  }
};

template <int POW2>
struct InverseHalfIntPowerHS_interaction : BaseInteraction {
  double const _eps;
  double const _pow;
  double const _sca;
  double const _infty;

  InverseHalfIntPowerHS_interaction(double eps, double sca)
      : _eps(eps), _pow(0.5 * POW2), _sca(sca), _infty(std::numeric_limits<double>::infinity()) {}

  /* calculate energy from distance squared */
  double energy(double r2, const double dij_hs) const {
    const double d_hs2 = dij_hs * dij_hs;
    if (r2 <= d_hs2) {
      return _infty;
    }
    const double d_c = dij_hs * (1.0 + _sca);
    if (r2 >= d_c * d_c) {
      return 0.0;
    }
    const double r = std::sqrt(r2);
    return pos_half_int_pow<POW2>(1.0 - r / d_c) * _eps / _pow;
  }

  /* calculate energy and gradient from distance squared, gradient is in
   * -(dv/drij)/|rij| */
  double energy_gradient(double r2, double *gij,
                         const double dij_hs) const {
    const double d_hs2 = dij_hs * dij_hs;
    if (r2 <= d_hs2) {
      *gij = _infty;
      return _infty;
    }
    const double d_c = dij_hs * (1.0 + _sca);
    if (r2 >= d_c * d_c) {
      *gij = 0.0;
      return 0.0;
    }
    const double r = std::sqrt(r2);
    const double factor = pos_half_int_pow<POW2>(1.0 - r / d_c) * _eps;
    *gij = -factor / ((r - d_c) * r);
    return factor / _pow;
  }

  double inline energy_gradient_hessian(double r2, double *gij, double *hij,
                                        const double dij_hs) const {
    const double d_hs2 = dij_hs * dij_hs;
    if (r2 <= d_hs2) {
      *gij = _infty;
      *hij = _infty;
      return _infty;
    }
    const double d_c = dij_hs * (1.0 + _sca);
    if (r2 >= d_c * d_c) {
      *gij = 0.0;
      *hij = 0.0;
      return 0.0;
    }
    const double r = std::sqrt(r2);
    const double factor = pos_half_int_pow<POW2>(1.0 - r / d_c) * _eps;
    const double denom = 1.0 / (r - d_c);
    *gij = -factor * denom / r;
    *hij = (_pow - 1.0) * factor * denom * denom;
    return factor / _pow;
  }
};

template <size_t ndim>
class InversePowerHS
    : public SimplePairwisePotential<InversePowerHS_interaction,
                                     cartesian_distance<ndim>> {
 public:
  InversePowerHS(double pow, double eps, double sca,
                 pele::Array<double> const radii, bool exact_sum = false,
                 double non_additivity = 0.0)
      : SimplePairwisePotential<InversePowerHS_interaction,
                                cartesian_distance<ndim>>(
            std::make_shared<InversePowerHS_interaction>(pow, eps, sca),
            radii, std::make_shared<cartesian_distance<ndim>>(), sca,
            exact_sum, non_additivity) {}
};

template <size_t ndim>
class InversePowerHSPeriodic
    : public SimplePairwisePotential<InversePowerHS_interaction,
                                     periodic_distance<ndim>> {
 public:
  InversePowerHSPeriodic(double pow, double eps, double sca,
                         pele::Array<double> const radii,
                         pele::Array<double> const boxvec,
                         bool exact_sum = false, double non_additivity = 0.0)
      : SimplePairwisePotential<InversePowerHS_interaction,
                                periodic_distance<ndim>>(
            std::make_shared<InversePowerHS_interaction>(pow, eps, sca),
            radii, std::make_shared<periodic_distance<ndim>>(boxvec), sca,
            exact_sum, non_additivity) {}
};

template <size_t ndim, int POW>
class InverseIntPowerHS
    : public SimplePairwisePotential<InverseIntPowerHS_interaction<POW>,
                                     cartesian_distance<ndim>> {
 public:
  InverseIntPowerHS(double eps, double sca, pele::Array<double> const radii,
                    bool exact_sum = false, double non_additivity = 0.0)
      : SimplePairwisePotential<InverseIntPowerHS_interaction<POW>,
                                cartesian_distance<ndim>>(
            std::make_shared<InverseIntPowerHS_interaction<POW>>(eps, sca),
            radii, std::make_shared<cartesian_distance<ndim>>(), sca,
            exact_sum, non_additivity) {}
};

template <size_t ndim, int POW>
class InverseIntPowerHSPeriodic
    : public SimplePairwisePotential<InverseIntPowerHS_interaction<POW>,
                                     periodic_distance<ndim>> {
 public:
  InverseIntPowerHSPeriodic(double eps, double sca,
                            pele::Array<double> const radii,
                            pele::Array<double> const boxvec,
                            bool exact_sum = false,
                            double non_additivity = 0.0)
      : SimplePairwisePotential<InverseIntPowerHS_interaction<POW>,
                                periodic_distance<ndim>>(
            std::make_shared<InverseIntPowerHS_interaction<POW>>(eps, sca),
            radii, std::make_shared<periodic_distance<ndim>>(boxvec), sca,
            exact_sum, non_additivity) {}
};

template <size_t ndim, int POW2>
class InverseHalfIntPowerHS
    : public SimplePairwisePotential<InverseHalfIntPowerHS_interaction<POW2>,
                                     cartesian_distance<ndim>> {
 public:
  InverseHalfIntPowerHS(double eps, double sca,
                        pele::Array<double> const radii,
                        bool exact_sum = false, double non_additivity = 0.0)
      : SimplePairwisePotential<InverseHalfIntPowerHS_interaction<POW2>,
                                cartesian_distance<ndim>>(
            std::make_shared<InverseHalfIntPowerHS_interaction<POW2>>(
                eps, sca),
            radii, std::make_shared<cartesian_distance<ndim>>(), sca,
            exact_sum, non_additivity) {}
};

template <size_t ndim, int POW2>
class InverseHalfIntPowerHSPeriodic
    : public SimplePairwisePotential<InverseHalfIntPowerHS_interaction<POW2>,
                                     periodic_distance<ndim>> {
 public:
  InverseHalfIntPowerHSPeriodic(double eps, double sca,
                                pele::Array<double> const radii,
                                pele::Array<double> const boxvec,
                                bool exact_sum = false,
                                double non_additivity = 0.0)
      : SimplePairwisePotential<InverseHalfIntPowerHS_interaction<POW2>,
                                periodic_distance<ndim>>(
            std::make_shared<InverseHalfIntPowerHS_interaction<POW2>>(
                eps, sca),
            radii, std::make_shared<periodic_distance<ndim>>(boxvec), sca,
            exact_sum, non_additivity) {}
};

template <size_t ndim>
class InversePowerHSPeriodicCellLists
    : public CellListPotential<InversePowerHS_interaction,
                               periodic_distance<ndim>> {
 public:
  InversePowerHSPeriodicCellLists(
      double pow, double eps, double sca, pele::Array<double> const radii,
      pele::Array<double> const boxvec, const double ncellx_scale = 1.0,
      bool exact_sum = false, double non_additivity = 0.0)
      : CellListPotential<InversePowerHS_interaction, periodic_distance<ndim>>(
            std::make_shared<InversePowerHS_interaction>(pow, eps, sca),
            std::make_shared<periodic_distance<ndim>>(boxvec), boxvec,
            (1.0 + sca) * 2.0 *
                (*std::max_element(radii.begin(), radii.end())),  // rcut
            ncellx_scale, radii, sca, true, exact_sum, non_additivity) {}
};

template <size_t ndim, int POW>
class InverseIntPowerHSPeriodicCellLists
    : public CellListPotential<InverseIntPowerHS_interaction<POW>,
                               periodic_distance<ndim>> {
 public:
  InverseIntPowerHSPeriodicCellLists(
      double eps, double sca, pele::Array<double> const radii,
      pele::Array<double> const boxvec, const double ncellx_scale = 1.0,
      bool exact_sum = false, double non_additivity = 0.0)
      : CellListPotential<InverseIntPowerHS_interaction<POW>,
                          periodic_distance<ndim>>(
            std::make_shared<InverseIntPowerHS_interaction<POW>>(eps, sca),
            std::make_shared<periodic_distance<ndim>>(boxvec), boxvec,
            (1.0 + sca) * 2.0 *
                (*std::max_element(
                    radii.begin(),
                    radii.end())),  // rcut, Note if boxvec or radii are of
                                    // size 0 this can lead to a segfault
            ncellx_scale, radii, sca, true, exact_sum, non_additivity) {}
};

template <size_t ndim, int POW2>
class InverseHalfIntPowerHSPeriodicCellLists
    : public CellListPotential<InverseHalfIntPowerHS_interaction<POW2>,
                               periodic_distance<ndim>> {
 public:
  InverseHalfIntPowerHSPeriodicCellLists(
      double eps, double sca, pele::Array<double> const radii,
      pele::Array<double> const boxvec, const double ncellx_scale = 1.0,
      bool exact_sum = false, double non_additivity = 0.0)
      : CellListPotential<InverseHalfIntPowerHS_interaction<POW2>,
                          periodic_distance<ndim>>(
            std::make_shared<InverseHalfIntPowerHS_interaction<POW2>>(
                eps, sca),
            std::make_shared<periodic_distance<ndim>>(boxvec), boxvec,
            (1.0 + sca) * 2.0 *
                (*std::max_element(radii.begin(), radii.end())),  // rcut,
            ncellx_scale, radii, sca, true, exact_sum, non_additivity) {}
};

}  // namespace pele

#endif  // #ifndef _PELE_INVERSEPOWER_HS_H 