//
//

#ifndef INVERSE_POWER_STILINGER_CUT_QUAD_H
#define INVERSE_POWER_STILINGER_CUT_QUAD_H

#include <cmath>
#include <cstddef>
#include <iostream>
#include <pele/meta_pow.hpp>
#include <pele/pairwise_potential_interface.hpp>
#include <stdexcept>

#include "base_interaction.hpp"
#include "pele/cell_list_potential.hpp"
#include "pele/simple_pairwise_potential.hpp"

namespace pele {

// Inverse power law potential with quadratic/quartic cutoff
// as seen in https://journals.aps.org/prx/pdf/10.1103/PhysRevX.12.021001
// as the soft sphere potential in the Appendix
// implemented to generate identical results as in paper
// The potential is given by
// U_{ij}(r) = v0 * ( (dij/r)^n + c0 + c2*(r/dij)^2 + c4*(r/dij)^4 )
// where dij is a factor dependent on diameters of the two particles
// and c0, c2, c4 are constants set based on the cutoff radius
// the cutoff is going to be rcutoff = cutoff_factor * dij
struct InversePowerStillingerQuadCutInteraction : BaseInteraction {
  const int m_pow;                // potential goes as r^{-m_pow}'
  const int m_pow_by_2;           // m_pow/2
  const double m_cutoff_factor;   // cutoff factor rcut = m_cutoff_factor * dij
  const double m_cutoff_factor2;  // cutoff factor2 = cutoff_factor^2
  const double c0;                // constant term in cutoff contribution
  const double c2;                // quadratic term in cutoff contribution
  const double c4;                // quartic term in cutoff contribution
  const double m_v0;              // overall energy scale for the full potential
  InversePowerStillingerQuadCutInteraction(const int pow, const double v0,
                                           const double cutoff_factor)
      : m_pow(pow),
        m_pow_by_2(pow / 2),
        m_cutoff_factor(cutoff_factor),
        m_cutoff_factor2(cutoff_factor * cutoff_factor),
        c0(-(1.0 / 8.0) * v0 * std::pow(m_cutoff_factor, -pow) *
           (8 + 6 * pow + pow * pow)),
        c2((1.0 / 4.0) * v0 * std::pow(m_cutoff_factor, -pow - 2) *
           (4 * pow + pow * pow)),
        c4(-(1.0 / 8.0) * v0 * std::pow(m_cutoff_factor, -pow - 4) *
           (pow * pow + 2 * pow)),
        m_v0(v0) {
    if (pow % 2 != 0) {
      // not implemented for odd powers
      throw std::runtime_error(
          "InversePowerStillingerQuadCutInteraction: only "
          "implemented for even powers");
    }
  }
  double energy(const double r2, const double cutoff) const {
    if (r2 > cutoff * cutoff) {
      return 0;
    }
    const double dij = cutoff / m_cutoff_factor;
    const double r2_scaled = r2 / (dij * dij);
    const double r4_scaled = r2_scaled * r2_scaled;
    const double r_pow_scaled = std::pow(r2_scaled, m_pow_by_2);

    return m_v0 / r_pow_scaled + c0 + c2 * r2_scaled + c4 * r4_scaled;
  }
  // calculate energy and gradient from distance squared, gradient is in
  // -(dv/drij)/|rij|
  double energy_gradient(double r2, double *gij, const double cutoff) const {
    if (r2 > cutoff * cutoff) {
      *gij = 0;
      return 0.;
    }
    const double dij = cutoff / m_cutoff_factor;
    const double dij_2 = dij * dij;
    const double r2_scaled = r2 / (dij_2);
    const double r4 = r2_scaled * r2_scaled;
    const double r_pow_scaled = std::pow(r2_scaled, m_pow_by_2);

    double e = m_v0 / r_pow_scaled + c0 + c2 * r2_scaled + c4 * r4;
    *gij = m_pow * m_v0 / (r_pow_scaled * r2) - 2 * c2 / dij_2 -
           4 * c4 * r2_scaled / (dij_2);
    return e;
  }
  double energy_gradient_hessian(const double r2, double *gij, double *hij,
                                 const double cutoff) const {
    if (r2 > cutoff * cutoff) {
      *gij = 0;
      *hij = 0;
      return 0.;
    }
    const double dij = cutoff / m_cutoff_factor;
    const double dij_2 = dij * dij;
    const double r2_scaled = r2 / (dij_2);
    const double r4_scaled = r2_scaled * r2_scaled;
    const double r_pow_scaled = std::pow(r2_scaled, m_pow_by_2);
    double e = m_v0 / r_pow_scaled + c0 + c2 * r2_scaled + c4 * r4_scaled;
    *gij = m_pow * m_v0 / (r_pow_scaled * r2) - 2 * c2 / dij_2 -
           4 * c4 * r2_scaled / (dij_2);
    *hij = m_pow * (m_pow + 1) * m_v0 / (r_pow_scaled * r2) + 2 * c2 / dij_2 +
           12 * c4 * r2_scaled / (dij_2);
    return e;
  }
};

/*
 * Templated potential for Stillinger-Weber potential with a cutoff.
 * This version is quicker because the power can be calculated faster when known
 * at compile time.
 */
template <int POW>
struct InversePowerStillingerQuadCutInteractionInt : BaseInteraction {
  const int m_pow;                // potential goes as r^{-m_pow}'
  const int m_pow_by_2;           // m_pow/2
  const double m_cutoff_factor;   // cutoff factor rcut = m_cutoff_factor * dij
  const double m_cutoff_factor2;  // cutoff factor2 = cutoff_factor^2
  const double c0;                // constant term in cutoff contribution
  const double c2;                // quadratic term in cutoff contribution
  const double c4;                // quartic term in cutoff contribution
  const double m_v0;              // overall energy scale for the full potential
  InversePowerStillingerQuadCutInteractionInt(double v0,
                                              const double cutoff_factor)
      : m_pow(POW),
        m_pow_by_2(POW / 2),
        m_cutoff_factor(cutoff_factor),
        m_cutoff_factor2(cutoff_factor * cutoff_factor),
        c0(-(1.0 / 8.0) * v0 * std::pow(m_cutoff_factor, -POW) *
           (8 + 6 * POW + POW * POW)),
        c2((1.0 / 4.0) * v0 * std::pow(m_cutoff_factor, -POW - 2) *
           (4 * POW + POW * POW)),
        c4(-(1.0 / 8.0) * v0 * std::pow(m_cutoff_factor, -POW - 4) *
           (POW * POW + 2 * POW)),
        m_v0(v0) {
    if (POW % 2 != 0) {
      // not implemented for odd powers
      throw std::runtime_error(
          "InversePowerStillingerQuadCutInteractionInt: only "
          "implemented for even powers");
    }
  }
  double energy(const double r2, const double cutoff) const {
    if (r2 > cutoff * cutoff) {
      return 0;
    }
    const double dij = cutoff / m_cutoff_factor;
    const double r2_scaled = r2 / (dij * dij);
    const double r4_scaled = r2_scaled * r2_scaled;
    const double r_pow_scaled = pos_int_pow<POW / 2>(r2_scaled);

    return m_v0 / r_pow_scaled + c0 + c2 * r2_scaled + c4 * r4_scaled;
  }
  // calculate energy and gradient from distance squared, gradient is in
  // -(dv/drij)/|rij|
  double energy_gradient(double r2, double *gij, const double cutoff) const {
    if (r2 > cutoff * cutoff) {
      *gij = 0;
      return 0.;
    }
    const double dij = cutoff / m_cutoff_factor;
    const double dij_2 = dij * dij;
    const double r2_scaled = r2 / (dij_2);
    const double r4 = r2_scaled * r2_scaled;
    const double r_pow_scaled = pos_int_pow<POW / 2>(r2_scaled);

    double e = m_v0 / r_pow_scaled + c0 + c2 * r2_scaled + c4 * r4;
    *gij = POW * m_v0 / (r_pow_scaled * r2) - 2 * c2 / dij_2 -
           4 * c4 * r2_scaled / (dij_2);
    return e;
  }
  double energy_gradient_hessian(const double r2, double *gij, double *hij,
                                 const double cutoff) const {
    if (r2 > cutoff * cutoff) {
      *gij = 0;
      *hij = 0;
      return 0.;
    }
    const double dij = cutoff / m_cutoff_factor;
    const double dij_2 = dij * dij;
    const double r2_scaled = r2 / (dij_2);
    const double r4_scaled = r2_scaled * r2_scaled;
    const double r_pow_scaled = pos_int_pow<POW / 2>(r2_scaled);
    double e = m_v0 / r_pow_scaled + c0 + c2 * r2_scaled + c4 * r4_scaled;
    *gij = POW * m_v0 / (r_pow_scaled * r2) - 2 * c2 / dij_2 -
           4 * c4 * r2_scaled / (dij_2);
    *hij = POW * (POW + 1) * m_v0 / (r_pow_scaled * r2) + 2 * c2 / dij_2 +
           12 * c4 * r2_scaled / (dij_2);
    return e;
  }
};

template <size_t ndim>
class InversePowerStillingerCutQuad
    : public SimplePairwisePotential<InversePowerStillingerQuadCutInteraction,
                                     cartesian_distance<ndim>> {
 public:
  InversePowerStillingerCutQuad(const int pow, const double v0,
                                const double cutoff_factor,
                                const pele::Array<double> radii,
                                double non_additivity = 0.0)
      : SimplePairwisePotential<InversePowerStillingerQuadCutInteraction,
                                cartesian_distance<ndim>>(
            std::make_shared<InversePowerStillingerQuadCutInteraction>(
                pow, v0, cutoff_factor),
            radii, NonAdditiveCutoffCalculator(non_additivity, cutoff_factor),
            std::make_shared<cartesian_distance<ndim>>(), 0.0, false) {}
};

template <size_t ndim>
class InversePowerStillingerCutQuadPeriodic
    : public SimplePairwisePotential<InversePowerStillingerQuadCutInteraction,
                                     periodic_distance<ndim>> {
 public:
  InversePowerStillingerCutQuadPeriodic(const int pow, const double v0,
                                        const double cutoff_factor,
                                        const pele::Array<double> radii,
                                        const pele::Array<double> boxvec,
                                        double non_additivity = 0.0)
      : SimplePairwisePotential<InversePowerStillingerQuadCutInteraction,
                                periodic_distance<ndim>>(
            std::make_shared<InversePowerStillingerQuadCutInteraction>(
                pow, v0, cutoff_factor),
            radii, NonAdditiveCutoffCalculator(non_additivity, cutoff_factor),
            std::make_shared<periodic_distance<ndim>>(boxvec), 0.0, false) {}
};

template <size_t ndim>
class InversePowerStillingerCutQuadPeriodicCellLists
    : public CellListPotential<InversePowerStillingerQuadCutInteraction,
                               periodic_distance<ndim>> {
 public:
  InversePowerStillingerCutQuadPeriodicCellLists(
      const int pow, const double v0, const double cutoff_factor,
      const pele::Array<double> radii, const pele::Array<double> boxvec,
      double ncellx_scale, double non_additivity = 0.0)
      : CellListPotential<InversePowerStillingerQuadCutInteraction,
                          periodic_distance<ndim>>(
            std::make_shared<InversePowerStillingerQuadCutInteraction>(
                pow, v0, cutoff_factor),
            std::make_shared<periodic_distance<ndim>>(boxvec), boxvec,
            ncellx_scale, radii,
            NonAdditiveCutoffCalculator(non_additivity, cutoff_factor), 0.0,
            true, false) {}
};

template <size_t ndim, size_t POW>
class InversePowerStillingerCutQuadInt
    : public SimplePairwisePotential<
          InversePowerStillingerQuadCutInteractionInt<POW>,
          cartesian_distance<ndim>> {
 public:
  InversePowerStillingerCutQuadInt(const double v0, const double cutoff_factor,
                                   const pele::Array<double> radii,
                                   double non_additivity = 0.0)
      : SimplePairwisePotential<
            InversePowerStillingerQuadCutInteractionInt<POW>,
            cartesian_distance<ndim>>(
            std::make_shared<InversePowerStillingerQuadCutInteractionInt<POW>>(
                v0, cutoff_factor),
            radii, NonAdditiveCutoffCalculator(non_additivity, cutoff_factor),
            std::make_shared<cartesian_distance<ndim>>(), 0.0, false) {}
};

template <size_t ndim, size_t POW>
class InversePowerStillingerCutQuadIntPeriodic
    : public SimplePairwisePotential<
          InversePowerStillingerQuadCutInteractionInt<POW>,
          periodic_distance<ndim>> {
 public:
  InversePowerStillingerCutQuadIntPeriodic(const double v0,
                                           const double cutoff_factor,
                                           const pele::Array<double> radii,
                                           const pele::Array<double> boxvec,
                                           double non_additivity = 0.0)
      : SimplePairwisePotential<
            InversePowerStillingerQuadCutInteractionInt<POW>,
            periodic_distance<ndim>>(
            std::make_shared<InversePowerStillingerQuadCutInteractionInt<POW>>(
                v0, cutoff_factor),
            radii, NonAdditiveCutoffCalculator(non_additivity, cutoff_factor),
            std::make_shared<periodic_distance<ndim>>(boxvec), 0.0, false) {}
};

template <size_t ndim, size_t POW>
class InversePowerStillingerCutQuadIntPeriodicCellLists
    : public CellListPotential<InversePowerStillingerQuadCutInteractionInt<POW>,
                               periodic_distance<ndim>> {
 public:
  InversePowerStillingerCutQuadIntPeriodicCellLists(
      const double v0, const double cutoff_factor,
      const pele::Array<double> radii, const pele::Array<double> boxvec,
      double ncellx_scale, double non_additivity = 0.0)
      : CellListPotential<InversePowerStillingerQuadCutInteractionInt<POW>,
                          periodic_distance<ndim>>(
            std::make_shared<InversePowerStillingerQuadCutInteractionInt<POW>>(
                v0, cutoff_factor),
            std::make_shared<periodic_distance<ndim>>(boxvec), boxvec,
            ncellx_scale, radii,
            NonAdditiveCutoffCalculator(non_additivity, cutoff_factor), 0.0,
            true, false) {}
};

}  // namespace pele

#endif  // !INVERSE_POWER_STILINGER_CUT_QUAD_H
