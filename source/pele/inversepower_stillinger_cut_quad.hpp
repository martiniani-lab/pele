//
//

#ifndef INVERSE_POWER_STILINGER_CUT_QUAD_H
#define INVERSE_POWER_STILINGER_CUT_QUAD_H

#include "base_interaction.hpp"
#include "pele/cell_list_potential.hpp"
#include "pele/simple_pairwise_potential.hpp"
#include <cmath>
#include <cstddef>
#include <stdexcept>

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

  const size_t m_pow;            // potential goes as r^{-m_pow}'
  const size_t m_pow_by_2;       // m_pow/2
  const double m_cutoff_factor;  // cutoff factor rcut = m_cutoff_factor * dij
  const double m_cutoff_factor2; // cutoff factor2 = cutoff_factor^2
  const double c0;               // constant term in cutoff contribution
  const double c2;               // quadratic term in cutoff contribution
  const double c4;               // quartic term in cutoff contribution
  const double m_v0;             // overall energy scale for the full potential
  InversePowerStillingerQuadCutInteraction(const size_t pow, const double v0,
                                           const double cutoff_factor,
                                           const double v0,
                                           const double cutoff_factor)
      : m_pow(pow), m_pow_by_2(pow / 2), m_v0(v0),
        m_cutoff_factor(cutoff_factor),
        m_cutoff_factor2(cutoff_factor * cutoff_factor),
        c0((1.0 / 8.0) * std::pow(m_cutoff_factor, -pow) *
           (8 + 6 * pow + pow * pow)),
        c2((1.0 / 4.0) * std::pow(m_cutoff_factor, -pow - 2) *
           (4 * pow + pow * pow)),
        c4(-(1.0 / 8.0) * std::pow(m_cutoff_factor, -pow - 4) *
           (pow * pow + 2 * pow)) {
    if (pow % 2 != 0) {
      // not implemented for odd powers
      throw std::runtime_error("InversePowerStillingerQuadCutInteraction: only "
                               "implemented for even powers");
    }
  }
  double energy(double r2, const double dij) const {
    if (r2 > dij * dij * m_cutoff_factor2) {
      return 0;
    }
    const double r4 = r2 * r2;
    const double r_pow = std::pow(r2, m_pow_by_2);

    return m_v0 / r_pow + c0 + c2 * r2 + c4 * r4;
  }
  // calculate energy and gradient from distance squared, gradient is in
  // -(dv/drij)/|rij|
  double energy_gradient(double r2, double *gij, const double dij) const {
    if (r2 > dij * dij * m_cutoff_factor2) {
      *gij = 0;
      return 0.;
    }
    const double r4 = r2 * r2;
    const double r_pow = std::pow(r2, m_pow_by_2);

    double e = m_v0 / r_pow + c0 + c2 * r2 + c4 * r4;
    *gij = -m_pow * m_v0 / r_pow / r2 + 2 * c2 + 4 * c4 * r2;
    return e;
  }
  double energy_gradient_hessian(double r2, double *gij, double *hij,
                                 const double dij) const {
    if (r2 > dij * dij * m_cutoff_factor2) {
      *gij = 0;
      *hij = 0;
      return 0.;
    }
    const double r4 = r2 * r2;
    const double r_pow = std::pow(r2, m_pow_by_2);

    double e = m_v0 / r_pow + c0 + c2 * r2 + c4 * r4;
    *gij = -m_pow * m_v0 / r_pow / r2 + 2 * c2 + 4 * c4 * r2;
    *hij = m_pow * (m_pow - 1) * m_v0 / r_pow / r4 - 2 * c2 - 12 * c4 * r2;
    return e;
  }
};

template <size_t ndim>
class InversePowerStillingerCutQuad
    : public SimplePairwisePotential<InversePowerStillingerQuadCutInteraction,
                                     cartesian_distance<ndim>> {
public:
  InversePowerStillingerCutQuad(const size_t pow, const double v0,
                                const double cutoff_factor,
                                const pele::Array<double> radii)
      : SimplePairwisePotential<InversePowerStillingerQuadCutInteraction,
                                cartesian_distance<ndim>>(
            std::make_shared<InversePowerStillingerQuadCutInteraction>(
                pow, v0, cutoff_factor),
            radii, std::make_shared<cartesian_distance<ndim>>()) {}
};

template <size_t ndim>
class InversePowerStillingerCutQuadPeriodic
    : public SimplePairwisePotential<InversePowerStillingerQuadCutInteraction,
                                     periodic_distance<ndim>> {
public:
  InversePowerStillingerCutQuadPeriodic(const size_t pow, const double v0,
                                        const double cutoff_factor,
                                        const pele::Array<double> radii,
                                        const pele::Array<double> boxvec)
      : SimplePairwisePotential<InversePowerStillingerQuadCutInteraction,
                                periodic_distance<ndim>>(
            std::make_shared<InversePowerStillingerQuadCutInteraction>(
                pow, v0, cutoff_factor),
            radii, std::make_shared<periodic_distance<ndim>>(boxvec)) {}
};

template <size_t ndim>
class InversePowerStillingerCutQuadPeriodicCellLists
    : public CellListPotential<InversePowerStillingerQuadCutInteraction,
                               periodic_distance<ndim>> {
public:
  InversePowerStillingerCutQuadPeriodicCellLists(const size_t pow, const double v0, const double cutoff_factor, 
                                             const pele::Array<double> radii,
                                             const pele::Array<double> boxvec,
                                             double ncellx_scale)
      : CellListPotential<InversePowerStillingerQuadCutInteraction,
                          periodic_distance<ndim>>(
            std::make_shared<InversePowerStillingerQuadCutInteraction>(pow,
                                                                       v0, cutoff_factor),
            std::make_shared<periodic_distance<ndim>>(boxvec), boxvec, 2.0 * cutoff_factor *
                (*std::max_element(radii.begin(), radii.end()),
            ncellx_scale, radii) {
            }
};

} // namespace pele

#endif // !INVERSE_POWER_STILINGER_CUT_QUAD_H
