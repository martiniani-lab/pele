#ifndef _PELE_INVERSEPOWER_STILLIGNER_H
#define _PELE_INVERSEPOWER_STILLIGNER_H

#include "base_interaction.hpp"
#include "pele/simple_pairwise_potential.hpp"

namespace pele {

struct InversePowerStillinger_interaction : BaseInteraction {
  // Inverse power law potential, see JCP 83, 4767 (1984),
  // http://dx.doi.org/10.1063/1.449840
  const double m_pow;
  InversePowerStillinger_interaction(const size_t pow) : m_pow(pow) {}
  double energy(double r2, const double dij) const {
    const double E = power_inp2((dij * dij) / r2, m_pow);
    return E;
  }
  // calculate energy and gradient from distance squared, gradient is in
  // -(dv/drij)/|rij|
  double energy_gradient(double r2, double *gij, const double dij) const {
    const double E = power_inp2((dij * dij) / r2, m_pow);
    *gij = m_pow * E / r2;
    return E;
  }
  double energy_gradient_hessian(double r2, double *gij, double *hij,
                                 const double dij) const {
    const double E = power_inp2((dij * dij) / r2, m_pow);
    *gij = m_pow * E / r2;
    *hij = *gij * (m_pow + 1);
    return E;
  }

  // Compute inp ** n.
  // See Skiena, p. 48.
  template <class T>
  T power(const T inp, const size_t n) const {
    if (n == 0) {
      return 1;
    }
    const T x = power(inp, n / 2);
    if (n % 2) {
      return inp * x * x;
    } else {
      return x * x;
    }
  }
  // Compute inp ** n, given inp ** 2.
  template <class T>
  T power_inp2(const T inp2, const size_t n) const {
    const T x = power(inp2, n / 2);
    if (n % 2) {
      return std::sqrt(inp2) * x;
    } else {
      return x;
    }
  }
};

template <size_t ndim>
class InversePowerStillinger
    : public SimplePairwisePotential<InversePowerStillinger_interaction,
                                     cartesian_distance<ndim>> {
 public:
  InversePowerStillinger(const size_t pow, const pele::Array<double> radii)
      : SimplePairwisePotential<InversePowerStillinger_interaction,
                                cartesian_distance<ndim>>(
            std::make_shared<InversePowerStillinger_interaction>(pow), radii,
            std::make_shared<cartesian_distance<ndim>>()) {}
};

template <size_t ndim>
class InversePowerStillingerPeriodic
    : public SimplePairwisePotential<InversePowerStillinger_interaction,
                                     periodic_distance<ndim>> {
 public:
  InversePowerStillingerPeriodic(const size_t pow,
                                 const pele::Array<double> radii,
                                 const pele::Array<double> boxvec)
      : SimplePairwisePotential<InversePowerStillinger_interaction,
                                periodic_distance<ndim>>(
            std::make_shared<InversePowerStillinger_interaction>(pow), radii,
            std::make_shared<periodic_distance<ndim>>(boxvec)) {}
};

}  // namespace pele

#endif  // #ifndef _PELE_INVERSEPOWER_STILLIGNER_H
