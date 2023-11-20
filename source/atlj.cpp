
#include "pele/atlj.hpp"
#include <array>
#include <iostream>
#include <limits>
#include <math.h>
#include <memory>

// Warning the following files were written to generate basins for ATLJ from
// LBFGS using an exact gradient for a three body potential in 3 dimensions.
// This needs to be expanded on for more than three atoms / different dimensions
// whenever it gets used for more
// A rewrite would be in order for potentials elsewhere;
// The AT gradient is only calculated exactly, the energy isn't
// so you might need to rewrite

// Make sure you rewrite the copy constructor for this thing

namespace pele {

ATLJ::ATLJ(double sig, double eps, double Z)
    : m_eps(eps),
      m_sig(sig),
      m_Z(Z),
      m_ndim(3),
      m_natoms(3),
      ljpot(4 * eps * pow(sig, 6), 4 * eps * pow(sig, 12))

{}

double ATLJ::get_energy(const Array<double> &x) {
  double lj_energy = ljpot.get_energy(x);
  double at_energy = get_axilrod_teller_energy(x);
  return lj_energy + at_energy;
};

double ATLJ::get_axilrod_teller_energy(const Array<double> &x) {
  double energy = 0;
  Array<double> drij(m_natoms);
  Array<double> drjk(m_natoms);
  Array<double> drki(m_natoms);

  // Rewrite into a three pairwise loop for more atoms
  for (size_t i = 0; i < m_ndim; ++i) {
    drij[i] = x[i] - x[i + m_natoms];
    drjk[i] = x[i + m_natoms] - x[i + 2 * m_natoms];
    drki[i] = x[i + 2 * m_natoms] - x[i];
  }
  double rij = norm(drij);
  double rjk = norm(drjk);
  double rki = norm(drki);
  // note that this is slightly differently written, the - sign is taken outside
  double costhetaprod =
      -(3 * dot(drij, drjk) * dot(drjk, drki) * dot(drki, drij)) /
      pow((rij * rjk * rki), 2);
  energy += m_Z * (1 + costhetaprod) / (pow((rij * rjk * rki), 3));
  return energy;
}

}  // namespace pele