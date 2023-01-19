#ifndef _ATLJ_H
#define _ATLJ_H
/**
 * @brief      PELE IMPLEMENTATION OF ATLJ POTENTIAL IN C++
 *
 *
 *
 *

 *
 *
 */
#include <memory>
#include <vector>

#include "array.hpp"
#include "base_potential.hpp"
#include "lj.hpp"
extern "C" {
#include "xsum.h"
}
#include <Eigen/Core>

// using namespace Eigen;

namespace pele {

class ATLJ : public BasePotential {
 private:
  double m_eps;
  double m_sig;
  double m_Z;
  double m_ndim;
  double m_natoms;
  LJ ljpot;
  double get_axilrod_teller_energy(Array<double> const &x);

 public:
  ATLJ(double sig, double eps, double Z);
  virtual ~ATLJ() {}
  double get_energy(Array<double> const &x);
};
}  // namespace pele

#endif  // end header