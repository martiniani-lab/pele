/**
 * @file mxd_end_only.hpp
 * @author Praharsh Suryadevara (praharsharmm@gmail.com)
 * @brief Mixed descent that switches to newton only at the end of the
 * optimization.
 * @version 0.1
 * @date 2021-11-05
 *
 * @copyright Copyright (c) 2021
 *
 */

#ifndef _PELE_MXD_END_ONLY_HPP
#define _PELE_MXD_END_ONLY_HPP

#include "base_potential.hpp"
#include "pele/array.hpp"
#include "pele/cvode.hpp"
#include "pele/newton.hpp"
#include <complex>
#include <cstdint>
#include <iostream>

namespace pele {

class MixedDescentEndOnly : public GradientOptimizer {
private:
  double _tol;
  double _newton_step_tol;
  bool use_newton_step;
  Array<double> particle_disp;  // Stores displacement for a single particle
  double _max_dist;             // Maximum distance a single particle has moved
  Array<uint8_t> _not_rattlers; // TODO: only because we need to save this

  CVODEBDFOptimizer _cvode_optimizer;
  Newton _newton_optimizer;

public:
  MixedDescentEndOnly(std::shared_ptr<BasePotential> potential,
                      const pele::Array<double> x0, double tol = 1e-6,
                      double newton_step_tol = 1e-6, double rtol = 1e-6,
                      double atol = 1e-6, double threshold = 1e-10,
                      bool iterative = false);
  ~MixedDescentEndOnly(){};

  void one_iteration();
  // since most of the calculations are done in the attached optimizers, we
  // defer the calculations to them
  inline int get_nhev() {
    std::cout << "cvode nhev" << _cvode_optimizer.get_nhev() << std::endl;
    std::cout << "newton nhev" << _newton_optimizer.get_nhev() << std::endl;
    return _cvode_optimizer.get_nhev() + _newton_optimizer.get_nhev();
  }
  inline int get_niter() { return iter_number_; }
  inline double get_nfev() {
    return _cvode_optimizer.get_nfev() + _newton_optimizer.get_nfev();
  }
  bool stop_criterion_satisfied();
  /**
   * @brief Get the step object in pele Array form
   *
   * @return Array<double>
   */
  inline Array<double> get_step_vec() {
    Eigen::VectorXd step = _newton_optimizer.get_step();
    return Array<double>(step.data(), step.size());
  }
};
} // namespace pele

#endif // _PELE_MXD_END_ONLY_HPP
