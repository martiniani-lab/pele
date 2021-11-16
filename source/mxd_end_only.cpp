/**
 * @file mxd_end_only.cpp
 * @author Praharsh Suryadevara
 * @brief
 * @version 0.1
 * @date 2021-11-05
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "pele/mxd_end_only.hpp"
#include "pele/array.hpp"
#include "pele/newton.hpp"
#include "pele/optimizer.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <memory>

namespace pele {

MixedDescentEndOnly::MixedDescentEndOnly(
    std::shared_ptr<BasePotential> potential, const pele::Array<double> x0,
    double tol, double newton_step_tol, double rtol, double atol,
    double threshold, bool iterative)
    : GradientOptimizer(potential, x0, tol),
      _cvode_optimizer(potential, x0, tol, rtol, atol,
                        iterative, false), // initialize the CVODE optimizer
      _newton_optimizer(potential, x0, tol,
                        threshold, true), // initialize the Newton optimizer with the rattler mask option passed as true
      _tol(tol), _newton_step_tol(newton_step_tol), use_newton_step(false),
    
      particle_disp(potential->get_ndim()) {}

void MixedDescentEndOnly::one_iteration() {
  if (use_newton_step) {
    _newton_optimizer.one_iteration();
  } else {
    _cvode_optimizer.one_iteration();
    Array<double> gradient = _cvode_optimizer.get_g();
    // use standard stopping criteria to figure out when to refine the step
    rms_ = norm(gradient) / sqrt(gradient.size());

    std::cout << "rms: " << rms_ << std::endl;
    if (rms_ < _tol) {
      use_newton_step = true;
      _newton_optimizer.set_x_and_find_rattlers(_cvode_optimizer.get_x());
    }
  }
  iter_number_++;
}



// The stop criteria refines the minima so that the we're closer to the minima.
bool MixedDescentEndOnly::stop_criterion_satisfied() {
  // stop criterion can only be satisfied in the Newton regime
  //std::cout << _newton_optimizer.get_niter() << "niter" << std::endl;
  bool jammed;

  jammed = _newton_optimizer.is_jammed();

  // TODO: Pass these on to the python interface since we don't want to keep do this twice
  if (use_newton_step & !jammed) {
    return true; // assuming that the find rattlers function in julia will take care of this
  }
  if (use_newton_step && _newton_optimizer.get_niter() > 0) {
    //std::cout << "are we actually here" << std::endl;
    double _ndim = potential_->get_ndim();
    int nparticles = _cvode_optimizer.get_x().size() / _ndim;
    Eigen::VectorXd _step = _newton_optimizer.get_step();


    particle_disp.assign(0.0);
    double max_disp = 0.0;

    for (size_t particle_index = 0; particle_index < nparticles;
         particle_index++) {
      for (size_t dim = 0; dim < _ndim; dim++) {
          particle_disp[dim] =
              _step(particle_index * _ndim + dim);
      }
    double disp = norm(particle_disp);
      if (disp > max_disp) {
        max_disp = disp;
      }
    }
    if (max_disp < _newton_step_tol) {
        //std::cout << "max_disp: " << max_disp << std::endl;
      return true;
    }
    else {
      return false;
    }

  } else {
    return false;
  }
  //std::cout << "should not be here" << std::endl;
}

} // namespace pele