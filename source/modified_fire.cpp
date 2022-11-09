#include "pele/modified_fire.hpp"
#include <pele/optimizer.hpp>

using std::cout;
using std::vector;

namespace pele {
/**
 * An implementation of the *modified* FIRE optimization algorithm in c++.
 *
 * The Fast Inertial Relaxation Engine is an optimization algorithm based
 * on molecular dynamics with modifications to the velocity and adaptive
 * time steps. The method is based on a blind skier searching for the
 * bottom of a valley and is described and tested here:
 *
 * Erik Bitzek, Pekka Koskinen, Franz Gaehler, Michael Moseler, and Peter
 * Gumbsch. Phys. Rev. Lett. 97, 170201 (2006)
 * http://link.aps.org/doi/10.1103/PhysRevLett.97.170201
 *
 * This implementation of the algorithm differs significantly from the original
 * algorithm in the order in which the steps are taken. Here we do the
 * following:
 *
 * -> initialise the velocities and gradients for some given coordinates
 * -> set the velocity to the fire *inertial* velocity
 *
 * Only then we use the MD integrator that here does things in this order:
 *
 * -> compute the velocity difference and add it to the current velocity
 * -> compute the new position given this velocity
 * -> recompute gradient and energy
 *
 * Once the integrator is done we continue following FIRE and compute
 *
 * P = -g * f
 *
 * here comes the other modification to the algorithm: if stepback is set to
 * false then proceed just as the original FIRE algorithm prescribes, if
 * stepback is set to true (default) then whenever P<=0 we undo the last step
 * besides carrying out the operations defined by the original FIRE algorithm.
 *
 */

MODIFIED_FIRE::MODIFIED_FIRE(std::shared_ptr<pele::BasePotential> potential,
                             pele::Array<double> &x0, double dtstart,
                             double dtmax, double maxstep, size_t Nmin,
                             double finc, double fdec, double fa, double astart,
                             double tol, bool stepback)
    : ODEBasedOptimizer(potential, x0,
                        tol), // call GradientOptimizer constructor
      _dtstart(dtstart), _dt(dtstart), _dtmax(dtmax), _maxstep(maxstep),
      _Nmin(Nmin), _finc(finc), _fdec(fdec), _fa(fa), _astart(astart),
      _a(astart), _fold(f_), _ifnorm(0), _vnorm(0), _v(x0.size(), 0),
      _dx(x0.size()), _xold(x0.copy()), _gold(g_.copy()), _fire_iter_number(0),
      _N(x_.size()), _stepback(stepback) {
#if PRINT_TO_FILE == 1
  trajectory_file.open("trajectory_modified_fire.txt");
  time_file.open("time_modified_fire.txt");
  gradient_file.open("gradient_modified_fire.txt");
#endif
#if OPTIMIZER_DEBUG_LEVEL > 0
  std::cout << "MODIFIED_FIRE Constructed with parameters:\n";
  std::cout << "dtstart: " << _dtstart << "\n";
  std::cout << "dtmax: " << _dtmax << "\n";
  std::cout << "maxstep: " << _maxstep << "\n";
  std::cout << "Nmin: " << _Nmin << "\n";
  std::cout << "finc: " << _finc << "\n";
  std::cout << "fdec: " << _fdec << "\n";
  std::cout << "fa: " << _fa << "\n";
  std::cout << "astart: " << _astart << "\n";
  std::cout << "tol: " << tol << "\n";
  std::cout << "stepback: " << _stepback << "\n";
#endif
}

/**
 * Overload initialize_func_gradient from parent class
 */

void MODIFIED_FIRE::initialize_func_gradient() {
  nfev_ += 1; // this accounts for the energy evaluation done by the integrator
  f_ = potential_->get_energy_gradient(x_, g_);
  _fold = f_;
  for (size_t k = 0; k < x_.size();
       ++k) { // set initial velocities (using forward Euler)
    _v[k] = -g_[k] * _dt;
  }
  _ifnorm = 1. / norm(g_);
  _vnorm = norm(_v);
  gradient_norm_ = 1. / (_ifnorm * sqrt(_N));
  func_initialized_ = true;
}

void MODIFIED_FIRE::set_func_gradient(double f, Array<double> grad) {
  if (grad.size() != g_.size()) {
    throw std::invalid_argument("the gradient has the wrong size");
  }
  if (iter_number_ > 0) {
    cout << "warning: setting f and grad after the first iteration.  this is "
            "dangerous.\n";
  }

  // copy the function and gradient
  f_ = f;
  _fold = f_;
  g_.assign(grad);
  for (size_t k = 0; k < x_.size();
       ++k) { // set initial velocities (using forward Euler)
    _v[k] = -g_[k] * _dt;
  }
  _ifnorm = 1. / norm(g_);
  _vnorm = norm(_v);
  gradient_norm_ = 1. / (_ifnorm * sqrt(_N));
  func_initialized_ = true;
}
} // namespace pele
