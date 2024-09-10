#ifndef _PELE_COMBINE_POTENTIALS_H_
#define _PELE_COMBINE_POTENTIALS_H_
#include <cstddef>
#include <list>
#include <memory>

#include "base_potential.hpp"

namespace pele {

/**
 * Potential wrapper which wraps multiple potentials to
 * act as one potential.  This can be used to implement
 * multiple types of interactions, e.g. a system with two
 * types atoms
 */
class CombinedPotential : public BasePotential {
 protected:
  std::list<std::shared_ptr<BasePotential>> _potentials;

 public:
  CombinedPotential() {}

  /**
   * destructor: destroy all the potentials in the list
   */
  virtual ~CombinedPotential() {}

  /**
   * add a potential to the list
   */
  virtual void add_potential(std::shared_ptr<BasePotential> potential) {
    _potentials.push_back(potential);
  }

  virtual double get_energy(Array<double> const &x) {
    double energy = 0.;
    for (auto &pot_ptr : _potentials) {
      energy += pot_ptr->get_energy(x);
    }
    return energy;
  }

  virtual double get_energy_gradient(Array<double> const &x,
                                     Array<double> &grad) {
    if (x.size() != grad.size()) {
      throw std::invalid_argument("the gradient has the wrong size");
    }

    double energy = 0.;
    grad.assign(0.);

    for (auto &pot_ptr : _potentials) {
      energy += pot_ptr->add_energy_gradient(x, grad);
    }
    return energy;
  }

  virtual double get_energy_gradient_hessian(Array<double> const &x,
                                             Array<double> &grad,
                                             Array<double> &hess) {
    if (x.size() != grad.size()) {
      throw std::invalid_argument("the gradient has the wrong size");
    }
    if (hess.size() != x.size() * x.size()) {
      throw std::invalid_argument("the Hessian has the wrong size");
    }

    double energy = 0.;
    grad.assign(0.);
    hess.assign(0.);

    for (auto &pot_ptr : _potentials) {
      energy += pot_ptr->add_energy_gradient_hessian(x, grad, hess);
    }
    return energy;
  }
  virtual void get_hessian(Array<double> const &x, Array<double> &hessian) {
    if (hessian.size() != x.size() * x.size()) {
      throw std::invalid_argument("the Hessian has the wrong size");
    }
    hessian.assign(0.);
    for (auto &pot_ptr : _potentials) {
      pot_ptr->add_hessian(x, hessian);
    }
  }
};

/**
 * @brief A wrapper for a combined potential that can switch from one potential
 * to the sum of that potential and another. Useful for extending the reach of
 * the potential.
 * @details This class is useful for extending the reach of the potential.
 * For example if you want to condition the newton's method in flat regions,
 * this potential should allow you to do so by adding a new potential to the sum
 * of the old potential and the new potential.
 */
class ExtendedPotential : public CombinedPotential {
 private:
  bool use_extended_potential_;
  size_t nhev_extension;
  size_t nfev_extension;
  size_t neev_extension;

 public:
  /**
   * @brief Construct a new Extended Potential object
   *
   * @param base_potential The potential to extend
   * @param extended_potential The potential to add to the base potential
   */
  ExtendedPotential(std::shared_ptr<BasePotential> main_potential,
                    std::shared_ptr<BasePotential> extension_potential)
      : CombinedPotential() {
    add_potential(main_potential);
    if (extension_potential != nullptr) {
      add_potential(extension_potential);
    }
    use_extended_potential_ = false;
  };

  void switch_on_extended_potential() { use_extended_potential_ = true; }

  void switch_off_extended_potential() { use_extended_potential_ = false; }

  /**
   * @brief get whether the extended potential is being used
   */
  bool get_use_extended_potential() { return use_extended_potential_; }

  /**
   * @brief get the energy of the extended potential
   * @details If the extended potential is switched on, the energy of the
   * extended potential is returned.
   * @param x The coordinates
   * @return double The energy
   */
  double get_energy(Array<double> const &x) {
    if (!use_extended_potential_) {
      return _potentials.front()->get_energy(x);
    } else {
      neev_extension += 1;
      return CombinedPotential::get_energy(x);
    }
  }

  double get_energy_gradient(Array<double> const &x, Array<double> &grad) {
    if (!use_extended_potential_) {
      return _potentials.front()->get_energy_gradient(x, grad);
    } else {
      nfev_extension += 1;
      return CombinedPotential::get_energy_gradient(x, grad);
    }
  }

  double get_energy_gradient_hessian(Array<double> const &x,
                                     Array<double> &grad, Array<double> &hess) {
    if (!use_extended_potential_) {
      return _potentials.front()->get_energy_gradient_hessian(x, grad, hess);
    } else {
      nhev_extension += 1;
      return CombinedPotential::get_energy_gradient_hessian(x, grad, hess);
    }
  }

  void get_hessian(Array<double> const &x, Array<double> &hess) {
    if (!use_extended_potential_) {
      return _potentials.front()->get_hessian(x, hess);
    } else {
      nhev_extension += 1;
      return CombinedPotential::get_hessian(x, hess);
    }
  }

  /**
   * @brief Get the hessian of the extended potential independent of the flag
   * @details This helps with convexity checks.
   * @param x
   * @param hess
   */
  void get_hessian_extended(Array<double> const &x, Array<double> &hess) {
    nhev_extension += 1;
    return CombinedPotential::get_hessian(x, hess);
  }

  size_t get_nfev_extension() { return nfev_extension; }
  size_t get_neev_extension() { return neev_extension; }
  size_t get_nhev_extension() { return nhev_extension; }
};  // ExtendedPotential

/**
 * Potential wrapper which negates a potential.
 */
class NegatedPotential final : public BasePotential {
 protected:
  std::shared_ptr<BasePotential> _potential;

 public:
  explicit NegatedPotential(std::shared_ptr<BasePotential> potential)
      : _potential(std::move(potential)) {}

  /**
   * destructor: destroy all the potentials in the list
   */
  ~NegatedPotential() override {}

  double get_energy(Array<double> const &x) override {
    return -_potential->get_energy(x);
  }

  double get_energy_gradient(Array<double> const &x,
                             Array<double> &grad) override {
    const double energy = _potential->get_energy_gradient(x, grad);
    grad *= -1.0;
    return -energy;
  }

  double get_energy_gradient_hessian(Array<double> const &x,
                                     Array<double> &grad,
                                     Array<double> &hess) override {
    const double energy =
        _potential->get_energy_gradient_hessian(x, grad, hess);
    grad *= -1.0;
    hess *= -1.0;
    return -energy;
  }

  void get_hessian(Array<double> const &x, Array<double> &hessian) override {
    _potential->get_hessian(x, hessian);
    hessian *= -1.0;
  }
};

}  // namespace pele

#endif
