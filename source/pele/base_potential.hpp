#ifndef PYGMIN_POTENTIAL_H
#define PYGMIN_POTENTIAL_H

#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "array.hpp"

extern "C" {
#include "xsum.h"
}

namespace pele {

/***
 * basic potential interface for native potentials
 */
class BasePotential {
 public:
  virtual ~BasePotential() {}

  /**
   * Return the energy of configuration x.  This is the only function which
   * must be overloaded
   */
  virtual double get_energy(Array<double> const &x) {
    throw std::runtime_error("BasePotential::get_energy must be overloaded");
  }

  /**
   * Return the change in energy when configuration A is changed to
   * configuration B.  It also takes in the vector of indices of atoms changed
   * to recalculate energies changed by those atoms.
   */
  virtual double get_energy_change(Array<double> const &old_coords,
                                   Array<double> const &new_coords,
                                   std::vector<size_t> const &) {
    return get_energy(new_coords) - get_energy(old_coords);
  }

  virtual size_t get_ndim() const {
    throw std::runtime_error("BasePotential::get_ndim must be overloaded");
  }

  virtual bool find_rattlers(pele::Array<double> const &minimum_coords,
                             Array<uint8_t> not_rattlers, bool &jammed) {
    throw std::runtime_error("BasePotential::find_rattlers must be overloaded");
  }
  /**
   * compute the energy and gradient, but don't initialize the gradient to zero
   */
  virtual double add_energy_gradient(Array<double> const &x,
                                     Array<double> &grad) {
    throw std::runtime_error(
        "BasePotential::add_energy_gradient must be overloaded");
  }

  /**
   * compute the energy and gradient.
   *
   * If not overloaded it will compute the numerical gradient
   */
  virtual double get_energy_gradient(Array<double> const &x,
                                     Array<double> &grad) {
    std::cout << "BasePotential::get_energy_gradient must be overloaded"
              << std::endl;
    double energy = get_energy(x);
    numerical_gradient(x, grad);
    return energy;
  }

  virtual double get_energy_gradient(
      Array<double> const &x, std::vector<xsum_small_accumulator> &exact_grad) {
    throw std::runtime_error(
        "A numerical version for the exact gradient has not been written");
  }

  /**
   * compute the energy, gradient, and Hessian, but don't initialize the
   * gradient or hessian to zero
   */
  virtual double add_energy_gradient_hessian(Array<double> const &x,
                                             Array<double> &grad,
                                             Array<double> &hess) {
    throw std::runtime_error(
        "BasePotential::add_energy_gradient_hessian must be overloaded");
  }

  virtual void add_hessian(Array<double> const &x, Array<double> &hess) {
    throw std::runtime_error("BasePotential::add_hessian must be overloaded");
  }
  /**
   * compute the energy and gradient and Hessian.
   *
   * If not overloaded it will compute the Hessian numerically and use
   * get_energy_gradient to get the energy and gradient.
   */
  virtual double get_energy_gradient_hessian(Array<double> const &x,
                                             Array<double> &grad,
                                             Array<double> &hess) {
    double energy = get_energy_gradient(x, grad);
    numerical_hessian(x, hess);
    return energy;
  }

  virtual double get_energy_gradient_hessian_rattlers(
      Array<double> const &x, Array<double> &grad, Array<double> &hess,
      Array<uint8_t> not_rattlers) {
    throw std::runtime_error(
        "BasePotential::get_energy_gradient_hessian with "
        "rattlers must be overloaded");
  }

  virtual double get_energy_gradient(Array<double> const &x,
                                     Array<double> &grad,
                                     Array<uint8_t> not_rattlers) {
    throw std::runtime_error(
        "BasePotential::get_energy_gradient with rattlers must be overloaded");
  }
  /**
   * compute the numerical gradient
   */
  virtual void numerical_gradient(Array<double> const &x, Array<double> &grad,
                                  double eps = 1e-6) {
    if (x.size() != grad.size()) {
      throw std::invalid_argument("grad.size() be the same as x.size()");
    }

    Array<double> xnew(x.copy());
    for (size_t i = 0; i < xnew.size(); ++i) {
      xnew[i] -= eps;
      double eminus = get_energy(xnew);
      xnew[i] += 2. * eps;
      double eplus = get_energy(xnew);
      grad[i] = (eplus - eminus) / (2. * eps);
      xnew[i] = x[i];
    }
  }

  /**
   * compute the hessian.
   *
   * If not overloaded it will call get_energy_gradient_hessian
   */
  virtual void get_hessian(Array<double> const &x, Array<double> &hess) {
    Array<double> grad(x.size());
    get_energy_gradient_hessian(x, grad, hess);
  }

  /**
   * compute the numerical gradient
   */
  virtual void numerical_hessian(Array<double> const &x, Array<double> &hess,
                                 double eps = 1e-6) {
    if (hess.size() != x.size() * x.size()) {
      throw std::invalid_argument(
          "hess.size() be the same as x.size()*x.size()");
    }
    size_t const N = x.size();

    Array<double> gplus(x.size());
    Array<double> gminus(x.size());

    Array<double> xnew(x.copy());
    for (size_t i = 0; i < xnew.size(); ++i) {
      double xbackup = xnew[i];
      xnew[i] -= eps;
      get_energy_gradient(xnew, gminus);
      xnew[i] += 2. * eps;
      get_energy_gradient(xnew, gplus);
      xnew[i] = xbackup;

      for (size_t j = 0; j < xnew.size(); ++j) {
        hess[N * i + j] = (gplus[j] - gminus[j]) / (2. * eps);
      }
    }
    /*//print hessian
    std::cout<<""<<std::endl;
    std::cout<<""<<std::endl;
    for (size_t i=0; i<hess.size(); ++i){
        std::cout<<hess[i]<<" ";
        int j = i+1;
        if (j % (int) (3*sqrt(hess.size())) == 0)
            std::cout<<"\n"<<std::endl;
        else if (j % (int) sqrt(hess.size()) == 0)
            std::cout<<""<<std::endl;
        else if (j % 3 == 0){
            std::cout<<"  ";
        }
    }*/
  }

  /**
   * compute a meaningful norm of all coordinates
   */
  virtual double compute_norm(pele::Array<double> const &x) { return norm(x); }
};
}  // namespace pele

#endif
