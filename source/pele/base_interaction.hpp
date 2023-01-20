#ifndef PYGMIN_BASE_INTERACTION_H
#define PYGMIN_BASE_INTERACTION_H

namespace pele {

struct BaseInteraction {
  virtual double energy(const double r2, const double dij) const {
    throw std::runtime_error("BaseInteraction::energy must be overloaded");
  }

  virtual double energy_gradient(const double r2, double *const gij,
                                 const double dij) const {
    throw std::runtime_error("BaseInteraction::energy must be overloaded");
  }

  virtual double energy_gradient_hessian(const double r2, double *const gij,
                                         double *const hij,
                                         const double dij) const {
    throw std::runtime_error("BaseInteraction::energy must be overloaded");
  }
};

}  // namespace pele

#endif  // PYGMIN_BASE_INTERACTION_H
