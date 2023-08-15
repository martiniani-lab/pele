#ifndef _PELE_TEST_FUNCS_H
#define _PELE_TEST_FUNCS_H

#include <cstddef>
#include <iostream>
#include <memory>
#include <numbers>
#include <vector>

#include "array.hpp"
#include "base_potential.hpp"

namespace pele {
/**
 * RosenBrock function with defaults for testing purposes
 */
class RosenBrock : public BasePotential {
 private:
  double m_a;
  double m_b;

 public:
  RosenBrock(double a = 1, double b = 100) : m_a(a), m_b(b){};
  virtual ~RosenBrock(){};
  inline double get_energy(Array<double> const &x) {
    return (m_a - x[0]) * (m_a - x[0]) +
           m_b * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]);
  }
  inline double get_energy_gradient(Array<double> const &x,
                                    Array<double> &grad) {
    grad.assign(0);
    grad[0] = 4 * m_b * x[0] * x[0] * x[0] - 4 * m_b * x[0] * x[1] +
              2 * m_a * x[0] - 2 * m_a;
    grad[1] = 2 * m_b * (x[1] - x[0] * x[0]);
    return (m_a - x[0]) * (m_a - x[0]) +
           m_b * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]);
  };
};

/**
 * Saddle point function for optimizer testing purposes
 */
class Saddle : public BasePotential {
 private:
  double m_a2;
  double m_b2;
  double m_a4;
  double m_b4;

 public:
  Saddle(double a2 = 2, double b2 = 2, double a4 = 1, double b4 = 1)
      : m_a2(a2), m_b2(b2), m_a4(a4), m_b4(b4){};
  virtual ~Saddle(){};
  inline double get_energy(Array<double> const &x) {
    return m_a2 * x[0] * x[0] - m_b2 * x[1] * x[1] +
           m_a4 * x[0] * x[0] * x[0] * x[0] + m_a4 * x[1] * x[1] * x[1] * x[1];
  }
  inline double get_energy_gradient(Array<double> const &x,
                                    Array<double> &grad) {
    grad.assign(0);
    grad[0] = 2 * m_a2 * x[0] + 4 * m_b4 * x[0] * x[0] * x[0];
    grad[1] = -2 * m_b2 * x[1] + 4 * m_b4 * x[1] * x[1] * x[1];
    return m_a2 * x[0] * x[0] - m_b2 * x[1] * x[1] +
           m_a4 * x[0] * x[0] * x[0] * x[0] + m_a4 * x[1] * x[1] * x[1] * x[1];
  };
};

/**
 * 1d x cubed function for testing optimizers. WARNING stop minimization before
 * the energy goes too low.
 *
 */
class XCube : public BasePotential {
 public:
  XCube(){};
  virtual ~XCube(){};
  inline double get_energy(Array<double> const &x) {
    return x[0] * x[0] * x[0];
  }
  inline double get_energy_gradient(Array<double> const &x,
                                    Array<double> &grad) {
    grad.assign(0);
    grad[0] = 3 * x[0] * x[0];
    std::cout << grad[0] << " graaadient \n";
    return x[0] * x[0] * x[0];
  };
};

/**
 * Flat Harmonic Potential. Flat in one dimension so that the hessian is
 * singular. Useful for testing optimizers where the singularity is handled.
 * The Potential is flat in the x+y direction. and the origin is at 0, 0.
 */

class FlatHarmonic : public BasePotential {
 private:
  double m_a;

 public:
  FlatHarmonic(double a = 1) : m_a(a){};
  virtual ~FlatHarmonic(){};
  inline double get_energy(Array<double> const &x) { return m_a * x[0] * x[0]; }
  inline double get_energy_gradient(Array<double> const &x,
                                    Array<double> &grad) {
    grad.assign(0);
    grad[0] = 2 * m_a * (x[0] - x[1]);
    grad[1] = -grad[0];
    return m_a * x[0] * x[0];
  };
  inline double get_energy_gradient_hessian(Array<double> const &x,
                                            Array<double> &grad,
                                            Array<double> &hess) {
    grad.assign(0);
    hess.assign(0);
    grad[0] = 2 * m_a * (x[0] - x[1]);
    grad[1] = -grad[0];
    hess[0] = 2 * m_a;
    hess[1] = -hess[0];
    hess[2] = -hess[0];
    hess[3] = hess[0];
    return m_a * x[0] * x[0];
  };
};
/**
 * Negative cosine product function for testing basin volume calculations.
 * Yields a hypercubic basin with volume period^dim.
 */
class NegativeCosProduct : public BasePotential {
 private:
  double period_factor;

  // precompute cos and sin values
  Array<double> cos_values;
  Array<double> sin_values;

 public:
  NegativeCosProduct(size_t dim, double period = 1)
      : period_factor(2 * std::numbers::pi / period),
        cos_values(dim),
        sin_values(dim){};
  virtual ~NegativeCosProduct(){};
  inline double get_energy(Array<double> const &x) {
    double energy = 1;
    for (auto x_i : x) {
      energy *= -std::cos(period_factor * x_i);
    }
    return energy;
  }

  double get_energy_gradient(Array<double> const &x, Array<double> &grad) {
    double cos_product = 1;
    for (size_t i = 0; i < x.size(); ++i) {
      cos_values[i] = std::cos(period_factor * x[i]);
      sin_values[i] = std::sin(period_factor * x[i]);
      cos_product *= cos_values[i];
    }

    for (size_t i = 0; i < x.size(); ++i) {
      grad[i] = cos_product;
      grad[i] *= sin_values[i] / cos_values[i];
    }
    return -cos_product;
  }

  double get_energy_gradient_hessian(Array<double> const &x,
                                     Array<double> &grad, Array<double> &hess) {
    double cos_product = 1;
    for (size_t i = 0; i < x.size(); ++i) {
      cos_values[i] = std::cos(period_factor * x[i]);
      sin_values[i] = std::sin(period_factor * x[i]);
      cos_product *= cos_values[i];
    }

    for (size_t i = 0; i < x.size(); ++i) {
      grad[i] = cos_product;
      grad[i] *= sin_values[i] / cos_values[i];
    }

    for (size_t i = 0; i < x.size(); ++i) {
      for (size_t j = 0; j < x.size(); ++j) {
        if (i == j) {
          hess[i * x.size() + j] = -cos_product;
        } else {
          hess[i * x.size() + j] = grad[i] * sin_values[j] / cos_values[j];
        }
      }
    }
    return -cos_product;
  }
};

}  // namespace pele
#endif
