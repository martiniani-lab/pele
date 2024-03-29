#ifndef PYGMIN_SIMPLE_PAIRWISE_POTENTIAL_H
#define PYGMIN_SIMPLE_PAIRWISE_POTENTIAL_H

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <memory>
#include <omp.h>
#include <sys/types.h>
#include <vector>

#include "array.hpp"
#include "distance.hpp"
#include "pairwise_potential_interface.hpp"
#include "vecn.hpp"

extern "C" {
#include "xsum.h"
}

namespace pele {
/**
 * Define a base class for potentials with simple pairwise interactions that
 * depend only on magnitude of the atom separation
 *
 * This class loops though atom pairs, computes the distances and get's the
 * value of the energy and gradient from the class pairwise_interaction.
 * pairwise_interaction is a passed parameter and defines the actual
 * potential function.
 */
template <typename pairwise_interaction,
          typename distance_policy = cartesian_distance<3>>
class SimplePairwisePotential : public PairwisePotentialInterface {
 private:
  /**
   * compute the energy and gradient for a single pair of atoms
   */
  inline void accumulate_pair_energy(double &energy_acc,
                                     Array<double> const &coords,
                                     const size_t coords_i,
                                     const size_t coords_j, const size_t atom_i,
                                     const size_t atom_j, double *dr_cache,
                                     const size_t m_ndim);

 protected:
  static const size_t m_ndim = distance_policy::_ndim;
  std::shared_ptr<pairwise_interaction> _interaction;
  std::shared_ptr<distance_policy> _dist;
  const double m_radii_sca;

  // variables for exact summation
  bool exact_sum;
  std::shared_ptr<std::vector<xsum_small_accumulator>> exact_gradient;
  bool exact_gradient_initialized;
  SimplePairwisePotential(std::shared_ptr<pairwise_interaction> interaction,
                          const Array<double> radii,
                          NonAdditiveCutoffCalculator cutoff_calculator,
                          std::shared_ptr<distance_policy> dist = NULL,
                          const double radii_sca = 0.0, bool exact_sum = false)
      : PairwisePotentialInterface(radii, cutoff_calculator),
        _interaction(interaction),
        _dist(dist),
        m_radii_sca(radii_sca),
        exact_gradient_initialized(false),
        exact_sum(exact_sum) {
    if (_dist == NULL) _dist = std::make_shared<distance_policy>();
  }

  SimplePairwisePotential(std::shared_ptr<pairwise_interaction> interaction,
                          const Array<double> radii,
                          std::shared_ptr<distance_policy> dist = NULL,
                          const double radii_sca = 0.0, bool exact_sum = false,
                          double non_additivity = 0.0)
      : PairwisePotentialInterface(radii, non_additivity),
        _interaction(interaction),
        _dist(dist),
        m_radii_sca(radii_sca),
        exact_gradient_initialized(false),
        exact_sum(exact_sum) {
    if (_dist == NULL) _dist = std::make_shared<distance_policy>();
  }

  SimplePairwisePotential(std::shared_ptr<pairwise_interaction> interaction,
                          std::shared_ptr<distance_policy> dist = NULL,
                          bool exact_sum = false)
      : _interaction(interaction),
        _dist(dist),
        m_radii_sca(0.0),
        exact_gradient_initialized(false),
        exact_sum(exact_sum) {
    if (_dist == NULL) _dist = std::make_shared<distance_policy>();
  }

 public:
  virtual ~SimplePairwisePotential() {}
  virtual inline size_t get_ndim() const { return m_ndim; }
  virtual double get_energy(Array<double> const &x);
  virtual double get_energy_change(Array<double> const &old_coords,
                                   Array<double> const &new_coords,
                                   std::vector<size_t> const &changed_atoms);
  virtual double get_energy_gradient(Array<double> const &x,
                                     Array<double> &grad) {
    grad.assign(0);
    return add_energy_gradient(x, grad);
  }

  virtual double get_energy_gradient(Array<double> const &x,
                                     Array<double> &grad,
                                     Array<uint8_t> not_rattlers) {
    grad.assign(0);
    return add_energy_gradient(x, grad, not_rattlers);
  }

  virtual double get_energy_gradient(
      Array<double> const &x, std::vector<xsum_small_accumulator> &exact_grad) {
    for (size_t i = 0; i < exact_grad.size(); ++i) {
      xsum_small_init(&(exact_grad[i]));
    };
    return add_energy_gradient(x, exact_grad);
  }

  virtual double get_energy_gradient_hessian(Array<double> const &x,
                                             Array<double> &grad,
                                             Array<double> &hess) {
    grad.assign(0);
    hess.assign(0);
    return add_energy_gradient_hessian(x, grad, hess);
  }
  virtual double get_energy_gradient_hessian_rattlers(
      Array<double> const &x, Array<double> &grad, Array<double> &hess,
      Array<uint8_t> not_rattlers) {
    grad.assign(0);
    hess.assign(0);
    std::cout << "rattlers" << std::endl;
    for (size_t i = 0; i < not_rattlers.size(); ++i) {
      std::cout << int(not_rattlers[i]) << ",";
    }
    std::cout << std::endl;
    return add_energy_gradient_hessian(x, grad, hess, not_rattlers);
  }
  virtual void get_hessian(Array<double> const &x, Array<double> &hess) {
    hess.assign(0);
    add_hessian(x, hess);
  }

  virtual double add_energy_gradient(Array<double> const &x,
                                     Array<double> &grad);

  virtual double add_energy_gradient(Array<double> const &x,
                                     Array<double> &grad,
                                     Array<uint8_t> not_rattlers);

  virtual double add_energy_gradient(
      Array<double> const &x, std::vector<xsum_small_accumulator> &exact_grad);
  virtual double add_energy_gradient_hessian(Array<double> const &x,
                                             Array<double> &grad,
                                             Array<double> &hess);
  virtual double add_energy_gradient_hessian(Array<double> const &x,
                                             Array<double> &grad,
                                             Array<double> &hess,
                                             Array<uint8_t> not_rattlers);
  virtual void add_hessian(Array<double> const &x, Array<double> &hess);
  virtual void get_neighbors(
      pele::Array<double> const &coords,
      pele::Array<std::vector<size_t>> &neighbor_indss,
      pele::Array<std::vector<std::vector<double>>> &neighbor_distss,
      const double cutoff_factor = 1.0);
  virtual void get_neighbors_picky(
      pele::Array<double> const &coords,
      pele::Array<std::vector<size_t>> &neighbor_indss,
      pele::Array<std::vector<std::vector<double>>> &neighbor_distss,
      pele::Array<short> const &include_atoms,
      const double cutoff_factor = 1.0);
  virtual std::vector<size_t> get_overlaps(Array<double> const &coords);
  virtual inline void get_rij(double *const r_ij, double const *const r1,
                              double const *const r2) const {
    return _dist->get_rij(r_ij, r1, r2);
  }
  virtual inline double get_interaction_energy_gradient(double r2, double *gij,
                                                        size_t atom_i,
                                                        size_t atom_j) const {
    return _interaction->energy_gradient(
        r2, gij, get_non_additive_cutoff(atom_i, atom_j));
  }
  virtual inline double get_interaction_energy_gradient_hessian(
      double r2, double *gij, double *hij, size_t atom_i, size_t atom_j) const {
    return _interaction->energy_gradient_hessian(
        r2, gij, hij, get_non_additive_cutoff(atom_i, atom_j));
  }

  // Compute the maximum of all single atom norms
  virtual inline double compute_norm(pele::Array<double> const &x) {
    const size_t natoms = x.size() / m_ndim;

    double max_x = 0;
    for (size_t atom_i = 0; atom_i < natoms; ++atom_i) {
      double atom_x = 0;
#pragma GCC unroll 3
      for (size_t j = 0; j < m_ndim; ++j) {
        size_t ind = atom_i * m_ndim + j;
        atom_x += x[ind] * x[ind];
      }
      max_x = std::max(max_x, atom_x);
    }
    return sqrt(max_x);
  }
};

template <typename pairwise_interaction, typename distance_policy>
inline double SimplePairwisePotential<pairwise_interaction, distance_policy>::
    add_energy_gradient(Array<double> const &x, Array<double> &grad) {
  const size_t natoms = x.size() / m_ndim;
  if (m_ndim * natoms != x.size()) {
    throw std::runtime_error(
        "x.size() is not divisible by the number of dimensions");
  }
  if (grad.size() != x.size()) {
    throw std::runtime_error("grad must have the same size as x");
  }
  double gij;
  double dr[m_ndim];
  if (exact_sum) {
    xsum_large_accumulator esum;
    xsum_large_init(&esum);
    if (!exact_gradient_initialized) {
      exact_gradient =
          std::make_shared<std::vector<xsum_small_accumulator>>(grad.size());
      exact_gradient_initialized = true;
    }

    for (size_t i = 0; i < grad.size(); ++i) {
      xsum_small_init(&((*exact_gradient)[i]));
    }
    for (size_t atom_i = 0; atom_i < natoms; ++atom_i) {
      const size_t i1 = m_ndim * atom_i;
      for (size_t atom_j = 0; atom_j < atom_i; ++atom_j) {
        const size_t j1 = m_ndim * atom_j;

        _dist->get_rij(dr, &x[i1], &x[j1]);
        double r2 = 0;
#pragma GCC unroll 3
        for (size_t k = 0; k < m_ndim; ++k) {
          r2 += dr[k] * dr[k];
        }
        xsum_large_add1(&esum,
                        _interaction->energy_gradient(
                            r2, &gij, get_non_additive_cutoff(atom_i, atom_j)));
        if (gij != 0) {
#pragma GCC unroll 3
          for (size_t k = 0; k < m_ndim; ++k) {
            dr[k] *= gij;
            xsum_small_add1(&((*exact_gradient)[i1 + k]), -dr[k]);
            xsum_small_add1(&((*exact_gradient)[j1 + k]), dr[k]);
          }
        }
      }
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < grad.size(); ++i) {
      grad[i] += xsum_small_round(&((*exact_gradient)[i]));
    }
#else
    for (size_t i = 0; i < grad.size(); ++i) {
      grad[i] += xsum_small_round(&((*exact_gradient)[i]));
    }
#endif
    return xsum_large_round(&esum);
  } else {
    double e = 0.;
    double gij;
    double dr[m_ndim];

    for (size_t atom_i = 0; atom_i < natoms; ++atom_i) {
      const size_t i1 = m_ndim * atom_i;
      for (size_t atom_j = 0; atom_j < atom_i; ++atom_j) {
        const size_t j1 = m_ndim * atom_j;

        _dist->get_rij(dr, &x[i1], &x[j1]);
        double r2 = 0;
#pragma GCC unroll 3
        for (size_t k = 0; k < m_ndim; ++k) {
          r2 += dr[k] * dr[k];
        }
        e += _interaction->energy_gradient(
            r2, &gij, get_non_additive_cutoff(atom_i, atom_j));

        if (gij != 0) {
#pragma GCC unroll 3
          for (size_t k = 0; k < m_ndim; ++k) {
            dr[k] *= gij;
            grad[i1 + k] -= dr[k];
            grad[j1 + k] += dr[k];
          }
        }
      }
    }
    return e;
  }
}

template <typename pairwise_interaction, typename distance_policy>
inline double SimplePairwisePotential<pairwise_interaction, distance_policy>::
    add_energy_gradient(Array<double> const &x, Array<double> &grad,
                        Array<uint8_t> mask) {
  const size_t natoms = x.size() / m_ndim;
  if (m_ndim * natoms != x.size()) {
    throw std::runtime_error(
        "x.size() is not divisible by the number of dimensions");
  }
  if (grad.size() != x.size()) {
    throw std::runtime_error("grad must have the same size as x");
  }
  double gij;
  double dr[m_ndim];
  // std::vector<double> grad_test (grad.size(), 0);

  if (exact_sum) {
    xsum_large_accumulator esum;
    xsum_large_init(&esum);
    if (!exact_gradient_initialized) {
      exact_gradient =
          std::make_shared<std::vector<xsum_small_accumulator>>(grad.size());
      exact_gradient_initialized = true;
    }

    for (size_t i = 0; i < grad.size(); ++i) {
      xsum_small_init(&((*exact_gradient)[i]));
    }
    for (size_t atom_i = 0; atom_i < natoms; ++atom_i) {
      const size_t i1 = m_ndim * atom_i;
      for (size_t atom_j = 0; atom_j < atom_i; ++atom_j) {
        const size_t j1 = m_ndim * atom_j;

        _dist->get_rij(dr, &x[i1], &x[j1]);
        double r2 = 0;
#pragma GCC unroll 3
        for (size_t k = 0; k < m_ndim; ++k) {
          r2 += dr[k] * dr[k];
        }
        if (mask[i1] && mask[j1]) {
          xsum_large_add1(
              &esum, _interaction->energy_gradient(
                         r2, &gij, get_non_additive_cutoff(atom_i, atom_j)));
        } else {
          gij = 0;
        }
        if (gij != 0) {
#pragma GCC unroll 3
          for (size_t k = 0; k < m_ndim; ++k) {
            dr[k] *= gij;
            xsum_small_add1(&((*exact_gradient)[i1 + k]), -dr[k]);
            xsum_small_add1(&((*exact_gradient)[j1 + k]), dr[k]);
          }
        }
      }
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < grad.size(); ++i) {
      grad[i] += xsum_small_round(&((*exact_gradient)[i]));
    }
#else
    for (size_t i = 0; i < grad.size(); ++i) {
      grad[i] += xsum_small_round(&((*exact_gradient)[i]));
    }
#endif
    return xsum_large_round(&esum);
  } else {
    double e = 0.;
    double gij;
    double dr[m_ndim];

    for (size_t atom_i = 0; atom_i < natoms; ++atom_i) {
      const size_t i1 = m_ndim * atom_i;
      for (size_t atom_j = 0; atom_j < atom_i; ++atom_j) {
        const size_t j1 = m_ndim * atom_j;

        _dist->get_rij(dr, &x[i1], &x[j1]);
        double r2 = 0;
#pragma GCC unroll 3
        for (size_t k = 0; k < m_ndim; ++k) {
          r2 += dr[k] * dr[k];
        }
        if (mask[i1] && mask[j1]) {
          e += _interaction->energy_gradient(
              r2, &gij, get_non_additive_cutoff(atom_i, atom_j));
        } else {
          gij = 0;
        }
        if (gij != 0) {
#pragma GCC unroll 3
          for (size_t k = 0; k < m_ndim; ++k) {
            dr[k] *= gij;
            grad[i1 + k] -= dr[k];
            grad[j1 + k] += dr[k];
          }
        }
      }
    }
    return e;
  }
}
template <typename pairwise_interaction, typename distance_policy>
inline double SimplePairwisePotential<pairwise_interaction, distance_policy>::
    add_energy_gradient(Array<double> const &x,
                        std::vector<xsum_small_accumulator> &exact_grad) {
  const size_t natoms = x.size() / m_ndim;
  if (m_ndim * natoms != x.size()) {
    throw std::runtime_error(
        "x.size() is not divisible by the number of dimensions");
  }
  if (exact_grad.size() != x.size()) {
    throw std::runtime_error("grad must have the same size as x");
  }

  xsum_large_accumulator esum;
  xsum_large_init(&esum);

  double gij;
  double dr[m_ndim];
  // std::vector<double> grad_test (grad.size(), 0);

  if (!exact_gradient_initialized) {
    exact_gradient = std::make_shared<std::vector<xsum_small_accumulator>>(
        exact_grad.size());
    exact_gradient_initialized = true;
  }

  for (size_t i = 0; i < exact_grad.size(); ++i) {
    xsum_small_init(&((*exact_gradient)[i]));
  }

  for (size_t atom_i = 0; atom_i < natoms; ++atom_i) {
    const size_t i1 = m_ndim * atom_i;
    for (size_t atom_j = 0; atom_j < atom_i; ++atom_j) {
      const size_t j1 = m_ndim * atom_j;

      _dist->get_rij(dr, &x[i1], &x[j1]);
      double r2 = 0;
#pragma GCC unroll 3
      for (size_t k = 0; k < m_ndim; ++k) {
        r2 += dr[k] * dr[k];
      }
      xsum_large_add1(&esum,
                      _interaction->energy_gradient(
                          r2, &gij, get_non_additive_cutoff(atom_i, atom_j)));
      if (gij != 0) {
#pragma GCC unroll 3
        for (size_t k = 0; k < m_ndim; ++k) {
          dr[k] *= gij;
          xsum_small_add1(&((*exact_gradient)[i1 + k]), -dr[k]);
          xsum_small_add1(&((*exact_gradient)[j1 + k]), dr[k]);
        }
      }
    }
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < exact_grad.size(); ++i) {
    // grad[i] += xsum_small_round(&((*exact_gradient)[i]));
    xsum_small_equal(&(exact_grad[i]), &((*exact_gradient)[i]));
  }
#else
  for (size_t i = 0; i < exact_grad.size(); ++i) {
    xsum_small_equal(&(exact_grad[i]), &((*exact_gradient)[i]));
  }
#endif
  return xsum_large_round(&esum);
}

template <typename pairwise_interaction, typename distance_policy>
inline double SimplePairwisePotential<pairwise_interaction, distance_policy>::
    add_energy_gradient_hessian(Array<double> const &x, Array<double> &grad,
                                Array<double> &hess) {
  double hij, gij;
  double dr[m_ndim];
  const size_t N = x.size();
  const size_t natoms = x.size() / m_ndim;
  if (m_ndim * natoms != x.size()) {
    throw std::runtime_error(
        "x.size() is not divisible by the number of dimensions");
  }
  if (x.size() != grad.size()) {
    throw std::invalid_argument("the gradient has the wrong size");
  }
  if (hess.size() != x.size() * x.size()) {
    throw std::invalid_argument("the Hessian has the wrong size");
  }

  double e = 0.;
  for (size_t atom_i = 0; atom_i < natoms; ++atom_i) {
    size_t i1 = m_ndim * atom_i;
    for (size_t atom_j = 0; atom_j < atom_i; ++atom_j) {
      size_t j1 = m_ndim * atom_j;

      _dist->get_rij(dr, &x[i1], &x[j1]);
      double r2 = 0;
#pragma GCC unroll 3
      for (size_t k = 0; k < m_ndim; ++k) {
        r2 += dr[k] * dr[k];
      }

      e += _interaction->energy_gradient_hessian(
          r2, &gij, &hij, get_non_additive_cutoff(atom_i, atom_j));

      if (gij != 0) {
#pragma GCC unroll 3
        for (size_t k = 0; k < m_ndim; ++k) {
          grad[i1 + k] -= gij * dr[k];
          grad[j1 + k] += gij * dr[k];
        }
      }

      if (hij != 0) {
#pragma GCC unroll 3
        for (size_t k = 0; k < m_ndim; ++k) {
          // diagonal block - diagonal terms
          double Hii_diag = (hij + gij) * dr[k] * dr[k] / r2 - gij;
          hess[N * (i1 + k) + i1 + k] += Hii_diag;
          hess[N * (j1 + k) + j1 + k] += Hii_diag;
          // off diagonal block - diagonal terms
          double Hij_diag = -Hii_diag;
          hess[N * (i1 + k) + j1 + k] += Hij_diag;
          hess[N * (j1 + k) + i1 + k] += Hij_diag;
#pragma GCC unroll 3
          for (size_t l = k + 1; l < m_ndim; ++l) {
            // diagonal block - off diagonal terms
            double Hii_off = (hij + gij) * dr[k] * dr[l] / r2;
            hess[N * (i1 + k) + i1 + l] += Hii_off;
            hess[N * (i1 + l) + i1 + k] += Hii_off;
            hess[N * (j1 + k) + j1 + l] += Hii_off;
            hess[N * (j1 + l) + j1 + k] += Hii_off;
            // off diagonal block - off diagonal terms
            double Hij_off = -Hii_off;
            hess[N * (i1 + k) + j1 + l] += Hij_off;
            hess[N * (i1 + l) + j1 + k] += Hij_off;
            hess[N * (j1 + k) + i1 + l] += Hij_off;
            hess[N * (j1 + l) + i1 + k] += Hij_off;
          }
        }
      }
    }
  }
  return e;
}

template <typename pairwise_interaction, typename distance_policy>
inline double SimplePairwisePotential<pairwise_interaction, distance_policy>::
    add_energy_gradient_hessian(Array<double> const &x, Array<double> &grad,
                                Array<double> &hess, Array<uint8_t> mask) {
  double hij, gij;
  double dr[m_ndim];
  const size_t N = x.size();
  const size_t natoms = x.size() / m_ndim;
  if (m_ndim * natoms != x.size()) {
    throw std::runtime_error(
        "x.size() is not divisible by the number of dimensions");
  }
  if (x.size() != grad.size()) {
    throw std::invalid_argument("the gradient has the wrong size");
  }
  if (hess.size() != x.size() * x.size()) {
    throw std::invalid_argument("the Hessian has the wrong size");
  }

  double e = 0.;
  for (size_t atom_i = 0; atom_i < natoms; ++atom_i) {
    size_t i1 = m_ndim * atom_i;
    for (size_t atom_j = 0; atom_j < atom_i; ++atom_j) {
      size_t j1 = m_ndim * atom_j;

      _dist->get_rij(dr, &x[i1], &x[j1]);
      double r2 = 0;
#pragma GCC unroll 3
      for (size_t k = 0; k < m_ndim; ++k) {
        r2 += dr[k] * dr[k];
      }
      if (mask[i1] && mask[j1]) {
        e += _interaction->energy_gradient_hessian(
            r2, &gij, &hij, get_non_additive_cutoff(atom_i, atom_j));
      } else {
        gij = 0;
        hij = 0;
      }
      if (gij != 0) {
#pragma GCC unroll 3
        for (size_t k = 0; k < m_ndim; ++k) {
          grad[i1 + k] -= gij * dr[k];
          grad[j1 + k] += gij * dr[k];
        }
      }

      if (hij != 0) {
#pragma GCC unroll 3
        for (size_t k = 0; k < m_ndim; ++k) {
          // diagonal block - diagonal terms
          double Hii_diag = (hij + gij) * dr[k] * dr[k] / r2 - gij;
          hess[N * (i1 + k) + i1 + k] += Hii_diag;
          hess[N * (j1 + k) + j1 + k] += Hii_diag;
          // off diagonal block - diagonal terms
          double Hij_diag = -Hii_diag;
          hess[N * (i1 + k) + j1 + k] = Hij_diag;
          hess[N * (j1 + k) + i1 + k] = Hij_diag;
#pragma GCC unroll 3
          for (size_t l = k + 1; l < m_ndim; ++l) {
            // diagonal block - off diagonal terms
            double Hii_off = (hij + gij) * dr[k] * dr[l] / r2;
            hess[N * (i1 + k) + i1 + l] += Hii_off;
            hess[N * (i1 + l) + i1 + k] += Hii_off;
            hess[N * (j1 + k) + j1 + l] += Hii_off;
            hess[N * (j1 + l) + j1 + k] += Hii_off;
            // off diagonal block - off diagonal terms
            double Hij_off = -Hii_off;
            hess[N * (i1 + k) + j1 + l] = Hij_off;
            hess[N * (i1 + l) + j1 + k] = Hij_off;
            hess[N * (j1 + k) + i1 + l] = Hij_off;
            hess[N * (j1 + l) + i1 + k] = Hij_off;
          }
        }
      }
    }
  }
  return e;
}
template <typename pairwise_interaction, typename distance_policy>
inline void
SimplePairwisePotential<pairwise_interaction, distance_policy>::add_hessian(
    Array<double> const &x, Array<double> &hess) {
  double hij;
  double dr[m_ndim];
  const size_t N = x.size();
  const size_t natoms = x.size() / m_ndim;
  if (m_ndim * natoms != x.size()) {
    throw std::runtime_error(
        "x.size() is not divisible by the number of dimensions");
  }
  if (hess.size() != x.size() * x.size()) {
    throw std::invalid_argument("the Hessian has the wrong size");
  }
  double e = 0.;
  double gij = 0;
  for (size_t atom_i = 0; atom_i < natoms; ++atom_i) {
    size_t i1 = m_ndim * atom_i;
    for (size_t atom_j = 0; atom_j < atom_i; ++atom_j) {
      size_t j1 = m_ndim * atom_j;

      _dist->get_rij(dr, &x[i1], &x[j1]);
      double r2 = 0;
#pragma GCC unroll 3
      for (size_t k = 0; k < m_ndim; ++k) {
        r2 += dr[k] * dr[k];
      }

      e += _interaction->energy_gradient_hessian(
          r2, &gij, &hij, get_non_additive_cutoff(atom_i, atom_j));

      if (hij != 0) {
#pragma GCC unroll 3
        for (size_t k = 0; k < m_ndim; ++k) {
          // diagonal block - diagonal terms
          double Hii_diag = (hij + gij) * dr[k] * dr[k] / r2 - gij;
          hess[N * (i1 + k) + i1 + k] += Hii_diag;
          hess[N * (j1 + k) + j1 + k] += Hii_diag;
          // off diagonal block - diagonal terms
          double Hij_diag = -Hii_diag;
          hess[N * (i1 + k) + j1 + k] += Hij_diag;
          hess[N * (j1 + k) + i1 + k] += Hij_diag;
#pragma GCC unroll 3
          for (size_t l = k + 1; l < m_ndim; ++l) {
            // diagonal block - off diagonal terms
            double Hii_off = (hij + gij) * dr[k] * dr[l] / r2;
            hess[N * (i1 + k) + i1 + l] += Hii_off;
            hess[N * (i1 + l) + i1 + k] += Hii_off;
            hess[N * (j1 + k) + j1 + l] += Hii_off;
            hess[N * (j1 + l) + j1 + k] += Hii_off;
            // off diagonal block - off diagonal terms
            double Hij_off = -Hii_off;
            hess[N * (i1 + k) + j1 + l] += Hij_off;
            hess[N * (i1 + l) + j1 + k] += Hij_off;
            hess[N * (j1 + k) + i1 + l] += Hij_off;
            hess[N * (j1 + l) + i1 + k] += Hij_off;
          }
        }
      }
    }
  }
}

template <typename pairwise_interaction, typename distance_policy>
inline void SimplePairwisePotential<pairwise_interaction, distance_policy>::
    accumulate_pair_energy(double &energy_accumulator,
                           Array<double> const &coords, const size_t coords_i,
                           const size_t coords_j, const size_t atom_i,
                           const size_t atom_j, double *dr_cache,
                           const size_t m_ndim) {
  _dist->get_rij(dr_cache, &coords[coords_i], &coords[coords_j]);
  double r2 = 0;
#pragma GCC unroll 3
  for (size_t k = 0; k < m_ndim; ++k) {
    r2 += dr_cache[k] * dr_cache[k];
  }
  double e_ij =
      _interaction->energy(r2, get_non_additive_cutoff(atom_i, atom_j));
  energy_accumulator += e_ij;
}

template <typename pairwise_interaction, typename distance_policy>
inline double
SimplePairwisePotential<pairwise_interaction, distance_policy>::get_energy(
    Array<double> const &coords) {
  const size_t natoms = coords.size() / m_ndim;

  if (m_ndim * natoms != coords.size()) {
    throw std::runtime_error(
        "x.size() is not divisible by "
        "the number of dimensions");
  }
  double energy_accumulator = 0.;
  double dr_cache[m_ndim];
  for (size_t atom_i = 0; atom_i < natoms; ++atom_i) {
    size_t coords_i = m_ndim * atom_i;
    for (size_t atom_j = 0; atom_j < atom_i; ++atom_j) {
      size_t coords_j = m_ndim * atom_j;
      accumulate_pair_energy(energy_accumulator, coords, coords_i, coords_j,
                             atom_i, atom_j, dr_cache, m_ndim);
    }
  }
  return energy_accumulator;
}

template <typename pairwise_interaction, typename distance_policy>
inline double SimplePairwisePotential<pairwise_interaction, distance_policy>::
    get_energy_change(Array<double> const &old_coords,
                      Array<double> const &new_coords,
                      const std::vector<size_t> &changed_atoms) {
  assert(old_coords.size() == new_coords.size());
  const size_t natoms = old_coords.size() / m_ndim;

  if (m_ndim * natoms != old_coords.size()) {
    throw std::runtime_error(
        "x.size() is not divisible by "
        "the number of dimensions");
  }
  double energy_contribution = 0.;
  double dr_cache[m_ndim];

  // First, calculate energy contributions of all atoms not in changed_atoms in
  // the old configuration
  for (auto atom_i : changed_atoms) {
    size_t coords_i = m_ndim * atom_i;
    for (size_t atom_j = 0; atom_j < natoms; ++atom_j) {
      if (std::find(changed_atoms.begin(), changed_atoms.end(), atom_j) ==
          changed_atoms.end()) {
        size_t coords_j = m_ndim * atom_j;
        accumulate_pair_energy(energy_contribution, old_coords, coords_i,
                               coords_j, atom_i, atom_j, dr_cache, m_ndim);
      }
    }
  }
  // Now, calculate energy contributions of atom_i, atom_j pairs
  for (auto atom_i : changed_atoms) {
    size_t coords_i = m_ndim * atom_i;
    for (auto atom_j : changed_atoms) {
      // Keep pairs unique
      if (atom_j < atom_i) {
        size_t coords_j = m_ndim * atom_j;
        accumulate_pair_energy(energy_contribution, old_coords, coords_i,
                               coords_j, atom_i, atom_j, dr_cache, m_ndim);
      }
    }
  }
  // because old energy is subtracted from the new energy contribution
  energy_contribution = -energy_contribution;

  // Now Accumulate the new energy contribution
  // First, calculate energy contributions of all atoms not in changed_atoms in
  // the old configuration
  for (auto atom_i : changed_atoms) {
    size_t coords_i = m_ndim * atom_i;
    for (size_t atom_j = 0; atom_j < natoms; ++atom_j) {
      if (std::find(changed_atoms.begin(), changed_atoms.end(), atom_j) ==
          changed_atoms.end()) {
        size_t coords_j = m_ndim * atom_j;
        accumulate_pair_energy(energy_contribution, new_coords, coords_i,
                               coords_j, atom_i, atom_j, dr_cache, m_ndim);
      }
    }
  }
  // Now, calculate energy contributions of atom_i, atom_j pairs
  for (auto atom_i : changed_atoms) {
    size_t coords_i = m_ndim * atom_i;
    for (auto atom_j : changed_atoms) {
      // Keep pairs unique
      if (atom_j < atom_i) {
        size_t coords_j = m_ndim * atom_j;
        accumulate_pair_energy(energy_contribution, new_coords, coords_i,
                               coords_j, atom_i, atom_j, dr_cache, m_ndim);
      }
    }
  }
  return energy_contribution;
}

template <typename pairwise_interaction, typename distance_policy>
void SimplePairwisePotential<pairwise_interaction, distance_policy>::
    get_neighbors(
        pele::Array<double> const &coords,
        pele::Array<std::vector<size_t>> &neighbor_indss,
        pele::Array<std::vector<std::vector<double>>> &neighbor_distss,
        const double cutoff_factor /*=1.0*/) {
  const size_t natoms = coords.size() / m_ndim;
  pele::Array<short> include_atoms(natoms, 1);
  get_neighbors_picky(coords, neighbor_indss, neighbor_distss, include_atoms,
                      cutoff_factor);
}

template <typename pairwise_interaction, typename distance_policy>
void SimplePairwisePotential<pairwise_interaction, distance_policy>::
    get_neighbors_picky(
        pele::Array<double> const &coords,
        pele::Array<std::vector<size_t>> &neighbor_indss,
        pele::Array<std::vector<std::vector<double>>> &neighbor_distss,
        pele::Array<short> const &include_atoms,
        const double cutoff_factor /*=1.0*/) {
  const size_t natoms = coords.size() / m_ndim;
  if (m_ndim * natoms != coords.size()) {
    throw std::runtime_error(
        "coords.size() is not divisible "
        "by the number of dimensions");
  }
  if (natoms != include_atoms.size()) {
    throw std::runtime_error(
        "include_atoms.size() is not "
        "equal to the number of atoms");
  }
  if (m_radii.size() == 0) {
    throw std::runtime_error(
        "Can't calculate neighbors, "
        "because the "
        "used interaction doesn't use "
        "radii. ");
  }
  std::vector<double> dr(m_ndim);
  std::vector<double> neg_dr(m_ndim);
  neighbor_indss = pele::Array<std::vector<size_t>>(natoms);
  neighbor_distss = pele::Array<std::vector<std::vector<double>>>(natoms);

  const double cutoff_sca = (1 + m_radii_sca) * cutoff_factor;

  for (size_t atom_i = 0; atom_i < natoms; ++atom_i) {
    if (include_atoms[atom_i]) {
      size_t i1 = m_ndim * atom_i;
      for (size_t atom_j = 0; atom_j < atom_i; ++atom_j) {
        if (include_atoms[atom_j]) {
          size_t j1 = m_ndim * atom_j;

          _dist->get_rij(dr.data(), &coords[i1], &coords[j1]);
          double r2 = 0;
#pragma GCC unroll 3
          for (size_t k = 0; k < m_ndim; ++k) {
            r2 += dr[k] * dr[k];
            neg_dr[k] = -dr[k];
          }

          const double r_H = get_non_additive_cutoff(atom_i, atom_j);
          const double r_S = cutoff_sca * r_H;
          const double r_S2 = r_S * r_S;
          if (r2 <= r_S2) {
            neighbor_indss[atom_i].push_back(atom_j);
            neighbor_indss[atom_j].push_back(atom_i);
            neighbor_distss[atom_i].push_back(dr);
            neighbor_distss[atom_j].push_back(neg_dr);
          }
        }
      }
    }
  }
}

template <typename pairwise_interaction, typename distance_policy>
std::vector<size_t>
SimplePairwisePotential<pairwise_interaction, distance_policy>::get_overlaps(
    Array<double> const &coords) {
  const size_t natoms = coords.size() / m_ndim;
  if (m_ndim * natoms != coords.size()) {
    throw std::runtime_error(
        "coords.size() is not divisible "
        "by the number of dimensions");
  }
  if (m_radii.size() == 0) {
    throw std::runtime_error(
        "Can't calculate neighbors, "
        "because the "
        "used interaction doesn't use "
        "radii. ");
  }
  pele::VecN<m_ndim, double> dr;
  std::vector<size_t> overlap_inds;

  for (size_t atom_i = 0; atom_i < natoms; ++atom_i) {
    size_t i1 = m_ndim * atom_i;
    for (size_t atom_j = 0; atom_j < atom_i; ++atom_j) {
      size_t j1 = m_ndim * atom_j;

      _dist->get_rij(dr.data(), &coords[i1], &coords[j1]);
      double r2 = 0;
#pragma GCC unroll 3
      for (size_t k = 0; k < m_ndim; ++k) {
        r2 += dr[k] * dr[k];
      }

      const double r_H = get_non_additive_cutoff(atom_i, atom_j);
      const double r_H2 = r_H * r_H;
      if (r2 <= r_H2) {
        overlap_inds.push_back(atom_i);
        overlap_inds.push_back(atom_j);
      }
    }
  }
  return overlap_inds;
}
}  // namespace pele

#endif  // #ifndef PYGMIN_SIMPLE_PAIRWISE_POTENTIAL_H
