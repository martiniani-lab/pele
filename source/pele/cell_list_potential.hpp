#ifndef _PELE_CELL_LIST_POTENTIAL_H
#define _PELE_CELL_LIST_POTENTIAL_H

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <math.h>
#include <memory>
#include <numeric>
#include <omp.h>
#include <pele/base_potential.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

#include "array.hpp"
#include "cell_lists.hpp"
#include "distance.hpp"
#include "pairwise_potential_interface.hpp"
#include "vecn.hpp"

namespace pele {

inline void init_energies(vector<double> &energies) {
#ifdef _OPENMP
  energies = std::vector<double>(omp_get_max_threads());
#pragma omp parallel
  { energies[omp_get_thread_num()] = 0; }
#else
  energies = std::vector<double>(1);
  energies[0] = 0;
#endif
}

inline void reset_energies(vector<double> &energies) {
#ifdef _OPENMP
#pragma omp parallel
  { energies[omp_get_thread_num()] = 0; }
#else
  energies[0] = 0;
#endif
}

inline void accumulate_energies_omp(vector<double> &energies, double energy) {
#ifdef _OPENMP
  energies[omp_get_thread_num()] += energy;
#else
  energies[0] += energy;
#endif
}

inline size_t thread() {
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

// class containing r2, dij, dr, , xi_off, xj_off
template <size_t ndim> class DistanceData {
public:
  double r2;                   // r2 = dr * dr
  double dij;                  // cutoff distance between particles i and j
  size_t xi_off;               // offset of particle i in the x array
  size_t xj_off;               // offset of particle j in the x array
  pele::VecN<ndim, double> dr; // vector from particle i to particle j
  DistanceData() : r2(0), dij(0), xi_off(0), xj_off(0), dr() {}
  inline void reset() {
    r2 = 0;
    dij = 0;
    xi_off = 0;
    xj_off = 0;
    std::fill(dr.begin(), dr.end(), 0);
  }
};
template <size_t ndim>
inline void
init_distance_datas(std::vector<DistanceData<ndim>> &distance_datas) {
#ifdef _OPENMP
  distance_datas = std::vector<DistanceData<ndim>>(omp_get_max_threads());
#else
  distance_datas = std::vector<DistanceData<ndim>>(1);
#endif
}

/*
 * calculates hessian contribution in cartesian coordinates from a pair of
 * particles given the hessian and gradient contribution in spherical
 * coordinates, then adds it to the total hessian to the cartesian coordinate
 * total hessian
 */
template <size_t ndim>
inline void accumulate_gradient(pele::Array<double> &gradient, const double gij,
                                DistanceData<ndim> const &data) {
  if (gij != 0) {
    for (size_t k = 0; k < ndim; ++k) {
      gradient[data.xi_off + k] -= data.dr[k] * gij;
      gradient[data.xj_off + k] += data.dr[k] * gij;
    }
  }
}

/*
 * calculates hessian contribution in cartesian coordinates from a pair of
 * particles given the hessian and gradient contribution in spherical
 * coordinates, then adds it to the total hessian to the cartesian coordinate
 * total hessian
 */
template <size_t ndim>
inline void accumulate_hessian(pele::Array<double> &hessian, const double hij,
                               const double gij, DistanceData<ndim> const &data,
                               const size_t coords_size) {
  if (hij != 0) {
    const size_t i1 = data.xi_off;
    const size_t j1 = data.xj_off;
    for (size_t k = 0; k < ndim; ++k) {
      // diagonal block - diagonal terms
      const double Hii_diag =
          (hij + gij) * data.dr[k] * data.dr[k] / data.r2 - gij;
      hessian[coords_size * (i1 + k) + i1 + k] += Hii_diag;
      hessian[coords_size * (j1 + k) + j1 + k] += Hii_diag;
      // off diagonal block - diagonal terms
      const double Hij_diag = -Hii_diag;
      hessian[coords_size * (i1 + k) + j1 + k] += Hij_diag;
      hessian[coords_size * (j1 + k) + i1 + k] += Hij_diag;
      for (size_t l = k + 1; l < ndim; ++l) {
        // diagonal block - off diagonal terms
        const double Hii_off = (hij + gij) * data.dr[k] * data.dr[l] / data.r2;
        hessian[coords_size * (i1 + k) + i1 + l] += Hii_off;
        hessian[coords_size * (i1 + l) + i1 + k] += Hii_off;
        hessian[coords_size * (j1 + k) + j1 + l] += Hii_off;
        hessian[coords_size * (j1 + l) + j1 + k] += Hii_off;
        // off diagonal block - off diagonal terms
        const double Hij_off = -Hii_off;
        hessian[coords_size * (i1 + k) + j1 + l] += Hij_off;
        hessian[coords_size * (i1 + l) + j1 + k] += Hij_off;
        hessian[coords_size * (j1 + k) + i1 + l] += Hij_off;
        hessian[coords_size * (j1 + l) + i1 + k] += Hij_off;
      }
    }
  }
}

/*
 * This Class forms the Base Class for all accumulator instances
 * that are used in the CellListPotential class.
 */
template <typename pairwise_interaction, typename distance_policy>
class BaseAccumulator {
protected:
  const static size_t m_ndim = distance_policy::_ndim;
  std::shared_ptr<pairwise_interaction> m_interaction;
  std::shared_ptr<distance_policy> m_dist;
  const Array<double> *m_coords;
  const Array<double> m_radii;
  NonAdditiveCutoffCalculator m_cutoff;
  std::vector<DistanceData<distance_policy::_ndim>>
      m_distance_datas; // distance data for every thread. Prevents being
                        // overwritten by other threads

public:
  BaseAccumulator(
      std::shared_ptr<pairwise_interaction> pairwise_interaction_ptr,
      std::shared_ptr<distance_policy> distance_policy_ptr,
      pele::Array<double> const &radii, NonAdditiveCutoffCalculator cutoff)
      : m_interaction(pairwise_interaction_ptr), m_dist(distance_policy_ptr),
        m_coords(nullptr), m_radii(radii), m_cutoff(cutoff),
        m_distance_datas(0) {
    init_distance_datas(m_distance_datas);
  }

  void reset_data(const pele::Array<double> *coords) { m_coords = coords; }

  /*
   * Calculates distance data for a pair of particles. i.e sets DistanceData
   * variables in the subdomain
   *
   */
  inline void
  calculate_distance_data(DistanceData<distance_policy::_ndim> &dist_data,
                          const pele::Array<double> &coords,
                          const size_t atom_i, const size_t atom_j) {

    dist_data.xi_off = m_ndim * atom_i;
    dist_data.xj_off = m_ndim * atom_j;
    m_dist->get_rij(dist_data.dr.data(), m_coords->data() + dist_data.xi_off,
                    m_coords->data() + dist_data.xj_off);
    dist_data.r2 = 0;
    for (size_t k = 0; k < m_ndim; ++k) {
      dist_data.r2 += dist_data.dr[k] * dist_data.dr[k];
    }
    if (m_radii.size() > 0) {
      dist_data.dij = m_cutoff.get_cutoff(m_radii[atom_i], m_radii[atom_j]);
    }
  }

  inline void calculate_dist_data_in_thread(const pele::Array<double> &coords,
                                            const size_t atom_i,
                                            const size_t atom_j) {
#ifdef _OPENMP
    calculate_distance_data(m_distance_datas[omp_get_thread_num()], coords,
                            atom_i, atom_j);
#else
    calculate_distance_data(m_distance_datas[0], coords, atom_i, atom_j);
#endif
  }
};

/**
 * class which accumulates the energy one pair interaction at a time
 */
template <typename pairwise_interaction, typename distance_policy>
class EnergyAccumulator
    : public BaseAccumulator<pairwise_interaction, distance_policy> {
  std::vector<double> m_energies;

public:
  ~EnergyAccumulator() {}

  EnergyAccumulator(
      std::shared_ptr<pairwise_interaction> &interaction,
      std::shared_ptr<distance_policy> &dist,
      pele::Array<double> const &radii = pele::Array<double>(0),
      NonAdditiveCutoffCalculator cutoff = NonAdditiveCutoffCalculator())
      : BaseAccumulator<pairwise_interaction, distance_policy>(
            interaction, dist, radii, cutoff),
        m_energies(0) {
    init_energies(m_energies);
  }

  void reset_data(const pele::Array<double> *coords) {
    this->m_coords = coords;
    reset_energies(m_energies);
  }

  void insert_atom_pair(const size_t atom_i, const size_t atom_j,
                        const size_t isubdom) {
    this->calculate_dist_data_in_thread(*this->m_coords, atom_i, atom_j);
    double energy =
        this->m_interaction->energy(this->m_distance_datas[thread()].r2,
                                    this->m_distance_datas[thread()].dij);
    accumulate_energies_omp(m_energies, energy);
  }
  double get_energy() {
    return std::reduce(m_energies.begin(),
                       m_energies.end()); // reduce sums up energies
  }
};

/**
 * class which accumulates the energy and gradient one pair interaction at a
 * time
 */
template <typename pairwise_interaction, typename distance_policy>
class EnergyGradientAccumulator
    : public BaseAccumulator<pairwise_interaction, distance_policy> {
  std::vector<double> m_energies;

public:
  pele::Array<double> *m_gradient;
  ~EnergyGradientAccumulator() {}
  EnergyGradientAccumulator(
      std::shared_ptr<pairwise_interaction> &interaction,
      std::shared_ptr<distance_policy> &dist,
      pele::Array<double> const &radii = pele::Array<double>(0),
      NonAdditiveCutoffCalculator cutoff = NonAdditiveCutoffCalculator())
      : BaseAccumulator<pairwise_interaction, distance_policy>(
            interaction, dist, radii, cutoff),
        m_energies(0) {
    init_energies(m_energies);
  }

  void reset_data(const pele::Array<double> *coords,
                  pele::Array<double> *gradient) {
    this->m_coords = coords;
    reset_energies(m_energies);
    m_gradient = gradient;
  }

  void insert_atom_pair(const size_t atom_i, const size_t atom_j,
                        const size_t isubdom) {
    this->calculate_dist_data_in_thread(*this->m_coords, atom_i, atom_j);
    double gij;
    double energy = this->m_interaction->energy_gradient(
        this->m_distance_datas[thread()].r2, &gij,
        this->m_distance_datas[thread()].dij);
    accumulate_energies_omp(m_energies, energy);
    accumulate_gradient(*m_gradient, gij, this->m_distance_datas[thread()]);
  }

  double get_energy() {
    return std::reduce(m_energies.begin(), m_energies.end());
  }
};

/**
 * class which accumulates the energy, gradient, and Hessian one pair
 * interaction at a time
 */
template <typename pairwise_interaction, typename distance_policy>
class EnergyGradientHessianAccumulator
    : public BaseAccumulator<pairwise_interaction, distance_policy> {
  std::vector<double> m_energies;

public:
  pele::Array<double> *m_gradient;
  pele::Array<double> *m_hessian;

  ~EnergyGradientHessianAccumulator() {}

  EnergyGradientHessianAccumulator(
      std::shared_ptr<pairwise_interaction> &interaction,
      std::shared_ptr<distance_policy> &dist,
      pele::Array<double> const &radii = pele::Array<double>(0),
      NonAdditiveCutoffCalculator cutoff = NonAdditiveCutoffCalculator())
      : BaseAccumulator<pairwise_interaction, distance_policy>(
            interaction, dist, radii, cutoff),
        m_energies(0) {
    init_energies(m_energies);
  }

  void reset_data(const pele::Array<double> *coords,
                  pele::Array<double> *gradient, pele::Array<double> *hessian) {
    this->m_coords = coords;
    reset_energies(m_energies);
    m_gradient = gradient;
    m_hessian = hessian;
  }
  void insert_atom_pair(const size_t atom_i, const size_t atom_j,
                        const size_t isubdom) {
    this->calculate_dist_data_in_thread(*this->m_coords, atom_i, atom_j);
    double gij, hij;
    double energy = this->m_interaction->energy_gradient_hessian(
        this->m_distance_datas[thread()].r2, &gij, &hij,
        this->m_distance_datas[thread()].dij);

    accumulate_energies_omp(m_energies, energy);
    accumulate_gradient(*m_gradient, gij, this->m_distance_datas[thread()]);
    accumulate_hessian(*m_hessian, hij, gij, this->m_distance_datas[thread()],
                       m_gradient->size());
  }
  double get_energy() {
    return std::reduce(m_energies.begin(), m_energies.end());
  }
};

/**
 * @brief Accumulates the energy and the hessian one pair interaction at a time
 *
 * @tparam pairwise_interaction interaction class
 * @tparam distance_policy how to calculate the distance between two atoms
 */
template <typename pairwise_interaction, typename distance_policy>
class EnergyHessianAccumulator
    : public BaseAccumulator<pairwise_interaction, distance_policy> {
  std::vector<double> m_energies;
  pele::Array<double> *m_hessian;

public:
  ~EnergyHessianAccumulator() {}

  EnergyHessianAccumulator(
      std::shared_ptr<pairwise_interaction> &interaction,
      std::shared_ptr<distance_policy> &dist,
      pele::Array<double> const &radii = pele::Array<double>(0),
      NonAdditiveCutoffCalculator cutoff = NonAdditiveCutoffCalculator())
      : BaseAccumulator<pairwise_interaction, distance_policy>(
            interaction, dist, radii, cutoff),
        m_energies(0) {
    init_energies(m_energies);
  }

  void reset_data(const pele::Array<double> *coords,
                  pele::Array<double> *hessian) {
    this->m_coords = coords;
    reset_energies(m_energies);
    m_hessian = hessian;
  }

  void insert_atom_pair(const size_t atom_i, const size_t atom_j,
                        const size_t isubdom) {
    this->calculate_dist_data_in_thread(*this->m_coords, atom_i, atom_j);
    double gij, hij;
    double energy = this->m_interaction->energy_gradient_hessian(
        this->m_distance_datas[thread()].r2, &gij, &hij,
        this->m_distance_datas[thread()].dij);

    accumulate_energies_omp(m_energies, energy);
    accumulate_hessian(*m_hessian, hij, gij, this->m_distance_datas[thread()],
                       this->m_coords->size());
  }
  double get_energy() {
    return std::reduce(m_energies.begin(), m_energies.end());
  }
};
/**
 * class which accumulates the energy one pair interaction at a time with
 * exact summation
 */
template <typename pairwise_interaction, typename distance_policy>
class EnergyAccumulatorExact {
  const static size_t m_ndim = distance_policy::_ndim;
  std::shared_ptr<pairwise_interaction> m_interaction;
  std::shared_ptr<distance_policy> m_dist;
  const pele::Array<double> *m_coords;
  const pele::Array<double> m_radii;
  std::vector<xsum_large_accumulator> m_energies;

public:
  ~EnergyAccumulatorExact() {
    // for(auto & energy : m_energies) {
    //   delete energy;
    // }
  }

  EnergyAccumulatorExact(
      std::shared_ptr<pairwise_interaction> &interaction,
      std::shared_ptr<distance_policy> &dist,
      pele::Array<double> const &radii = pele::Array<double>(0))
      : m_interaction(interaction), m_dist(dist), m_radii(radii) {
#ifdef _OPENMP
    m_energies = std::vector<xsum_large_accumulator>(omp_get_max_threads());
#pragma omp parallel
    { xsum_large_init(&(m_energies[omp_get_thread_num()])); }
#else
    m_energies = std::vector<xsum_large_accumulator>(1);
    xsum_large_init(&(m_energies[0]));
#endif
  }

  void reset_data(const pele::Array<double> *coords) {
    m_coords = coords;
#ifdef _OPENMP
#pragma omp parallel
    { xsum_large_init(&(m_energies[omp_get_thread_num()])); }
#else
    xsum_large_init(&(m_energies[0]));
#endif
  }

  void insert_atom_pair(const size_t atom_i, const size_t atom_j,
                        const size_t isubdom) {
    const size_t xi_off = m_ndim * atom_i;
    const size_t xj_off = m_ndim * atom_j;
    pele::VecN<m_ndim, double> dr;
    m_dist->get_rij(dr.data(), m_coords->data() + xi_off,
                    m_coords->data() + xj_off);
    double r2 = 0;
    for (size_t k = 0; k < m_ndim; ++k) {
      r2 += dr[k] * dr[k];
    }
    double dij = 0;
    if (m_radii.size() > 0) {
      dij = m_radii[atom_i] + m_radii[atom_j];
    }
#ifdef _OPENMP
    xsum_large_add1(&(m_energies[isubdom]), m_interaction->energy(r2, dij));
#else
    xsum_large_add1(&(m_energies[0]), m_interaction->energy(r2, dij));
#endif
  }

  double get_energy() {
    double energy = 0;
    for (size_t i = 0; i < m_energies.size(); ++i) {
      energy += xsum_large_round(&(m_energies[i]));
    }
    return energy;
  }
};

/**
 * class which accumulates the energy and gradient one pair interaction at a
 * time, with exact summation
 */
template <typename pairwise_interaction, typename distance_policy>
class EnergyGradientAccumulatorExact {
  const static size_t m_ndim = distance_policy::_ndim;
  std::shared_ptr<pairwise_interaction> m_interaction;
  std::shared_ptr<distance_policy> m_dist;
  const pele::Array<double> *m_coords;
  const pele::Array<double> m_radii;
  std::vector<xsum_large_accumulator> m_energies;

public:
  std::shared_ptr<std::vector<xsum_small_accumulator>> m_gradient;
  ~EnergyGradientAccumulatorExact() {
    // for(auto & energy : m_energies) {
    //   delete energy;
    // }
  }

  EnergyGradientAccumulatorExact(
      std::shared_ptr<pairwise_interaction> &interaction,
      std::shared_ptr<distance_policy> &dist,
      pele::Array<double> const &radii = pele::Array<double>(0))
      : m_interaction(interaction), m_dist(dist), m_radii(radii) {
#ifdef _OPENMP
    m_energies = std::vector<xsum_large_accumulator>(omp_get_max_threads());
#pragma omp parallel
    { xsum_large_init(&(m_energies[omp_get_thread_num()])); }
#else
    m_energies = std::vector<xsum_large_accumulator>(1);
    xsum_large_init(&(m_energies[0]));
#endif
  }

  void
  reset_data(const pele::Array<double> *coords,
             std::shared_ptr<std::vector<xsum_small_accumulator>> &gradient) {
    m_coords = coords;
#ifdef _OPENMP
#pragma omp parallel
    { xsum_large_init(&(m_energies[omp_get_thread_num()])); }
#else
    xsum_large_init(&(m_energies[0]));
#endif
    m_gradient = gradient;
  }

  void insert_atom_pair(const size_t atom_i, const size_t atom_j,
                        const size_t isubdom) {
    pele::VecN<m_ndim, double> dr;
    const size_t xi_off = m_ndim * atom_i;
    const size_t xj_off = m_ndim * atom_j;
    m_dist->get_rij(dr.data(), m_coords->data() + xi_off,
                    m_coords->data() + xj_off);
    double r2 = 0;
    for (size_t k = 0; k < m_ndim; ++k) {
      r2 += dr[k] * dr[k];
    }
    double gij;
    double dij = 0;
    if (m_radii.size() > 0) {
      dij = m_radii[atom_i] + m_radii[atom_j];
    }
#ifdef _OPENMP
    xsum_large_add1(&(m_energies[isubdom]),
                    m_interaction->energy_gradient(r2, &gij, dij));
#else
    xsum_large_add1(&(m_energies[0]),
                    m_interaction->energy_gradient(r2, &gij, dij));
#endif
    if (gij != 0) {
      for (size_t k = 0; k < m_ndim; ++k) {
        dr[k] *= gij;
        xsum_small_add1(&((*m_gradient)[xi_off + k]), -dr[k]);
        xsum_small_add1(&((*m_gradient)[xj_off + k]), dr[k]);
      }
    }
  }

  double get_energy() {
    double energy = 0;
    for (size_t i = 0; i < m_energies.size(); ++i) {
      energy += xsum_large_round(&(m_energies[i]));
    }
    return energy;
  }
};

/**
 * TODO: Exact summation has not been implemented in this class yet.
 */
template <typename pairwise_interaction, typename distance_policy>
class EnergyGradientHessianAccumulatorExact {
  const static size_t m_ndim = distance_policy::_ndim;
  std::shared_ptr<pairwise_interaction> m_interaction;
  std::shared_ptr<distance_policy> m_dist;
  const pele::Array<double> *m_coords;
  const pele::Array<double> m_radii;
  std::vector<double *> m_energies;

public:
  pele::Array<double> *m_gradient;
  pele::Array<double> *m_hessian;

  ~EnergyGradientHessianAccumulatorExact() {
    for (auto &energy : m_energies) {
      delete energy;
    }
  }

  EnergyGradientHessianAccumulatorExact(
      std::shared_ptr<pairwise_interaction> &interaction,
      std::shared_ptr<distance_policy> &dist,
      pele::Array<double> const &radii = pele::Array<double>(0))
      : m_interaction(interaction), m_dist(dist), m_radii(radii) {
#ifdef _OPENMP
    m_energies = std::vector<double *>(omp_get_max_threads());
#pragma omp parallel
    { m_energies[omp_get_thread_num()] = new double(); }
#else
    m_energies = std::vector<double *>(1);
    m_energies[0] = new double();
#endif
  }

  void reset_data(const pele::Array<double> *coords,
                  pele::Array<double> *gradient, pele::Array<double> *hessian) {
    m_coords = coords;
#ifdef _OPENMP
#pragma omp parallel
    { *m_energies[omp_get_thread_num()] = 0; }
#else
    *m_energies[0] = 0;
#endif
    m_gradient = gradient;
    m_hessian = hessian;
  }

  void insert_atom_pair(const size_t atom_i, const size_t atom_j,
                        const size_t isubdom) {
    pele::VecN<m_ndim, double> dr;
    const size_t xi_off = m_ndim * atom_i;
    const size_t xj_off = m_ndim * atom_j;
    m_dist->get_rij(dr.data(), m_coords->data() + xi_off,
                    m_coords->data() + xj_off);
    double r2 = 0;
    for (size_t k = 0; k < m_ndim; ++k) {
      r2 += dr[k] * dr[k];
    }
    double gij, hij;
    double dij = 0;
    if (m_radii.size() > 0) {
      dij = m_radii[atom_i] + m_radii[atom_j];
    }
#ifdef _OPENMP
    *m_energies[isubdom] +=
        m_interaction->energy_gradient_hessian(r2, &gij, &hij, dij);
#else
    *m_energies[0] +=
        m_interaction->energy_gradient_hessian(r2, &gij, &hij, dij);
#endif
    if (gij != 0) {
      for (size_t k = 0; k < m_ndim; ++k) {
        (*m_gradient)[xi_off + k] -= gij * dr[k];
        (*m_gradient)[xj_off + k] += gij * dr[k];
      }
    }
    // this part is copied from simple_pairwise_potential.h
    //(even more so than the rest)
    const size_t N = m_gradient->size();
    const size_t i1 = xi_off;
    const size_t j1 = xj_off;
    for (size_t k = 0; k < m_ndim; ++k) {
      // diagonal block - diagonal terms
      const double Hii_diag = (hij + gij) * dr[k] * dr[k] / r2 - gij;
      (*m_hessian)[N * (i1 + k) + i1 + k] += Hii_diag;
      (*m_hessian)[N * (j1 + k) + j1 + k] += Hii_diag;
      // off diagonal block - diagonal terms
      const double Hij_diag = -Hii_diag;
      (*m_hessian)[N * (i1 + k) + j1 + k] += Hij_diag;
      (*m_hessian)[N * (j1 + k) + i1 + k] += Hij_diag;
      for (size_t l = k + 1; l < m_ndim; ++l) {
        // diagonal block - off diagonal terms
        const double Hii_off = (hij + gij) * dr[k] * dr[l] / r2;
        (*m_hessian)[N * (i1 + k) + i1 + l] += Hii_off;
        (*m_hessian)[N * (i1 + l) + i1 + k] += Hii_off;
        (*m_hessian)[N * (j1 + k) + j1 + l] += Hii_off;
        (*m_hessian)[N * (j1 + l) + j1 + k] += Hii_off;
        // off diagonal block - off diagonal terms
        const double Hij_off = -Hii_off;
        (*m_hessian)[N * (i1 + k) + j1 + l] += Hij_off;
        (*m_hessian)[N * (i1 + l) + j1 + k] += Hij_off;
        (*m_hessian)[N * (j1 + k) + i1 + l] += Hij_off;
        (*m_hessian)[N * (j1 + l) + i1 + k] += Hij_off;
      }
    }
  }

  double get_energy() {
    double energy = 0;
    for (size_t i = 0; i < m_energies.size(); ++i) {
      energy += *m_energies[i];
    }
    return energy;
  }
};

/**
 * class which accumulates the energy one pair interaction at a time
 */
template <typename pairwise_interaction, typename distance_policy>
class NeighborAccumulator {
  // BasePotential
  PairwisePotentialInterface *m_pot;
  const static size_t m_ndim = distance_policy::_ndim;
  std::shared_ptr<pairwise_interaction> m_interaction;
  std::shared_ptr<distance_policy> m_dist;
  const pele::Array<double> m_coords;
  const pele::Array<double> m_radii;
  const double m_cutoff_sca;
  const pele::Array<short> m_include_atoms;

public:
  pele::Array<std::vector<size_t>> m_neighbor_indexes;
  pele::Array<std::vector<std::vector<double>>> m_neighbor_displacements;

  NeighborAccumulator(PairwisePotentialInterface *pot,
                      std::shared_ptr<pairwise_interaction> &interaction,
                      std::shared_ptr<distance_policy> &dist,
                      pele::Array<double> const &coords,
                      pele::Array<double> const &radii, const double cutoff_sca,
                      pele::Array<short> const &include_atoms)
      : m_pot(pot), m_interaction(interaction), m_dist(dist), m_coords(coords),
        m_radii(radii), m_cutoff_sca(cutoff_sca),
        m_include_atoms(include_atoms), m_neighbor_indexes(radii.size()),
        m_neighbor_displacements(radii.size()) {}

  void insert_atom_pair(const size_t atom_i, const size_t atom_j,
                        const size_t) {
    if (m_include_atoms[atom_i] && m_include_atoms[atom_j]) {
      std::vector<double> dr(m_ndim);
      std::vector<double> neg_dr(m_ndim);
      const size_t xi_off = m_ndim * atom_i;
      const size_t xj_off = m_ndim * atom_j;
      m_dist->get_rij(dr.data(), m_coords.data() + xi_off,
                      m_coords.data() + xj_off);
      double r2 = 0;
      for (size_t k = 0; k < m_ndim; ++k) {
        r2 += dr[k] * dr[k];
        neg_dr[k] = -dr[k];
      }
      const double dij = m_pot->get_non_additive_cutoff(atom_i, atom_j);
      const double r_S = m_cutoff_sca * dij;
      const double r_S2 = r_S * r_S;
      if (r2 <= r_S2) {
        m_neighbor_indexes[atom_i].push_back(atom_j);
        m_neighbor_indexes[atom_j].push_back(atom_i);
        m_neighbor_displacements[atom_i].push_back(dr);
        m_neighbor_displacements[atom_j].push_back(neg_dr);
      }
    }
  }
};

/**
 * class which accumulates the energy one pair interaction at a time
 */
template <typename pairwise_interaction, typename distance_policy>
class OverlapAccumulator {
  const static size_t m_ndim = distance_policy::_ndim;
  std::shared_ptr<pairwise_interaction> m_interaction;
  std::shared_ptr<distance_policy> m_dist;
  const pele::Array<double> m_coords;
  const pele::Array<double> m_radii;

public:
  std::vector<size_t> m_overlap_inds;

  OverlapAccumulator(std::shared_ptr<pairwise_interaction> &interaction,
                     std::shared_ptr<distance_policy> &dist,
                     pele::Array<double> const &coords,
                     pele::Array<double> const &radii)
      : m_interaction(interaction), m_dist(dist), m_coords(coords),
        m_radii(radii) {}

  void insert_atom_pair(const size_t atom_i, const size_t atom_j,
                        const size_t isubdom) {
    pele::VecN<m_ndim, double> dr;
    const size_t xi_off = m_ndim * atom_i;
    const size_t xj_off = m_ndim * atom_j;
    m_dist->get_rij(dr.data(), m_coords.data() + xi_off,
                    m_coords.data() + xj_off);
    double r2 = 0;
    for (size_t k = 0; k < m_ndim; ++k) {
      r2 += dr[k] * dr[k];
    }
    const double dij = m_radii[atom_i] + m_radii[atom_j];
    const double r_H2 = dij * dij;
    if (r2 <= r_H2) {
#pragma omp critical(add_overlap)
      {
        m_overlap_inds.push_back(atom_i);
        m_overlap_inds.push_back(atom_j);
      }
    }
  }
};

/**
 * Potential to loop over the list of atom pairs generated with the
 * cell list implementation in cell_lists.h.
 * This should also do the cell list construction and refresh, such that
 * the interface is the same for the user as with SimplePairwise.
 */
template <typename pairwise_interaction, typename distance_policy>
class CellListPotential : public PairwisePotentialInterface {
protected:
  const static size_t m_ndim = distance_policy::_ndim;
  pele::CellLists<distance_policy> m_cell_lists;
  std::shared_ptr<pairwise_interaction> m_interaction;
  std::shared_ptr<distance_policy> m_dist;
  const double m_radii_sca;
  bool exact_sum;

  // the following are used for the exact sum
  // We may need to write a more general case if other methods are added
  bool exact_gradient_initialized;
  std::shared_ptr<std::vector<xsum_small_accumulator>> exact_gradient;

  EnergyAccumulator<pairwise_interaction, distance_policy> *m_eAcc;
  EnergyGradientAccumulator<pairwise_interaction, distance_policy> *m_egAcc;
  EnergyGradientHessianAccumulator<pairwise_interaction, distance_policy>
      *m_eghAcc;
  EnergyHessianAccumulator<pairwise_interaction, distance_policy> *m_ehAcc;
  EnergyAccumulatorExact<pairwise_interaction, distance_policy> *m_eAccExact;
  EnergyGradientAccumulatorExact<pairwise_interaction, distance_policy>
      *m_egAccExact;
  EnergyGradientHessianAccumulatorExact<pairwise_interaction, distance_policy>
      *m_eghAccExact;

public:
  ~CellListPotential() {
    if (exact_sum) {
      delete m_eAccExact;
      delete m_egAccExact;
      delete m_eghAccExact;
    } else {
      delete m_eAcc;
      delete m_egAcc;
      delete m_eghAcc;
    }
  }

  CellListPotential(std::shared_ptr<pairwise_interaction> interaction,
                    std::shared_ptr<distance_policy> dist,
                    pele::Array<double> const &boxvec,
                    double ncellx_scale, const pele::Array<double> radii,
                    NonAdditiveCutoffCalculator cutoff_calculator,
                    const double radii_sca = 0.0, const bool balance_omp = true,
                    const bool exact_sum = false)
      : PairwisePotentialInterface(radii, cutoff_calculator),
        m_cell_lists(dist, boxvec, this->get_max_cutoff(), ncellx_scale, balance_omp),
        m_interaction(interaction), m_dist(dist), m_radii_sca(radii_sca),
        exact_gradient_initialized(false), exact_sum(exact_sum) {
    if (exact_sum) {
      std::cout
          << "WARNING: Exact sum is not being maintained and will be removed"
          << std::endl;
      m_eAccExact =
          new EnergyAccumulatorExact<pairwise_interaction, distance_policy>(
              m_interaction, m_dist, m_radii);
      m_egAccExact = new EnergyGradientAccumulatorExact<pairwise_interaction,
                                                        distance_policy>(
          m_interaction, m_dist, m_radii);
      m_eghAccExact =
          new EnergyGradientHessianAccumulatorExact<pairwise_interaction,
                                                    distance_policy>(
              m_interaction, m_dist, m_radii);
      std::cout << "Warning: exact sum not implemented for get_hessian"
                << std::endl;
      m_ehAcc =
          new EnergyHessianAccumulator<pairwise_interaction, distance_policy>(
              m_interaction, m_dist, m_radii);
    } else {
      m_eAcc = new EnergyAccumulator<pairwise_interaction, distance_policy>(
          interaction, dist, m_radii, cutoff_calculator);
      m_egAcc =
          new EnergyGradientAccumulator<pairwise_interaction, distance_policy>(
              interaction, dist, m_radii, cutoff_calculator);
      m_eghAcc = new EnergyGradientHessianAccumulator<pairwise_interaction,
                                                      distance_policy>(
          interaction, dist, m_radii, cutoff_calculator);
      m_ehAcc =
          new EnergyHessianAccumulator<pairwise_interaction, distance_policy>(
              m_interaction, m_dist, m_radii, cutoff_calculator);
    }
  }

  CellListPotential(std::shared_ptr<pairwise_interaction> interaction,
                    std::shared_ptr<distance_policy> dist,
                    pele::Array<double> const &boxvec, double rcut,
                    double ncellx_scale, const pele::Array<double> radii,
                    const double radii_sca = 0.0, const bool balance_omp = true,
                    const bool exact_sum = false,
                    const double non_additivity = 0.0)
      : PairwisePotentialInterface(radii, non_additivity),
        m_cell_lists(dist, boxvec, rcut, ncellx_scale, balance_omp),
        m_interaction(interaction), m_dist(dist), m_radii_sca(radii_sca),
        exact_gradient_initialized(false), exact_sum(exact_sum) {
    if (exact_sum) {
      std::cout
          << "WARNING: Exact sum is not being maintained and will be removed"
          << std::endl;
      m_eAccExact =
          new EnergyAccumulatorExact<pairwise_interaction, distance_policy>(
              m_interaction, m_dist, m_radii);
      m_egAccExact = new EnergyGradientAccumulatorExact<pairwise_interaction,
                                                        distance_policy>(
          m_interaction, m_dist, m_radii);
      m_eghAccExact =
          new EnergyGradientHessianAccumulatorExact<pairwise_interaction,
                                                    distance_policy>(
              m_interaction, m_dist, m_radii);
      std::cout << "Warning: exact sum not implemented for get_hessian"
                << std::endl;
      m_ehAcc =
          new EnergyHessianAccumulator<pairwise_interaction, distance_policy>(
              m_interaction, m_dist, m_radii);
    } else {
      m_eAcc = new EnergyAccumulator<pairwise_interaction, distance_policy>(
          interaction, dist, m_radii,
          NonAdditiveCutoffCalculator(non_additivity));
      m_egAcc =
          new EnergyGradientAccumulator<pairwise_interaction, distance_policy>(
              interaction, dist, m_radii,
              NonAdditiveCutoffCalculator(non_additivity));
      m_eghAcc = new EnergyGradientHessianAccumulator<pairwise_interaction,
                                                      distance_policy>(
          interaction, dist, m_radii,
          NonAdditiveCutoffCalculator(non_additivity));
      m_ehAcc =
          new EnergyHessianAccumulator<pairwise_interaction, distance_policy>(
              m_interaction, m_dist, m_radii,
              NonAdditiveCutoffCalculator(non_additivity));
    }
  }

  CellListPotential(std::shared_ptr<pairwise_interaction> interaction,
                    std::shared_ptr<distance_policy> dist,
                    pele::Array<double> const &boxvec, double rcut,
                    double ncellx_scale, const bool balance_omp = true,
                    const bool exact_sum = false)
      : m_cell_lists(dist, boxvec, rcut, ncellx_scale, balance_omp),
        m_interaction(interaction), m_dist(dist), m_radii_sca(0.0),
        exact_gradient_initialized(false), exact_sum(exact_sum) {
    if (exact_sum) {
      m_eAccExact =
          new EnergyAccumulatorExact<pairwise_interaction, distance_policy>(
              m_interaction, m_dist);
      m_egAccExact = new EnergyGradientAccumulatorExact<pairwise_interaction,
                                                        distance_policy>(
          m_interaction, m_dist);
      m_eghAccExact = new EnergyGradientHessianAccumulatorExact<
          pairwise_interaction, distance_policy>(m_interaction, m_dist);
      std::cout << "Warning: exact sum not implemented for get_hessian"
                << std::endl;
      m_ehAcc =
          new EnergyHessianAccumulator<pairwise_interaction, distance_policy>(
              m_interaction, m_dist, m_radii);
    } else {
      m_eAcc = new EnergyAccumulator<pairwise_interaction, distance_policy>(
          interaction, dist);
      m_egAcc =
          new EnergyGradientAccumulator<pairwise_interaction, distance_policy>(
              interaction, dist);
      m_eghAcc = new EnergyGradientHessianAccumulator<pairwise_interaction,
                                                      distance_policy>(
          interaction, dist);
      m_ehAcc =
          new EnergyHessianAccumulator<pairwise_interaction, distance_policy>(
              m_interaction, m_dist);
    }
  }

  virtual size_t get_ndim() { return m_ndim; }

  virtual double get_energy(Array<double> const &coords) {

    const size_t natoms = coords.size() / m_ndim;
    if (m_ndim * natoms != coords.size()) {
      throw std::runtime_error(
          "coords.size() is not divisible by the number of dimensions");
    }

    if (!std::isfinite(coords[0]) ||
        !std::isfinite(coords[coords.size() - 1])) {
      return NAN;
    }

    update_iterator(coords);
    if (exact_sum) {
      m_eAccExact->reset_data(&coords);
      auto looper = m_cell_lists.get_atom_pair_looper(*m_eAccExact);
      looper.loop_through_atom_pairs();
      return m_eAccExact->get_energy();
    } else {
      m_eAcc->reset_data(&coords);
      auto looper = m_cell_lists.get_atom_pair_looper(*m_eAcc);
      looper.loop_through_atom_pairs();
      return m_eAcc->get_energy();
    }
  }

  virtual double get_energy_gradient(Array<double> const &coords,
                                     Array<double> &grad) {
    const size_t natoms = coords.size() / m_ndim;
    if (m_ndim * natoms != coords.size()) {
      throw std::runtime_error(
          "coords.size() is not divisible by the number of dimensions");
    }
    if (coords.size() != grad.size()) {
      throw std::invalid_argument("the gradient has the wrong size");
    }

    if (!std::isfinite(coords[0]) ||
        !std::isfinite(coords[coords.size() - 1])) {
      grad.assign(NAN);
      return NAN;
    }

    update_iterator(coords);
    grad.assign(0.);
    if (exact_sum) {
      if (!exact_gradient_initialized) {
        exact_gradient =
            std::make_shared<std::vector<xsum_small_accumulator>>(grad.size());
        exact_gradient_initialized = true;
      } else {
        for (size_t i = 0; i < grad.size(); ++i) {
          xsum_small_init(&((*exact_gradient)[i]));
        }
      }
      m_egAccExact->reset_data(&coords, exact_gradient);
      auto looper = m_cell_lists.get_atom_pair_looper(*m_egAccExact);
      looper.loop_through_atom_pairs();

      // assign gradient to the output array
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
      for (size_t i = 0; i < grad.size(); ++i) {
        grad[i] = xsum_small_round(&((*exact_gradient)[i]));
      }
#else
      for (size_t i = 0; i < grad.size(); ++i) {
        grad[i] = xsum_small_round(&((*exact_gradient)[i]));
      }
#endif

      return m_egAccExact->get_energy();
    } else {
      return add_energy_gradient(coords, grad);
    }
  }

  /**
   * @brief Adds the gradient and hessian of the potential to the arrays
   *
   * @param coords
   * @param grad
   * @param hess
   * @return double
   */
  double add_energy_gradient_hessian(Array<double> const &coords,
                                     Array<double> &grad, Array<double> &hess) {
    m_eghAcc->reset_data(&coords, &grad, &hess);
    auto looper = m_cell_lists.get_atom_pair_looper(*m_eghAcc);
    looper.loop_through_atom_pairs();
    return m_eghAcc->get_energy();
  }

  /**
   * @brief Adds the gradient of the potential to the array
   *
   * @param coords
   * @param grad
   * @return double
   */
  double add_energy_gradient(Array<double> const &coords, Array<double> &grad) {
    if (!std::isfinite(coords[0]) ||
        !std::isfinite(coords[coords.size() - 1])) {
      grad.assign(NAN);
      return NAN;
    }
    m_egAcc->reset_data(&coords, &grad);
    auto looper = m_cell_lists.get_atom_pair_looper(*m_egAcc);
    looper.loop_through_atom_pairs();
    return m_egAcc->get_energy();
  }

  double add_energy_hessian(Array<double> const &coords, Array<double> &hess) {
    if (!std::isfinite(coords[0]) ||
        !std::isfinite(coords[coords.size() - 1])) {
      hess.assign(NAN);
      return NAN;
    }
    m_ehAcc->reset_data(&coords, &hess);
    auto looper = m_cell_lists.get_atom_pair_looper(*m_ehAcc);
    looper.loop_through_atom_pairs();
    return m_ehAcc->get_energy();
  }

  void add_hessian(Array<double> const &coords, Array<double> &hess) {
    if (!std::isfinite(coords[0]) ||
        !std::isfinite(coords[coords.size() - 1])) {
      hess.assign(NAN);
    }
    m_ehAcc->reset_data(&coords, &hess);
    auto looper = m_cell_lists.get_atom_pair_looper(*m_ehAcc);
    looper.loop_through_atom_pairs();
  }

  double get_energy_hessian(Array<double> const &coords, Array<double> &hess) {

    const size_t natoms = coords.size() / m_ndim;
    if (m_ndim * natoms != coords.size()) {
      throw std::runtime_error(
          "coords.size() is not divisible by the number of dimensions");
    }
    if (coords.size() != hess.size() * hess.size()) {
      throw std::invalid_argument("the hessian has the wrong size");
    }
    if (!std::isfinite(coords[0]) ||
        !std::isfinite(coords[coords.size() - 1])) {
      hess.assign(NAN);
      return NAN;
    }
    update_iterator(coords);
    hess.assign(0.);
    if (exact_sum) {
      throw std::runtime_error(
          "get hessian not implemented with exact summation");
    } else {
      return add_energy_hessian(coords, hess);
    }
    // assign hessian to the output array
  }

  virtual double get_energy_gradient_hessian(Array<double> const &coords,
                                             Array<double> &grad,
                                             Array<double> &hess) {
    const size_t natoms = coords.size() / m_ndim;
    if (m_ndim * natoms != coords.size()) {
      throw std::runtime_error(
          "coords.size() is not divisible by the number of dimensions");
    }
    if (coords.size() != grad.size()) {
      throw std::invalid_argument("the gradient has the wrong size");
    }
    if (hess.size() != coords.size() * coords.size()) {
      throw std::invalid_argument("the Hessian has the wrong size");
    }

    if (!std::isfinite(coords[0]) ||
        !std::isfinite(coords[coords.size() - 1])) {
      grad.assign(NAN);
      hess.assign(NAN);
      return NAN;
    }

    update_iterator(coords);
    grad.assign(0.);
    hess.assign(0.);
    if (exact_sum) {
      m_eghAccExact->reset_data(&coords, &grad, &hess);
      auto looper = m_cell_lists.get_atom_pair_looper(*m_eghAccExact);
      looper.loop_through_atom_pairs();
      return m_eghAccExact->get_energy();
    } else {
      return add_energy_gradient_hessian(coords, grad, hess);
    }
  }

  virtual void get_neighbors(
      pele::Array<double> const &coords,
      pele::Array<std::vector<size_t>> &neighbor_indexes,
      pele::Array<std::vector<std::vector<double>>> &neighbor_displacements,
      const double cutoff_factor = 1.0) {
    const size_t natoms = coords.size() / m_ndim;
    pele::Array<short> include_atoms(natoms, 1);
    get_neighbors_picky(coords, neighbor_indexes, neighbor_displacements,
                        include_atoms, cutoff_factor);
  }

  virtual void get_neighbors_picky(
      pele::Array<double> const &coords,
      pele::Array<std::vector<size_t>> &neighbor_indexes,
      pele::Array<std::vector<std::vector<double>>> &neighbor_displacements,
      pele::Array<short> const &include_atoms,
      const double cutoff_factor = 1.0) {
    const size_t natoms = coords.size() / m_ndim;
    if (m_ndim * natoms != coords.size()) {
      throw std::runtime_error(
          "coords.size() is not divisible by the number of dimensions");
    }
    if (natoms != include_atoms.size()) {
      throw std::runtime_error(
          "include_atoms.size() is not equal to the number of atoms");
    }
    if (m_radii.size() == 0) {
      throw std::runtime_error("Can't calculate neighbors, because the "
                               "used interaction doesn't use radii. ");
    }

    if (!std::isfinite(coords[0]) ||
        !std::isfinite(coords[coords.size() - 1])) {
      return;
    }

    update_iterator(coords);
    NeighborAccumulator<pairwise_interaction, distance_policy> accumulator(
        this, m_interaction, m_dist, coords, m_radii,
        (1 + m_radii_sca) * cutoff_factor, include_atoms);
    auto looper = m_cell_lists.get_atom_pair_looper(accumulator);

    looper.loop_through_atom_pairs();

    neighbor_indexes = accumulator.m_neighbor_indexes;
    neighbor_displacements = accumulator.m_neighbor_displacements;
  }

  virtual std::vector<size_t> get_overlaps(Array<double> const &coords) {
    const size_t natoms = coords.size() / m_ndim;
    if (m_ndim * natoms != coords.size()) {
      throw std::runtime_error(
          "coords.size() is not divisible by the number of dimensions");
    }
    if (m_radii.size() == 0) {
      throw std::runtime_error("Can't calculate neighbors, because the "
                               "used interaction doesn't use radii. ");
    }

    if (!std::isfinite(coords[0]) ||
        !std::isfinite(coords[coords.size() - 1])) {
      return std::vector<size_t>(2, 0);
    }

    update_iterator(coords);
    OverlapAccumulator<pairwise_interaction, distance_policy> accumulator(
        m_interaction, m_dist, coords, m_radii);
    auto looper = m_cell_lists.get_atom_pair_looper(accumulator);

    looper.loop_through_atom_pairs();

    return accumulator.m_overlap_inds;
  }

  virtual pele::Array<size_t> get_atom_order(Array<double> &coords) {
    const size_t natoms = coords.size() / m_ndim;
    if (m_ndim * natoms != coords.size()) {
      throw std::runtime_error(
          "coords.size() is not divisible by the number of dimensions");
    }

    if (!std::isfinite(coords[0]) ||
        !std::isfinite(coords[coords.size() - 1])) {
      return pele::Array<size_t>(0);
    }

    update_iterator(coords);
    return m_cell_lists.get_order(natoms);
  }

  virtual inline size_t get_ndim() const { return m_ndim; }

  virtual inline void get_rij(double *const r_ij, double const *const r1,
                              double const *const r2) const {
    return m_dist->get_rij(r_ij, r1, r2);
  }

  virtual inline double get_interaction_energy_gradient(double r2, double *gij,
                                                        size_t atom_i,
                                                        size_t atom_j) const {
    double energy = m_interaction->energy_gradient(
        r2, gij, get_non_additive_cutoff(atom_i, atom_j));
    *gij *= sqrt(r2);
    return energy;
  }

  virtual inline double
  get_interaction_energy_gradient_hessian(double r2, double *gij, double *hij,
                                          size_t atom_i, size_t atom_j) const {
    double energy = m_interaction->energy_gradient_hessian(
        r2, gij, hij, get_non_additive_cutoff(atom_i, atom_j));
    *gij *= sqrt(r2);
    return energy;
  }

  // Compute the maximum of all single atom norms
  virtual inline double compute_norm(pele::Array<double> const &x) {
    const size_t natoms = x.size() / m_ndim;

    double max_x = 0;
    for (size_t atom_i = 0; atom_i < natoms; ++atom_i) {
      double atom_x = 0;
#pragma GCC unroll 3
      for (size_t j = 0; j < m_ndim; ++j) {
        atom_x += x[atom_i * m_ndim + j] * x[atom_i * m_ndim + j];
      }
      max_x = std::max(max_x, atom_x);
    }
    return sqrt(max_x);
  }

protected:
  void update_iterator(Array<double> const &coords) {
    m_cell_lists.update(coords);
  }
};

} // namespace pele

#endif //#ifndef _PELE_CELL_LIST_POTENTIAL_H
