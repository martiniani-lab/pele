#ifndef PYGMIN_PAIRWISE_POTENTIAL_INTERFACE_H
#define PYGMIN_PAIRWISE_POTENTIAL_INTERFACE_H

#include "array.hpp"
#include "base_potential.hpp"
#include "utils.hpp"
#include <bits/stdc++.h>
#include <cassert>
#include <cstddef>
#include <set>

namespace pele {

class PairwisePotentialInterface : public BasePotential {
protected:
  const Array<double> m_radii;

  // Data members for finding rattlers close to minima
  Array<bool> m_not_rattlers; // true if the atom is a rattler
  bool m_jammed;              // true if the system is jammed

  double sum_radii(const size_t atom_i, const size_t atom_j) const {
    if (m_radii.size() == 0) {
      return 0;
    } else {
      return m_radii[atom_i] + m_radii[atom_j];
    }
  }

public:
  PairwisePotentialInterface() : m_radii(0) {
    // rattler finding at minima not implemented when radii are not explicitly
    // set
  }
  PairwisePotentialInterface(pele::Array<double> const &radii)
      : m_radii(radii.copy()), m_not_rattlers(radii.size(), false),
        m_jammed(false) {
    if (radii.size() == 0) {
      throw std::runtime_error(
          "PairwisePotentialInterface: illegal input: radii");
    }
  }

  virtual ~PairwisePotentialInterface() {}

  /**
   * Return the radii (if the interaction potential actually uses radii).
   */
  virtual pele::Array<double> get_radii() { return m_radii.copy(); }

  /**
   * Return the number of dimensions (box dimensions).
   * Ideally this should be overloaded.
   */
  virtual inline size_t get_ndim() const {
    throw std::runtime_error(
        "PairwisePotentialInterface::get_ndim must be overloaded");
  }

  /**
   * Return the distance as measured by the distance policy.
   */
  virtual inline void get_rij(double *const r_ij, double const *const r1,
                              double const *const r2) const {
    throw std::runtime_error(
        "PairwisePotentialInterface::get_rij must be overloaded");
  }

  /**
   * Return energy_gradient of interaction.
   */
  virtual inline double get_interaction_energy_gradient(double r2, double *gij,
                                                        size_t atom_i,
                                                        size_t atom_j) const {
    throw std::runtime_error("PairwisePotentialInterface::get_interaction_"
                             "energy_gradient must be overloaded");
  }

  /**
   * Return gradient and Hessian of interaction.
   */
  virtual inline double
  get_interaction_energy_gradient_hessian(double r2, double *gij, double *hij,
                                          size_t atom_i, size_t atom_j) const {
    throw std::runtime_error("PairwisePotentialInterface::get_interaction_"
                             "energy_gradient_hessian must be overloaded");
  }

  /**
   * Return lists of neighbors considering only certain atoms.
   */
  virtual void get_neighbors_picky(
      pele::Array<double> const &coords,
      pele::Array<std::vector<size_t>> &neighbor_indss,
      pele::Array<std::vector<std::vector<double>>> &neighbor_distss,
      pele::Array<short> const &include_atoms,
      const double cutoff_factor = 1.0) {
    throw std::runtime_error(
        "PairwisePotentialInterface::get_neighbors_picky must be overloaded");
  }

  /**
   * Return lists of neighbors.
   */
  virtual void
  get_neighbors(pele::Array<double> const &coords,
                pele::Array<std::vector<size_t>> &neighbor_indss,
                pele::Array<std::vector<std::vector<double>>> &neighbor_distss,
                const double cutoff_factor = 1.0) {
    throw std::runtime_error(
        "PairwisePotentialInterface::get_neighbors must be overloaded");
  }

  /**
   * Return linearized vector of overlapping atoms.
   *
   * Every two entries in the resulting vector correspond to a pair of
   * overlapping atoms.
   */
  virtual std::vector<size_t> get_overlaps(Array<double> const &coords) {
    throw std::runtime_error(
        "PairwisePotentialInterface::get_overlaps must be overloaded");
  }

  /**
   * Return a computationally sensible order for sorting the atoms
   *
   * This is primarily used for better caching when using cell lists.
   * If sorting atoms is not useful, it returns an empty Pele-Array.
   */
  virtual pele::Array<size_t> get_atom_order(Array<double> &coords) {
    return pele::Array<size_t>(0);
  }

  pele::Array<bool> get_rattlers() const { return m_not_rattlers.copy(); }

  /**
   * @brief Finds the rattlers in the Found minimum and saves them
   * @param coords The coordinates of the found minimum. important: Needs to be
   * at the minimum
   * To look at a readable implementation, see the julia code in
   * check_same_structure.jl in the basinerrror library. (It's also probably
   * faster since this isn't vectorized)
   */
  void find_rattlers(pele::Array<double> const &minimum_coords) {

    m_not_rattlers.assign(true);
    size_t dim = get_ndim();

    // Not implemented for higher dimensions
    assert(dim == 2) size_t n_particles = minimum_coords.size() / dim;

    size_t zmin = dim + 1

                  // check if the radii /coordinate array sizes are consistent
                  // with the dimensionality
                  assert(n_particles == m_radii.size());

    // get neighbors of atoms
    pele::Array<std::vector<size_t>> neighbor_indss;
    pele::Array<std::vector<std::vector<double>>> neighbor_distss;
    get_neighbors(minimum_coords, neighbor_indss, neighbor_distss);

    size_t n_rattlers = 0;
    std::set<size_t> rattler_check_list = std::set<size_t>();
    for (size_t i = 0; i < n_particles; ++i) {
      rattler_check_list.insert(i);
    }
    std::set<size_t> current_check_list;

    bool found_rattler = false;
    size_t no_of_neighbors = 0;
    // allocation for the index of atom i in the neighbors of atom j
    size_t i_in_j;
    std::vector<size_t> neighbors_of_atom;

    while (rattler_check_list.size() > 0) {
      current_check_list = rattler_check_list;
      rattler_check_list.clear();
      for (size_t atomi : rattler_check_list) {
        found_rattler = false;
        no_of_neighbors = neighbor_indss[atomi].size();
        if (no_of_neighbors < zmin) {
          found_rattler = true;
        } else {
          found_rattler = origin_in_hull_2d(neighbor_distss[atomi]);
        }
        if (found_rattler) {
          m_not_rattlers[atomi] = false;
          n_rattlers++;

          // remove atom from the list of atoms to check if present in the check
          // list
          if (current_check_list.find(atomi) != current_check_list.end()) {
            current_check_list.erase(atomi);
          }

          for (size_t neighbor_index : neighbor_indss[atomi]) {
            rattler_check_list.insert(neighbor_index);
            neighbors_of_atom = neighbor_indss[neighbor_index];
            for (size_t j = 0; j < neighbors_of_atom.size(); ++j) {
              i_in_j = neighbors_of_atom[j];
              if (current_check_list.find(i_in_j) != current_check_list.end()) {
                current_check_list.erase(i_in_j);
              }
              // remove the displacements of the rattlers from neigbhor_distss
              for (size_t k = 0; k < neighbor_distss[i_in_j].size(); ++k) {
                if (neighbor_distss[i_in_j][k][0] == atomi) {
                  neighbor_distss[i_in_j].erase(
                      neighbor_distss[i_in_j].begin() + k);
                  break;
                }
              }
            }
          }
        }
      }
    } // end while

    size_t n_stable_particles = n_particles - n_rattlers;

    size_t total_contacts = 0;
    for (size_t i = 0; i < n_particles; ++i) {
      total_contacts += neighbor_indss[i].size();
    }

    int minimum_jammed_contacts = 2 * ((n_stable_particles - 1) * dim + 1);

    if ((total_contacts < minimum_jammed_contacts) ||
        (n_stable_particles < dim)) {
      m_jammed = false;
    }
    m_jammed = true;
  }
};

} // namespace pele

#endif // #ifndef PYGMIN_PAIRWISE_POTENTIAL_INTERFACE_H
