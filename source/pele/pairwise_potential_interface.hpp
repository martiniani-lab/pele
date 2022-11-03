#ifndef PYGMIN_PAIRWISE_POTENTIAL_INTERFACE_H
#define PYGMIN_PAIRWISE_POTENTIAL_INTERFACE_H

#include "array.hpp"
#include "base_potential.hpp"
#include "utils.hpp"
#include <bits/stdc++.h>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <set>
#include <vector>
using namespace std;

namespace pele {

class PairwisePotentialInterface : public BasePotential {
protected:
  const Array<double> m_radii;
  const double non_additivity;



public:
  PairwisePotentialInterface() : m_radii(0), non_additivity(0) {}
  PairwisePotentialInterface(pele::Array<double> const &radii,
                             double non_addivity = 0)
      : m_radii(radii.copy()), non_additivity(non_addivity) {
    if (radii.size() == 0) {
      throw std::runtime_error(
          "PairwisePotentialInterface: illegal input: radii");
    }
    const auto [min_radius_address, max_radius_address] = std::minmax_element(radii.begin(),
                                                            radii.end());
    size_t min_radius_index  = std::distance(radii.begin(), min_radius_address);
    size_t max_radius_index  = std::distance(radii.begin(), max_radius_address);

    auto dij = get_dij(min_radius_index,max_radius_index);

    if (dij < 0) {
      std::cout << "dij = " << dij << std::endl;
      std::cout << "min_radius = " << m_radii[min_radius_index] << std::endl;
      std::cout << "max_radius = " << m_radii[max_radius_index] << std::endl;
      throw std::runtime_error(
          "cutoff(dij) is negative");
    }
  }

  virtual ~PairwisePotentialInterface() {}

  /**
   * Return the radii (if the interaction potential actually uses radii).
   */
  virtual pele::Array<double> get_radii() { return m_radii.copy(); }

  /*
   * Finds the interaction distance between two particles, i and j. defaults to
   * radii[i] + radii[j] when epsilon is zero
   */
  inline double get_dij(const std::size_t atom_i,
                        const std::size_t atom_j) const {
    if (m_radii.size() == 0) {
      return 0;
    } else {
      // uses the diameters being twice the radii
      return (m_radii[atom_i] + m_radii[atom_j]) *
             (1 - 2 * non_additivity * abs(m_radii[atom_i] - m_radii[atom_j]));
    }
  }

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
  virtual inline void get_rij(double *const, double const *const,
                              double const *const) const {
    throw std::runtime_error(
        "PairwisePotentialInterface::get_rij must be overloaded");
  }

  /**
   * Return energy_gradient of interaction.
   */
  virtual inline double get_interaction_energy_gradient(double, double *,
                                                        size_t,
                                                        size_t) const {
    throw std::runtime_error("PairwisePotentialInterface::get_interaction_"
                             "energy_gradient must be overloaded");
  }

  /**
   * Return gradient and Hessian of interaction.
   */
  virtual inline double
  get_interaction_energy_gradient_hessian(double, double *, double *,
                                          size_t, size_t) const {
    throw std::runtime_error("PairwisePotentialInterface::get_interaction_"
                             "energy_gradient_hessian must be overloaded");
  }

  /**
   * Return lists of neighbors considering only certain atoms.
   */
  virtual void get_neighbors_picky(
      pele::Array<double> const &,
      pele::Array<std::vector<size_t>> &,
      pele::Array<std::vector<std::vector<double>>> &,
      pele::Array<short> const &,
      const double cutoff_factor = 1.0) {
    throw std::runtime_error(
        "PairwisePotentialInterface::get_neighbors_picky must be overloaded");
  }

  /**
   * Prints neighbor lists and displacements.
   */
  inline void print_neighbor_index_and_displacements(
      pele::Array<std::vector<size_t>> &neighbor_indss,
      pele::Array<std::vector<std::vector<double>>> &neighbor_distss) {
    for (size_t i = 0; i < neighbor_indss.size(); i++) {
      cout << "neighbor list for atom " << i << ": ";
      for (size_t j = 0; j < neighbor_indss[i].size(); j++) {
        cout << "neighbor " << neighbor_indss[i][j] << ": ";
        cout << endl;
        cout << "displacement: ";
        for (size_t k = 0; k < neighbor_distss[i][j].size(); k++) {
          cout << neighbor_distss[i][j][k] << " ";
        }
        cout << endl;
      }
    }
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

  /**
   * @brief Finds the rattlers in the found minimum and saves them.
   *
   * @param coords The coordinates of the found minimum. important: Needs to be
   *               at the minimum
   * @param not_rattlers whether a particle is a rattler or not
   * @param jammed whether the system is jammed or not
   * To look at a readable implementation, see the julia code in
   * check_same_structure.jl in the basinerrror library. (It's also probably
   * faster since this isn't vectorized)
   * @WARNING: This function segfaults at destructors when called from python.
   */
  bool find_rattlers(pele::Array<double> const &minimum_coords,
                     Array<uint8_t> not_rattlers, bool &jammed) {
    cout << "find rattlers started " << endl;
    not_rattlers.assign(1);
    size_t dim = get_ndim();

    // Not implemented for higher dimensions
    assert(dim == 2);
    size_t n_particles = minimum_coords.size() / dim;

    size_t zmin = dim + 1;

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
    size_t i = 0;
    while (rattler_check_list.size() > 0) {
      i++;
      print_neighbor_index_and_displacements(neighbor_indss, neighbor_distss);
      current_check_list = rattler_check_list;
      rattler_check_list.clear();

      for (size_t atom_i : current_check_list) {
        found_rattler = false;
        no_of_neighbors = neighbor_indss[atom_i].size();
        if (no_of_neighbors < zmin) {
          found_rattler = true;
        } else {
          cout << "atom " << atom_i << " has " << no_of_neighbors
               << " neighbors" << endl;
          cout << "atom_i" << atom_i << " neighbor distsss size"
               << neighbor_distss[atom_i].size() << endl;
          found_rattler = origin_in_hull_2d(neighbor_distss[atom_i]);
        }
        if (found_rattler) {
          // atom i is a rattler
          not_rattlers[atom_i] = 0;
          n_rattlers++;

          // remove atom from the list of atoms to check if present in the check
          // list
          if (rattler_check_list.find(atom_i) != rattler_check_list.end()) {
            rattler_check_list.erase(atom_i);
          }

          for (size_t atom_j : neighbor_indss[atom_i]) {
            rattler_check_list.insert(atom_j);

            // find index of atom i in the neighbors of atom j
            i_in_j = 0;
            for (size_t k = 0; k < neighbor_indss[atom_j].size(); ++k) {
              if (neighbor_indss[atom_j][k] == atom_i) {
                i_in_j = k;
                break;
              }
            }

            // remove atom i from the list of neighbors of atom j
            neighbor_indss[atom_j].erase(neighbor_indss[atom_j].begin() +
                                         i_in_j);
          }
        }
      }
    } // end while
    size_t n_stable_particles = n_particles - n_rattlers;

    size_t total_contacts = 0;
    for (size_t i = 0; i < n_particles; ++i) {
      total_contacts += neighbor_indss[i].size();
    }
    // print out not_rattlers
    cout << "not_rattlers: ";
    for (size_t i = 0; i < n_particles; ++i) {
      cout << int(not_rattlers[i]) << " ";
    }
    cout << endl;

    int minimum_jammed_contacts = 2 * ((n_stable_particles - 1) * dim + 1);

    if ((total_contacts < minimum_jammed_contacts) ||
        (n_stable_particles < dim)) {
      jammed = false;
    }
    jammed = true;
    cout << "find rattlers finished " << endl;
    return jammed;
  }
};

} // namespace pele

#endif // #ifndef PYGMIN_PAIRWISE_POTENTIAL_INTERFACE_H
