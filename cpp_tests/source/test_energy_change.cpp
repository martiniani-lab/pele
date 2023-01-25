#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <vector>

#include "pele/array.hpp"
#include "pele/base_potential.hpp"
#include "pele/inversepower.hpp"
#include "pele/lbfgs.hpp"
#include "pele/pairwise_potential_interface.hpp"

using namespace pele;

class EnergyChangeSimplePairwise : public ::testing::Test {
 public:
  std::shared_ptr<PairwisePotentialInterface> potential;
  Array<double> quenched_coordinates;
  static constexpr size_t dim = 2;

  std::vector<size_t> neighboring_indices;

 protected:
  static constexpr int pow = 2;
  double eps;
  double non_additivity;
  bool exact_sum;
  Array<double> boxvec;
  Array<double> radii;

  void SetUp() {
    size_t n_particles = 32;
    eps = 1.0;
    non_additivity = 0.2;
    exact_sum = false;

    double dmin_by_dmax = 0.449;
    double d_mean = 1.0;
    double phi = 1.2;

    BerthierDistribution3d berthier_dist =
        BerthierDistribution3d(dmin_by_dmax, d_mean);
    radii = berthier_dist.sample(n_particles);

    double box_length = get_box_length(radii, dim, phi);
    boxvec = {box_length, box_length};

    Array<double> coordinates =
        generate_random_coordinates(box_length, n_particles, dim);

    potential = std::make_shared<InverseIntPowerPeriodic<dim, pow>>(
        eps, radii, boxvec, exact_sum, non_additivity);

    std::shared_ptr<BasePotential> potential_base =
        static_pointer_cast<BasePotential>(potential);
    LBFGS lbfgs = LBFGS(potential_base, coordinates);
    lbfgs.run();
    quenched_coordinates = lbfgs.get_x().copy();

    Array<std::vector<size_t>> neighbor_indices_cache;
    Array<std::vector<std::vector<double>>> neighbor_distances_cache;
    potential->get_neighbors(quenched_coordinates, neighbor_indices_cache,
                             neighbor_distances_cache);

    // Search indices for two particles that are neighbors
    bool found;
    for (std::size_t i = 0; i < n_particles; ++i) {
      // Automatically takes care of case when particle has no neighbors
      // (i.e. neighbor_indices_cache[i].size() == 0)
      for (std::size_t j = 0; j < neighbor_indices_cache[i].size(); ++j) {
        neighboring_indices = {i, neighbor_indices_cache[i][j]};
        return;
      }
    }
  }

  void test_energy_change() {
    double old_energy = potential->get_energy(quenched_coordinates);
    double delta = 0.1;

    // Calculate whether the change is correctly calculated
    Array<double> new_coordinates = quenched_coordinates.copy();

    // Change them in opposite directions
    // to make sure there is an energy change
    // due to the two particles themselves
    new_coordinates[neighboring_indices[0] * dim] += delta;
    new_coordinates[neighboring_indices[1] * dim] -= delta;

    // Changed ind
    double new_energy = potential->get_energy(new_coordinates);

    double energy_change = potential->get_energy_change(
        quenched_coordinates, new_coordinates, neighboring_indices);
    ASSERT_NEAR(new_energy, old_energy + energy_change, 1e-9);
  }
};

class EnergyChangeCellLists : public EnergyChangeSimplePairwise {
 private:
 public:
  void SetUp() {
    double ncellx_scale = 1.0;
    EnergyChangeSimplePairwise::SetUp();
    // Make the same potential but with cell lists
    potential = std::make_shared<InverseIntPowerPeriodicCellLists<dim, pow>>(
        A eps, radii, boxvec, ncellx_scale, exact_sum, non_additivity);
  }
};

TEST_F(EnergyChangeSimplePairwise, TestEnergyChange) { test_energy_change(); }

TEST_F(EnergyChangeCellLists, TestEnergyChange) { test_energy_change(); }