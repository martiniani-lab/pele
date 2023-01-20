#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>

#include "pele/array.hpp"
#include "pele/inversepower.hpp"
#include "pele/lbfgs.hpp"

using namespace pele;

class EnergyChangeSimplePairwise : public ::testing::Test {
 public:
  std::shared_ptr<BasePotential> simple_pairwise_inverse_power;
  Array<double> quenched_coordinates;
  static constexpr size_t dim = 2;
  void SetUp() {
    constexpr int pow = 2;

    size_t n_particles = 32;
    double eps = 1.0;
    double non_additivity = 0.2;
    bool exact_sum = false;

    double cutoff_factor = 1.25;

    double dmin_by_dmax = 0.449;
    double d_mean = 1.0;
    BerthierDistribution3d berthier_dist =
        BerthierDistribution3d(dmin_by_dmax, d_mean);
    Array<double> radii = berthier_dist.sample(n_particles);

    double phi = 1.2;
    double box_length = get_box_length(radii, dim, phi);
    Array<double> boxvec = {box_length, box_length};
    double ncellsx_scale = 1.0;

    Array<double> coordinates =
        generate_random_coordinates(box_length, n_particles, dim);

    simple_pairwise_inverse_power =
        std::make_shared<InverseIntPowerPeriodic<dim, pow>>(
            eps, radii, boxvec, exact_sum, non_additivity);

    LBFGS lbfgs = LBFGS(simple_pairwise_inverse_power, coordinates);
    lbfgs.run();
    quenched_coordinates = lbfgs.get_x().copy();
  }
};

TEST_F(EnergyChangeSimplePairwise, TestEnergyChange) {
  double old_energy =
      simple_pairwise_inverse_power->get_energy(quenched_coordinates);

  double delta = 0.1;

  // Calculate whether the change is correctly calculated
  std::vector<size_t> changed_indices = {2, 5};
  Array<double> new_coordinates = quenched_coordinates.copy();

  //
  for (size_t i : changed_indices) {
    new_coordinates[i * dim] += delta;
  }
  double new_energy =
      simple_pairwise_inverse_power->get_energy(new_coordinates);

  double energy_change = simple_pairwise_inverse_power->get_energy_change(
      quenched_coordinates, new_coordinates, changed_indices);
  ASSERT_NEAR(new_energy, old_energy + energy_change, 1e-9);
}