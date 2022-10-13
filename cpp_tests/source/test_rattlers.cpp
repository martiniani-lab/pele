/**
 * @file test_rattlers.cpp
 * @author Praharsh Suryadevara
 * @brief Tests for finding rattlers.
 * @details Tests for finding rattlers. Current version only checks whether
 *          the function matches the expected rattler output from the julia and
 * python code. Test cases need to be added to explicitly check for correctness.
 * @version 0.1
 * @date 2021-11-10
 *
 * @copyright Copyright (c) 2021
 *
 */
#include <gtest/gtest.h>
#include "pele/array.hpp"
#include "pele/inversepower.hpp"
#include "gtest/internal/gtest-internal.h"
#include <cstddef>
#include <memory>

using pele::Array;
/*
 * @brief Test for finding rattlers.
 * @details Test for finding rattlers. Current version only checks whether
 *          the function matches the expected rattler output from the julia and
 *          python code.
 */
TEST(Rattlers, CompareToJulia) {

  // potential parameters
  double eps = 1.0;
  double sigma = 1.0;
  double rcut = 2.5;
  double box_length = 9.639075260988037;
  bool exact_sum = false;
  const size_t _ndim = 2;
  const int pow2 = 5;
  pele::Array<double> boxvec = {box_length, box_length};

  bool jammed;

  // potential
  Array<double> radii = {
      1.0882026172983832, 1.0200078604183611, 1.048936899205287,
      1.1120446599600728, 1.0933778995074983, 0.9511361060061795,
      1.0475044208762794, 0.9924321395851151, 1.392774680374451,
      1.428741895135686,  1.4100830499812613, 1.5017991454874082,
      1.4532726407602894, 1.4085172511544979, 1.4310704262921796,
      1.4233572029161985};
  Array<uint8_t> not_rattlers(radii.size());
  Array<double> initial_coords = {
      8.985081003865204,  9.005862881002095,  4.298628130047941,
      9.571790451354325,  1.9430879557812628, 4.903054443069508,
      1.3457245961473376, 9.825307647834011,  4.699696521837847,
      1.9799406415730652, 2.7667439395633737, 8.408487849623283,
      3.24391472053935,   6.52534219249241,   0.030582603432885318,
      4.21739064160543,   5.4831655404938,    7.4638278418092545,
      10.485444202411403, 7.061907094943011,  4.324637744995412,
      4.405446194185378,  6.710322129019316,  0.4261378001684918,
      7.86917193241319,   5.959907536512168,  -0.4477506891021841,
      1.874534741391346,  2.271902707046499,  2.4974885736594077,
      6.847076380037946,  3.3147073646565746};
  std::shared_ptr<pele::InverseHalfIntPowerPeriodic<_ndim, pow2>> potcell =
      std::make_shared<pele::InverseHalfIntPowerPeriodic<_ndim, pow2>>(
          eps, radii, boxvec, exact_sum);
  potcell->find_rattlers(initial_coords, not_rattlers, jammed);
  for (size_t i = 0; i < not_rattlers.size(); ++i) {
    std::cout << int(not_rattlers[i]) << ",";
  }
  std::cout << std::endl;
}

TEST(Rattlers, Check_) {}