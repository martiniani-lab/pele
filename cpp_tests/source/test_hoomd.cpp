// #include <cassert>
// #include <cmath>
// #include <functional>
// #include <iostream>
// #include <memory>
// #include <stdexcept>
// #include <vector>

// #include <gtest/gtest.h>

// #include "cvode/cvode_ls.h"
// #include "hoomd/md/PotentialPair.h"
// #include "pele/harmonic.hpp"
// #include "pele/meta_pow.hpp"

// #include <hoomd/Initializers.h>
// #include "hoomd/md/EvaluatorPairLJ.h"
// #include "hoomd/md/PotentialPair.h"
// #include <hoomd/md/NeighborListTree.h>

// #include "test_utils.hpp"

// using pele::Array;
// using pele::pos_int_pow;
// using namespace hoomd::md;
// using namespace hoomd;
// typedef PotentialPair<EvaluatorPairLJ> PotentialPairLJ;

// static double const EPS = std::numeric_limits<double>::min();
// #define EXPECT_NEAR_RELATIVE(A, B, T) \
//   EXPECT_NEAR(A / (fabs(A) + fabs(B) + EPS), B / (fabs(A) + fabs(B) + EPS),
//   T)

// typedef std::function<std::shared_ptr<PotentialPairLJ>(
//     std::shared_ptr<SystemDefinition> sysdef,
//     std::shared_ptr<NeighborList> nlist)>
//     ljforce_creator;

// TEST(HOOMD, GET_FORCE_AND_ENERGY) {
//     // this 3-particle test subtly checks several conditions
//     // the particles are arranged on the x axis,  1   2   3
//     // such that 2 is inside the cutoff radius of 1 and 3, but 1 and 3 are
//     outside the cutoff
//     // of course, the buffer will be set on the neighborlist so that 3 is
//     included in it
//     // thus, this case tests the ability of the force summer to sum more than
//     one force on
//     // a particle and ignore a particle outside the radius

//     // periodic boundary conditions will be handled in another test

//     ljforce_creator lj_creator = [] (std::shared_ptr<SystemDefinition>
//     sys_def, std::shared_ptr<NeighborList> nlist) {
//       return std::shared_ptr<PotentialPairLJ>(new PotentialPairLJ(sys_def,
//       nlist));
//     };

//     // std::shared_ptr<ExecutionConfiguration> exec_conf =
//     std::shared_ptr<ExecutionConfiguration>(new
//     ExecutionConfiguration(ExecutionConfiguration::CPU));
//     // std::shared_ptr<SystemDefinition> sysdef_3(new SystemDefinition(3,
//     BoxDim(1000.0), 1, 0, 0, 0, 0, exec_conf));
//     // std::shared_ptr<ParticleData> pdata_3 = sysdef_3->getParticleData();
//     // pdata_3->setFlags(~PDataFlags(0));

//     // {
//     // ArrayHandle<Scalar4> h_pos(pdata_3->getPositions(),
//     access_location::host, access_mode::readwrite);
//     // h_pos.data[0].x = h_pos.data[0].y = h_pos.data[0].z = 0.0;
//     // h_pos.data[1].x = Scalar(pow(2.0,1.0/6.0)); h_pos.data[1].y =
//     h_pos.data[1].z = 0.0;
//     // h_pos.data[2].x = Scalar(2.0*pow(2.0,1.0/6.0)); h_pos.data[2].y =
//     h_pos.data[2].z = 0.0;
//     // }
//     // std::shared_ptr<NeighborListTree> nlist_3(new
//     NeighborListTree(sysdef_3, Scalar(1.3), Scalar(3.0)));
//     // std::shared_ptr<PotentialPairLJ> fc_3 = lj_creator(sysdef_3, nlist_3);
//     // fc_3->setRcut(0, 0, Scalar(1.3));

//     // // first test: setup a sigma of 1.0 so that all forces will be 0
//     // Scalar epsilon = Scalar(1.15);
//     // Scalar sigma = Scalar(1.0);
//     // Scalar alpha = Scalar(1.0);
//     // Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma,Scalar(12.0));
//     // Scalar lj2 = alpha * Scalar(4.0) * epsilon * pow(sigma,Scalar(6.0));
//     // fc_3->setParams(0,0,make_scalar2(lj1,lj2));

//     // // compute the forces
//     // fc_3->compute(0);
// }