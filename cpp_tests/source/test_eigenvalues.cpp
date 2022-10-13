#include "pele/array.hpp"
#include "pele/eigen_interface.hpp"
#include "pele/inversepower.hpp"
#include "pele/lbfgs.hpp"
#include "pele/lowest_eig_potential.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace Eigen;
using namespace pele;


void compute_lowest_eigenvalue(MatrixXd m) {
  // check whether the lowest eigenvalue is negative
}

void setup() {
  static const size_t _ndim = 3;
  size_t nr_particles;
  size_t nr_dof;
  double eps;
  double sca;
  double rsca;
  double energy;
  //////////// parameters for inverse power potential at packing fraction 0.7 in
  ///3d
  //////////// as generated from code params.py in basinerror library

  pele::Array<double> x;
  pele::Array<double> hs_radii;
  pele::Array<double> radii;
  pele::Array<double> xfinal;
  pele::Array<double> boxvec;
  double power = 2.5;
  double k;
#ifdef _OPENMP
  omp_set_num_threads(1);
#endif
  // phi = 0.7
  nr_particles = 32;
  nr_dof = nr_particles * _ndim;
  eps = 1.0;
  double box_length = 7.240260952504541;
  boxvec = {box_length, box_length, box_length};
  radii = {1.08820262, 1.02000786, 1.0489369,  1.11204466, 1.0933779,
           0.95113611, 1.04750442, 0.99243214, 0.99483906, 1.02052993,
           1.00720218, 1.07271368, 1.03805189, 1.00608375, 1.02219316,
           1.01668372, 1.50458554, 1.38563892, 1.42191474, 1.3402133,
           1.22129071, 1.4457533,  1.46051053, 1.34804845, 1.55888282,
           1.2981944,  1.4032031,  1.38689713, 1.50729455, 1.50285511,
           1.41084632, 1.42647138};
  x = {2.60293101, 3.16422539, 5.05103191, 0.43604813, 4.82756501, 4.85559318,
       1.52322464, 0.93346004, 2.28378357, 2.63336089, 4.12837341, 3.17558941,
       7.15608451, 0.73883106, 1.51232222, 1.167923,   4.72867471, 1.8338973,
       3.37621168, 1.76970507, 1.15098127, 0.79914482, 4.7519975,  1.00048063,
       1.4233076,  2.66966646, 5.94420522, 0.70303858, 6.06693979, 0.69577755,
       7.06982134, 3.393157,   7.07200517, 4.3792394,  5.35246123, 0.28372984,
       2.04759621, 0.87025447, 2.14413231, 0.85961967, 2.3022812,  2.99937218,
       0.46444461, 5.01367885, 4.10234238, 1.92148917, 3.78845245, 0.68015381,
       4.17000292, 6.72834697, 2.30652235, 4.83222531, 0.95425092, 5.18639589,
       2.09537563, 1.32635327, 4.2465067,  0.14558388, 6.00174213, 0.03399647,
       4.9075686,  1.95492819, 5.32299657, 6.96649615, 1.80103767, 4.17152945,
       4.28653808, 4.14325313, 1.61516923, 6.89815147, 3.23730442, 6.12821966,
       5.06441248, 2.15352114, 5.89210858, 2.87080503, 6.37941707, 4.20856728,
       6.38399411, 5.01410943, 5.25103024, 3.62971935, 6.92229501, 4.66265709,
       3.06882116, 4.39044511, 0.13896376, 2.18348037, 4.77982869, 2.10023757,
       4.47459298, 3.10439728, 0.98086758, 2.15964188, 4.12669469, 4.27807298};

  double ncellsx_scale = get_ncellx_scale(radii, boxvec, 1);
  std::shared_ptr<pele::InversePowerPeriodicCellLists<_ndim>> potcell =
      std::make_shared<pele::InversePowerPeriodicCellLists<_ndim>>(
          power, eps, radii, boxvec, ncellsx_scale);

  pele::Array<double> g = pele::Array<double>(x.size());
  potcell->get_energy_gradient(x, g);
  std::cout << g << "\n";
  pele::LowestEigPotential lep = pele::LowestEigPotential(potcell, x, _ndim);

  std::shared_ptr<pele::LowestEigPotential> lep_ptr =
      std::make_shared<pele::LowestEigPotential>(lep);

  std::shared_ptr<pele::LBFGS> lbfgs =
      std::shared_ptr<pele::LBFGS>(new pele::LBFGS(lep_ptr, x, 1e-6, 10));
  lep_ptr->set_x_opt(lbfgs->get_x());
  lbfgs->run(100);

  double lowesteigenvalue = lbfgs->get_f();

  pele::Array<double> hess = pele::Array<double>(x.size() * x.size());
  potcell->get_energy_gradient_hessian(x, g, hess);

  std::cout << hess << "\n";

  Eigen::MatrixXd hess_eig = Eigen::MatrixXd(x.size(), x.size());

  eig_mat_eq_pele(hess_eig, hess);

  std::cout << hess_eig << "\n";

  std::cout << hess_eig.eigenvalues().real().minCoeff() << "eigenvalues h \n";

  std::cout << lowesteigenvalue << "lowest eigenvalue from pele \n";
}


TEST(EIG, sym) { setup(); }
