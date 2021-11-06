#include "pele/cell_lists.hpp"
#include "pele/lbfgs.hpp"
#include "pele/cvode.hpp"
#include "pele/mxopt.hpp"
#include "pele/utils.hpp"
#include "pele/inversepower.hpp"
#include "pele/gradient_descent.hpp"
#include "pele/mxd_end_only.hpp"

//#include "pele/mxd_end_only.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include "pele/array.hpp"

using pele::Array;
using std::cout;



// Test for whether the switch works for mixed descent
// TODO: seed the random number generator
TEST(CVODE, TEST_256_RUN){
    static const size_t _ndim = 2;
    size_t n_particles;
    size_t n_dof;

    double power;
    double eps;

    double phi;

    pele::Array<double> x;
    pele::Array<double> radii;
    pele::Array<double> xfinal;
    pele::Array<double> boxvec;

    eps =1.0;
    power = 2.5;
    const int pow2 = 5;

    n_particles = 256;
    n_dof = n_particles * _ndim;
    phi = 0.9;

    int n_1 = n_particles/2;
    int n_2 = n_particles - n_1;

    double r_1 = 1.0;
    double r_2 = 1.0;
    double r_std1 = 0.05;
    double r_std2 = 0.05;

    radii = pele::generate_radii(n_1, n_2, r_1, r_2, r_std1, r_std2);

    double box_length = pele::get_box_length(radii, _ndim, phi);

    boxvec = {box_length, box_length};



    x = pele::generate_random_coordinates(box_length, n_particles, _ndim);
    
    



    
//////////// parameters for inverse power potential at packing fraction 0.7 in 3d
    //////////// as generated from code params.py in basinerror library

    
    double k;
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    bool exact_sum = false;
    double ncellsx_scale = get_ncellx_scale(radii, boxvec,
                                            1);
    // std::shared_ptr<pele::InversePowerPeriodicCellLists<_ndim> > potcell = std::make_shared<pele::InversePowerPeriodicCellLists<_ndim>>(power, eps, radii, boxvec, ncellsx_scale);
    // std::shared_ptr<pele::InversePowerPeriodic<_ndim> > pot = std::make_shared<pele::InversePowerPeriodic<_ndim> >(power, eps, radii, boxvec);
    std::shared_ptr<pele::InverseHalfIntPowerPeriodicCellLists<_ndim, pow2>> potcell = std::make_shared<pele::InverseHalfIntPowerPeriodicCellLists<_ndim, pow2>>(eps, radii, boxvec, ncellsx_scale, exact_sum);
    
    potcell->get_energy(x);
    
    pele::CVODEBDFOptimizer optimizer(potcell, x);
    optimizer.run();
    std::cout << optimizer.get_nfev() << "nfev \n";
    std::cout << optimizer.get_nhev() << "nhev \n";
    std::cout << optimizer.get_rms() << "rms \n";
    std::cout << optimizer.get_niter() << "\n";
}

// Test for whether the switch works for mixed descent
// TODO: seed the random number generator
TEST(MXD_END_ONLY, TEST_256_RUN){
    static const size_t _ndim = 2;
    size_t n_particles;
    size_t n_dof;

    double power;
    double eps;

    double phi;

    pele::Array<double> x;
    pele::Array<double> radii;
    pele::Array<double> xfinal;
    pele::Array<double> boxvec;

    eps =1.0;
    power = 2.5;
    const int pow2 = 5;

    n_particles = 256;
    n_dof = n_particles * _ndim;
    phi = 0.9;

    int n_1 = n_particles/2;
    int n_2 = n_particles - n_1;

    double r_1 = 1.0;
    double r_2 = 1.0;
    double r_std1 = 0.05;
    double r_std2 = 0.05;

    radii = pele::generate_radii(n_1, n_2, r_1, r_2, r_std1, r_std2);

    double box_length = pele::get_box_length(radii, _ndim, phi);

    boxvec = {box_length, box_length};



    x = pele::generate_random_coordinates(box_length, n_particles, _ndim);
    
    



    
//////////// parameters for inverse power potential at packing fraction 0.7 in 3d
    //////////// as generated from code params.py in basinerror library

    
    double k;
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    bool exact_sum = false;
    double ncellsx_scale = get_ncellx_scale(radii, boxvec,
                                            1);
    // std::shared_ptr<pele::InversePowerPeriodicCellLists<_ndim> > potcell = std::make_shared<pele::InversePowerPeriodicCellLists<_ndim>>(power, eps, radii, boxvec, ncellsx_scale);
    // std::shared_ptr<pele::InversePowerPeriodic<_ndim> > pot = std::make_shared<pele::InversePowerPeriodic<_ndim> >(power, eps, radii, boxvec);
    std::shared_ptr<pele::InverseHalfIntPowerPeriodicCellLists<_ndim, pow2>> potcell = std::make_shared<pele::InverseHalfIntPowerPeriodicCellLists<_ndim, pow2>>(eps, radii, boxvec, ncellsx_scale, exact_sum);
    
    potcell->get_energy(x);
    
    pele::MixedDescentEndOnly optimizer(potcell, x);
    optimizer.run(100000);
    std::cout << optimizer.get_nfev() << "nfev \n";
    std::cout << optimizer.get_nhev() << "nhev \n";
    std::cout << optimizer.get_rms() << "rms \n";
    std::cout << optimizer.get_niter() << "\n";
}