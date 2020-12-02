/**
 * Tests for sparse methods for hessians in pele 
 */


#include "pele/array.hpp"
#include "pele/lj.hpp"
#include "pele/ngt.hpp"
#include "pele/inversepower.hpp"
#include "pele/cell_lists.hpp"
    
#include "petscsys.h"
#include "petscsystypes.h"
#include "petscviewer.h"
#include <petscmat.h>
#include <petscvec.h>
#include <gtest/gtest.h>

using namespace pele;
    

TEST(sparse_energy_gradient_hessian, Energy_Gradient_Hessian_Works){
    auto lj = std::make_shared<pele::LJ> (1., 1.);

    // setup coords
    int natoms = 3;
    Array<double> x;
    x = Array<double>(3*natoms);
    x[0]  = 0.1;
    x[1]  = 0.2;
    x[2]  = 0.3;
    x[3]  = 0.44;
    x[4]  = 0.55;
    x[5]  = 1.66;
    x[6] = 0;
    x[7] = 0;
    x[8] = -3.;
    int N = 3*natoms;
    Array<double> g(N);
    Array<double> h(N*N);
    

    double e = lj->get_energy_gradient_hessian(x, g, h);  
    // set up sparse gradient hessian;
    PetscInitializeNoArguments();
    PetscInt blocksize = 1;
    PetscInt hessav = 9;
    Mat petsc_hess;
    Vec petsc_grad;
    PetscInt i_petsc[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    PetscInt j_petsc[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    PetscScalar sparse_hess_vals[N * N];    
    MatCreateSeqSBAIJ(PETSC_COMM_SELF, blocksize, N, N, hessav, NULL, &petsc_hess);
    VecCreateSeq(PETSC_COMM_SELF, N, &petsc_grad);
    double e_sparse = lj->get_energy_gradient_hessian_sparse(x, petsc_grad, petsc_hess);
    MatType hesstype;
    MatGetType(petsc_hess, &hesstype);
    double val;
    // to view what the hessian looks like
    // MatView(petsc_hess,PETSC_VIEWER_STDOUT_WORLD);
    for (int i =0; i < N; ++i) {
        for (int j = i; j < N; ++j) {
            MatGetValue(petsc_hess, i, j, &val);
            ASSERT_NEAR(val, h[N * i + j], 1e-10);         
        }
    }
}


/**
 * Test on a bunch of coordinates given a packing
 */

TEST(sparse_energy_gradient, Energy_Gradient_Works){
    auto lj = std::make_shared<pele::LJ> (1., 1.);

    // setup coords
    int natoms = 3;
    Array<double> x;
    x = Array<double>(3*natoms);
    x[0]  = 0.1;
    x[1]  = 0.2;
    x[2]  = 0.3;
    x[3]  = 0.44;
    x[4]  = 0.55;
    x[5]  = 1.66;
    x[6] = 0;
    x[7] = 0;
    x[8] = -3.;
    int N = 3*natoms;
    Array<double> g(N);
    double e = lj->get_energy_gradient(x, g);   
    // set up sparse gradient hessian;
    PetscInitializeNoArguments();
    PetscInt blocksize = 1;
    PetscInt hessav = 2;
    Mat petsc_hess;
    Vec petsc_grad;
    VecCreateSeq(PETSC_COMM_SELF, N, &petsc_grad);
    double e_sparse = lj->get_energy_gradient_sparse(x, petsc_grad);

    PetscInt i_petsc[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    PetscScalar petsc_grad_vals[9];
    VecGetValues(petsc_grad, 9, i_petsc, petsc_grad_vals);
    
    ASSERT_NEAR(e, e_sparse, 1e-10);
    for (int i = 0; i < N; ++i) {
        ASSERT_NEAR(petsc_grad_vals[i], g[i], 1e-10);
    }
}



TEST(CellListsSparseGradientCheck, FullRunPeriodic)
{                   
    static const size_t _ndim = 3;
    size_t nr_particles;
    size_t nr_dof;
    double eps;
    double sca;
    double rsca;
    double energy;
    pele::Array<double> x;
    pele::Array<double> hs_radii;
    pele::Array<double> radii;
    pele::Array<double> xfinal;
    pele::Array<double> boxvec;
    double k;
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    nr_particles = 32;
    nr_dof = nr_particles * _ndim;
    eps =1.0;
    double power = 2.5;
    x = pele::Array<double>(nr_dof, 0);
    Array<double> gradcell(nr_dof);

    //////////// parameters for inverse power potential at packing fraction 0.7 in 3d
    //////////// as generated from code params.py in basinerror library
    double box_length = 7.240260952504541;
    boxvec = {box_length, box_length, box_length} ;
    radii = {1.08820262, 1.02000786, 1.0489369,
        1.11204466, 1.0933779,  0.95113611,
        1.04750442, 0.99243214, 0.99483906,
        1.02052993, 1.00720218, 1.07271368,
        1.03805189, 1.00608375, 1.02219316,
        1.01668372, 1.50458554, 1.38563892,
        1.42191474, 1.3402133,  1.22129071,
        1.4457533 , 1.46051053, 1.34804845,
        1.55888282, 1.2981944,  1.4032031,
        1.38689713, 1.50729455, 1.50285511,
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
    double ncellsx_scale = get_ncellx_scale(radii, boxvec,
                                            1);
    std::shared_ptr<pele::InversePowerPeriodicCellLists<_ndim> > potcell = std::make_shared<pele::InversePowerPeriodicCellLists<_ndim>>(power, eps, radii, boxvec, ncellsx_scale);
    double e = potcell->get_energy_gradient(x, gradcell);

    // Sparse initialization
    double e_sparse;
    PetscInitializeNoArguments();
    Vec petsc_grad;
    VecCreateSeq(PETSC_COMM_SELF, nr_dof, &petsc_grad);
    e_sparse = potcell->get_energy_gradient_sparse(x, petsc_grad);

    // get all the values in the gradient
    PetscInt vals[nr_dof];
    double petsc_grad_vals[nr_dof];
    for (int i; i < nr_dof; ++i) {
        vals[i] = i;
    }
    VecGetValues(petsc_grad, nr_dof, vals, petsc_grad_vals);
    
    ASSERT_NEAR(e, e_sparse, 1e-10);
    for (int i = 0; i < nr_dof; ++i) {
        ASSERT_NEAR(petsc_grad_vals[i], gradcell[i], 1e-10);
    }
    
};


TEST(CellListsSparseHessianCheck, FullRunPeriodic)
{                   
    static const size_t _ndim = 3;
    size_t nr_particles;
    size_t nr_dof;
    double eps;
    double sca;
    double rsca;
    double energy;
    pele::Array<double> x;
    pele::Array<double> hs_radii;
    pele::Array<double> radii;
    pele::Array<double> xfinal;
    pele::Array<double> boxvec;
    double k;
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    nr_particles = 32;
    nr_dof = nr_particles * _ndim;
    eps =1.0;
    double power = 2.5;
    x = pele::Array<double>(nr_dof, 0);
    Array<double> grad(nr_dof);
    Array<double> gradcell(nr_dof);
    Array<double> hesscell(nr_dof*nr_dof);

    //////////// parameters for inverse power potential at packing fraction 0.7 in 3d
    //////////// as generated from code params.py in basinerror library
    double box_length = 7.240260952504541;
    boxvec = {box_length, box_length, box_length} ;
    radii = {1.08820262, 1.02000786, 1.0489369,
        1.11204466, 1.0933779,  0.95113611,
        1.04750442, 0.99243214, 0.99483906,
        1.02052993, 1.00720218, 1.07271368,
        1.03805189, 1.00608375, 1.02219316,
        1.01668372, 1.50458554, 1.38563892,
        1.42191474, 1.3402133,  1.22129071,
        1.4457533 , 1.46051053, 1.34804845,
        1.55888282, 1.2981944,  1.4032031,
        1.38689713, 1.50729455, 1.50285511,
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
    double ncellsx_scale = get_ncellx_scale(radii, boxvec,
                                            1);
    std::shared_ptr<pele::InversePowerPeriodicCellLists<_ndim> > potcell = std::make_shared<pele::InversePowerPeriodicCellLists<_ndim>>(power, eps, radii, boxvec, ncellsx_scale);
    // initializing a second potential to make sure set values from one function don't propagate to the other;
    std::shared_ptr<pele::InversePowerPeriodicCellLists<_ndim> > potcell_sparse = std::make_shared<pele::InversePowerPeriodicCellLists<_ndim>>(power, eps, radii, boxvec, ncellsx_scale);
    std::shared_ptr<pele::InversePowerPeriodic<_ndim> > pot = std::make_shared<pele::InversePowerPeriodic<_ndim> >(power, eps, radii, boxvec);
    double e = potcell->get_energy_gradient_hessian(x, gradcell, hesscell);

    // Sparse initialization
    double e_sparse;
    PetscInitializeNoArguments();
    Vec petsc_grad;
    Mat petsc_hess;
    VecCreateSeq(PETSC_COMM_SELF, nr_dof, &petsc_grad);
    int blocksize = 1;
    int hessav = 50;
    MatCreateSeqSBAIJ(PETSC_COMM_SELF, blocksize, nr_dof, nr_dof, hessav, NULL, &petsc_hess);
    e_sparse = potcell_sparse->get_energy_gradient_hessian_sparse(x, petsc_grad, petsc_hess);
    // MatView(petsc_hess, PETSC_VIEWER_STDOUT_SELF);
    // std::cout << "------------------------------------------" << "\n";
    // Sparse Negative hessian initialization
    Mat negative_hess;
    MatCreateSeqSBAIJ(PETSC_COMM_SELF, blocksize, nr_dof, nr_dof, hessav, NULL, &negative_hess);
    potcell_sparse->get_negative_hessian_sparse(x, negative_hess);
    // MatView(negative_hess, PETSC_VIEWER_STDOUT_SELF);



    // get all the values in the gradient
    PetscInt vals[nr_dof];
    PetscReal petsc_grad_vals[nr_dof];
    for (int i; i < nr_dof; ++i) {
        vals[i] = i;
    }
    VecGetValues(petsc_grad, nr_dof, vals, petsc_grad_vals);

    
    ASSERT_NEAR(e, e_sparse, 1e-10);

    for (int i = 0; i < nr_dof; ++i) {
        ASSERT_NEAR(petsc_grad_vals[i], gradcell[i], 1e-10);
    }

    double val;
    double negative_val;
    for (int i =0; i < nr_dof; ++i) {
        for (int j = i; j < nr_dof; ++j) {
            MatGetValue(petsc_hess, i, j, &val);
            MatGetValue(negative_hess, i, j, &negative_val);
            ASSERT_NEAR(val, -negative_val, 1e-10);
            ASSERT_NEAR(val, hesscell[nr_dof*i+j], 1e-10);
        }
    }
};




