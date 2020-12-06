#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "pele/array.hpp"
#include "pele/harmonic.hpp"
#include "pele/meta_pow.hpp"

#include "petscmat.h"
#include "petscsys.h"
#include "petscsystypes.h"
#include "petscvec.h"
#include "petscviewer.h"
#include "test_utils.hpp"

using pele::Array;
using pele::pos_int_pow;
using std::shared_ptr;

static double const EPS = std::numeric_limits<double>::min();
#define EXPECT_NEAR_RELATIVE(A, B, T)  EXPECT_NEAR(A/(fabs(A)+fabs(B) + EPS), B/(fabs(A)+fabs(B) + EPS), T)

class HarmonicTest : public ::testing::Test {
public:
    size_t nr_particles;
    size_t box_dimension;
    size_t nr_dof;
    Array<double> x;
    double k;
    double displ;
    Array<double> xd;
    double e_ini_true;
    void SetUp()
    {
        nr_particles = 100;
        box_dimension = 3;
        nr_dof = nr_particles * box_dimension;
        x = Array<double>(nr_dof, 0);
        k = 12.12;
        displ = 43.44;
        xd = Array<double>(nr_dof, displ);
        e_ini_true = 0.5 * k * nr_dof * pos_int_pow<2>(displ);
    }
};

TEST_F(HarmonicTest, SetKGetK_Works) {
    auto pot = std::make_shared<pele::Harmonic>(x, k, box_dimension);
    for (size_t k = 400; k < 500; ++k) {
        pot->set_k(k);
        const double pot_k = pot->get_k();
        EXPECT_DOUBLE_EQ(pot_k, k);
    }
}

TEST_F(HarmonicTest, Works) {
    auto pot = std::make_shared<pele::Harmonic>(x, k, box_dimension);
    const double e_ini = pot->get_energy(xd);
    EXPECT_NEAR_RELATIVE(e_ini, e_ini_true, 1e-10);
    for (size_t k = 0; k < 100; ++k) {
        pot->set_k(k);
        const double e_pot = pot->get_energy(xd);
        const double e_true = 0.5 * k * nr_dof * pos_int_pow<2>(displ);
        EXPECT_NEAR_RELATIVE(e_pot, e_true, 1e-10);
        Array<double> actual_grad(nr_dof, 0);
        const double e_pot_grad = pot->get_energy_gradient(xd, actual_grad);
        EXPECT_NEAR_RELATIVE(e_pot, e_pot_grad, 1e-10);
        const double e_true_pair = 0.5 * k * pos_int_pow<2>(displ);
        const double e_pair = pele::harmonic_interaction(k).energy(pos_int_pow<2>(displ), 0);
        EXPECT_NEAR_RELATIVE(e_pair, e_true_pair, 1e-10);
        for (size_t i = 0; i < nr_dof; ++i) {
            EXPECT_NEAR_RELATIVE(actual_grad[i], k * displ, 1e-10);
        }
    }
}

TEST_F(HarmonicTest, COM_Works) {
    auto pot = std::make_shared<pele::HarmonicCOM>(x, k, box_dimension);
    // Put center of mass to zero.
    for (size_t i = 0; i < xd.size() / 2; ++i) {
        xd[i] *= -1;
    }
    const double e_ini = pot->get_energy(xd);
    const double e_ini_true = 0.5 * k * nr_particles * box_dimension * pos_int_pow<2>(displ);
    EXPECT_NEAR_RELATIVE(e_ini, e_ini_true, 1e-10);
}

class HarmonicAtomListTest :  public PotentialTest {
public:
    double natoms;
    double k;

    virtual void setup_potential(){
        pele::Array<size_t> atoms(natoms);
        for (size_t i =0; i<atoms.size(); ++i){
            atoms[i] = i;
        }
        pot = std::shared_ptr<pele::BasePotential> (new pele::HarmonicAtomList(
                k, atoms
                ));
    }


    virtual void SetUp()
    {
        natoms = 3;
        k = 1.;
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

        etrue = 17.6197;

        setup_potential();
    }
};

TEST_F(HarmonicAtomListTest, Energy_Works){
    test_energy();
}

TEST_F(HarmonicAtomListTest, EnergyGradient_AgreesWithNumerical){
    test_energy_gradient();
}

TEST_F(HarmonicAtomListTest, EnergyGradientHessian_AgreesWithNumerical){
    test_energy_gradient_hessian();}


class HarmonicNeighborListTest :  public HarmonicAtomListTest {
public:
    virtual void setup_potential(){
        pele::Array<size_t> nlist(natoms * (natoms - 1) );
        size_t count = 0;
        for (size_t i = 0; i < natoms; ++i){
            for (size_t j = i+1; j < natoms; ++j){
                nlist[count++] = i;
                nlist[count++] = j;
            }
        }
        assert(count == nlist.size());
        pot = std::shared_ptr<pele::BasePotential> (new pele::HarmonicNeighborList(
                k, nlist));
    }
};

TEST_F(HarmonicNeighborListTest, Energy_Works){
    test_energy();
}

TEST_F(HarmonicNeighborListTest, EnergyGradient_AgreesWithNumerical){
    test_energy_gradient();
}

TEST_F(HarmonicNeighborListTest, EnergyGradientHessian_AgreesWithNumerical){
    test_energy_gradient_hessian();
}

/**
 * Note that these need to be written into a different testing infrastucture see
 * https://scicomp.stackexchange.com/questions/8516/any-recommendations-for-unit-testing-frameworks-compatible-with-code-libraries-t
 */
TEST(HarmonicPetsc, EnergyGradientWorks) {
    Vec x0;
    Vec grad;
    PetscInt dim = 2;
    double k = 1;
    std::shared_ptr<pele::Harmonic> potential = std::make_shared<pele::Harmonic>(Array<double>(2, 0), k, dim);
    PetscInitializeNoArguments();
    VecCreateSeq(PETSC_COMM_SELF, dim, &x0);
    VecDuplicate(x0, &grad);
    PetscReal coord_vals[2] = {2, 1};
    PetscInt indices[2] = {0, 1};
    VecSetValues(x0, 2, indices, coord_vals, INSERT_VALUES);
    VecAssemblyBegin(x0);
    VecAssemblyEnd(x0);
    double energy = potential->get_energy_gradient_petsc(x0, grad);
    PetscReal grad_vals[2];
    PetscReal x0_vals[2];    
    VecGetValues(grad, 2, indices, grad_vals);
    VecGetValues(x0, 2, indices, x0_vals);
    for (int i =0; i < 2; ++i) {
        ASSERT_NEAR(grad_vals[i], x0_vals[i], 1e-10);
    }
    PetscFinalize();
}

/**
 * Note that these need to be written into a different testing infrastucture see
 * https://scicomp.stackexchange.com/questions/8516/any-recommendations-for-unit-testing-frameworks-compatible-with-code-libraries-t
 */
TEST(HarmonicPetsc, HessianWorks) {
    Vec x0;
    Mat hess;
    PetscInt dim = 2;
    PetscInt sparse_non_zeros = 1;
    PetscInt sparse_block_size=1;
    double k = 1;
    Array<double> origin = {0, 0};
    shared_ptr<pele::Harmonic> harmonic = std::make_shared<pele::Harmonic>(origin, k, dim);    
    // the get_energy_gradient works only for the origin at 0
    PetscInitializeNoArguments();
    MatCreateSeqSBAIJ(PETSC_COMM_SELF, sparse_block_size, dim, dim, sparse_non_zeros, NULL, &hess);
    VecCreateSeq(PETSC_COMM_SELF, dim, &x0);
    PetscReal coord_vals[2] = {2, 1};
    PetscInt indices[2] = {0, 1};
    VecSetValues(x0, 2, indices, coord_vals, INSERT_VALUES);
    VecAssemblyBegin(x0);
    VecAssemblyEnd(x0);
    harmonic->get_hessian_petsc(x0, hess);
    MatView(hess, PETSC_VIEWER_STDOUT_SELF);
    PetscReal hess_data[4];
    MatGetValues(hess, 2, indices, 2, indices, hess_data);
    PetscReal hess_data_true[4] = {1, 0, 0, 1};
    for (auto i = 0; i < 4; ++i) {
        ASSERT_NEAR(hess_data[i], hess_data_true[i], 1e-10);
    }

    PetscFinalize();
}
