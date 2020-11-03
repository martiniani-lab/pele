#include "pele/array.hpp"
#include "pele/base_interaction.hpp"
#include "pele/base_potential.hpp"
#include "pele/wca.hpp"
#include "pele/hs_wca.hpp"
#include "pele/meta_pow.hpp"
#include "test_utils.hpp"

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <gtest/gtest.h>
#include <cmath>

static double const EPS = std::numeric_limits<double>::min();
#define EXPECT_NEAR_RELATIVE(A, B, T)  EXPECT_NEAR(A/(fabs(A)+fabs(B) + EPS), B/(fabs(A)+fabs(B) + EPS), T)

using pele::Array;
using pele::WCA;
using pele::HS_WCA;
using pele::pos_int_pow;

class WCATest :  public PotentialTest
{
public:
    double sig, eps;
    size_t natoms;

    virtual void setup_potential(){
        pot = std::shared_ptr<pele::BasePotential> (new pele::WCA(sig, eps));
    }

    virtual void SetUp(){
        natoms = 3;
        sig = 1.4;
        eps = 2.1;
        x = Array<double>(natoms*3);
        x[0] = 0.1;
        x[1] = 0.2;
        x[2] = 0.3;
        x[3] = 0.44;
        x[4] = 0.55;
        x[5] = 1.66;
        x[6] = 0.88;
        x[7] = 1.1;
        x[8] = 3.32;
        etrue = 0.9009099166892105;

        setup_potential();
    }
};

TEST_F(WCATest, Energy_Works){
    test_energy();
}
TEST_F(WCATest, EnergyGradient_AgreesWithNumerical){
    test_energy_gradient();
}
TEST_F(WCATest, EnergyGradientHessian_AgreesWithNumerical){
    test_energy_gradient_hessian();
}

class WCAAtomListTest :  public WCATest
{
public:
    virtual void setup_potential(){
        pele::Array<size_t> atoms(natoms);
        for (size_t i =0; i<atoms.size(); ++i){
            atoms[i] = i;
        }
        pot = std::shared_ptr<pele::BasePotential> (new pele::WCAAtomList(
                sig, eps, atoms
                ));
    }
};

TEST_F(WCAAtomListTest, Energy_Works){
    test_energy();
}
TEST_F(WCAAtomListTest, EnergyGradient_AgreesWithNumerical){
    test_energy_gradient();
}
TEST_F(WCAAtomListTest, EnergyGradientHessian_AgreesWithNumerical){
    test_energy_gradient_hessian();
}



/*
 * HS_WCA tests
 */
class HS_WCATest :  public ::testing::Test
{
public:
    double eps, sca, etrue;
    Array<double> x, g, gnum, radii;
    virtual void SetUp(){
        eps = 2.1;
        sca = 1.4;
        x = Array<double>(9);
        x[0] = 0.1;
        x[1] = 0.2;
        x[2] = 0.3;
        x[3] = 0.44;
        x[4] = 0.55;
        x[5] = 1.66;
        x[6] = 0.88;
        x[7] = 1.1;
        x[8] = 3.32;
        radii = Array<double>(3);
        double f = .35;
        radii[0] = .91 * f;
        radii[1] = 1.1 * f;
        radii[2] = 1.13 * f;
        etrue = 189.41835811474974;
        g = Array<double>(x.size());
        gnum = Array<double>(x.size());
    }
};

TEST_F(HS_WCATest, Energy_Works){
    HS_WCA<3> pot(eps, sca, radii);
    double e = pot.get_energy(x);
    ASSERT_NEAR(e, etrue, 1e-10);
}

class OtherfHS_WCA {
public:
    OtherfHS_WCA(const double r_sum_, const double infinity_, const double epsilon_, const double alpha_)
        : r_sum(r_sum_),
          infinity(infinity_),
          epsilon(epsilon_),
          alpha(alpha_),
          r_sum_soft((1 + alpha) * r_sum)
    {}
    double operator()(const double r) const
    {
        if (r >= r_sum_soft) {
            return 0;
        }
        if (r <= r_sum) {
            return infinity;
        }
        const double numerator = sigma();
        const double denominator = pos_int_pow<2>(r) - pos_int_pow<2>(r_sum);
        const double ratio = numerator / denominator;
        return std::max<double>(0, epsilon * (4 * (pos_int_pow<12>(ratio) - pos_int_pow<6>(ratio)) + 1));
    }
    double grad(const double r) const
    {
        if (r >= r_sum_soft) {
            return 0;
        }
        if (r <= r_sum) {
            return 0;
        }
        return (-8 * epsilon * r * sigma()) / pos_int_pow<2>(pos_int_pow<2>(r) - pos_int_pow<2>(r_sum)) * (12 * pos_int_pow<11>(g(r)) - 6 * pos_int_pow<5>(g(r)));
    }
    // This is not the gradient. Rather grad(r) / (-r).
    double scaled_grad(const double r) const
    {
        return grad(r) / (-r);
    }
    double sigma() const { return (2 * alpha + pos_int_pow<2>(alpha)) * pos_int_pow<2>(r_sum) * std::pow(2, -static_cast<double>(1) / static_cast<double>(6)); }
    double g(const double r) const { return sigma() / (pos_int_pow<2>(r) - pos_int_pow<2>(r_sum)); }
private:
    const double r_sum;
    const double infinity;
    const double epsilon;
    const double alpha;
    const double r_sum_soft;
};

class OthersfHS_WCA {
public:
    OthersfHS_WCA(const double r_sum_, const double epsilon_, const double alpha_, const double delta_=1e-10)
        : r_sum(r_sum_),
          epsilon(epsilon_),
          alpha(alpha_),
          delta(delta_),
          fHS_WCA(r_sum, std::numeric_limits<double>::max(), epsilon, alpha),
          r_sum_soft((1 + alpha) * r_sum),
          r_X(r_sum + delta)
    {}
    double operator()(const double r) const
    {
        if (r > r_sum_soft) {
            return 0;
        }
        if (r > r_X) {
            return fHS_WCA(r);
        }
        //return fHS_WCA(r_X) - (r - r_X) * fHS_WCA.scaled_grad(r_X);
        //return fHS_WCA(r_X) - (r - r_X) * fHS_WCA.grad(r_X) / (-r_X);
        return fHS_WCA(r_X) + (r - r_X) * fHS_WCA.grad(r_X);
    }
private:
    const double r_sum;
    const double epsilon;
    const double alpha;
    const double delta;
    const OtherfHS_WCA fHS_WCA;
    const double r_sum_soft;
    const double r_X;
};

TEST_F(HS_WCATest, ExtendedEnergyTest_Works){
    HS_WCA<3> pot(eps, sca, radii);
    const double e = pot.get_energy(x);
    EXPECT_DOUBLE_EQ(e, etrue);
    pele::HS_WCA_interaction pair_pot(eps, sca);
    const size_t atom_a = 0;
    const size_t atom_b = 1;
    const double r_sum = radii[atom_a] + radii[atom_b];
    const double rmin = r_sum / 2;
    const size_t nr_points = 10000;
    const double rmax = 3 * r_sum * (sca + 1);
    const double rdelta = (rmax - rmin) / (nr_points - 1);
    const double infinity = pair_pot._infty;
    OtherfHS_WCA other_implementation(r_sum, infinity, eps, sca);
    for (size_t i = 0; i < nr_points; ++i) {
        const double r = rmin + i * rdelta;
        const double e_pair_pot_f_energy = pair_pot.energy(pos_int_pow<2>(r), r_sum);
        double gab;
        const double e_pair_pot_f_energy_gradient = pair_pot.energy_gradient(pos_int_pow<2>(r), &gab, r_sum);
        double hab;
        const double e_pair_pot_f_energy_gradient_hessian = pair_pot.energy_gradient_hessian(pos_int_pow<2>(r), &gab, &hab, r_sum);
        EXPECT_LE(0, e_pair_pot_f_energy);
        EXPECT_TRUE(almostEqual(e_pair_pot_f_energy, e_pair_pot_f_energy_gradient, 45));
        EXPECT_LE(0, e_pair_pot_f_energy_gradient);
        EXPECT_TRUE(almostEqual(e_pair_pot_f_energy, e_pair_pot_f_energy_gradient_hessian, 45));
        EXPECT_LE(0, e_pair_pot_f_energy_gradient_hessian);
        const double e_other = other_implementation(r);
        EXPECT_NEAR_RELATIVE(e_pair_pot_f_energy, e_other, 1e-10);
        EXPECT_LE(0, e_other);
        if (r > (sca + 1) * r_sum) {
            EXPECT_DOUBLE_EQ(e_other, 0);
            EXPECT_DOUBLE_EQ(e_pair_pot_f_energy, 0);
        }
        else if (r < r_sum) {
            EXPECT_DOUBLE_EQ(e_other, infinity);
            EXPECT_DOUBLE_EQ(e_pair_pot_f_energy, infinity);
        }
    }
    // Un-comment the following to plot sfHS-WCA potential.
    /*
    std::vector<double> x_;
    std::vector<double> y_;
    const size_t scale = 2;
    //pele::sf_HS_WCA_interaction<>(eps, sca).evaluate_pair_potential(rmin, rmax, nr_points, atom_a, atom_b, x_, y_);
    pele::sf_HS_WCA_interaction<>(eps, sca).evaluate_pair_potential(0.70348, 0.70352, scale * nr_points, atom_a, atom_b, x_, y_);
    //std::ofstream out("test_sfhs_wca_shape.txt");
    std::ofstream out("test_sfhs_wca_shape_zoom_grad.txt");
    out.precision(std::numeric_limits<double>::digits10);
    for (size_t i = 0; i < scale * nr_points; ++i) {
        out << x_.at(i) << "\t" << y_.at(i) << "\n";
    }
    out.close();
    */
    // Below HS_WCA_interaction's infinity, HS_WCA_interaction and
    // sf_HS_WCA_interaction have to be the same.
    // Also sf_HS_WCA_interaction should agree with the second
    // alternative implementation given above, for all points.
    pele::sf_HS_WCA_interaction<> sf_pair_pot(eps, sca);
    OthersfHS_WCA sf_other_implementation(r_sum, eps, sca);
    for (size_t i = 0; i < nr_points; ++i) {
        const double r = rmin + i * rdelta;
        double pair_pot_gab;
        const double pair_pot_e = pair_pot.energy_gradient(pos_int_pow<2>(r), &pair_pot_gab, r_sum);
        double sf_pair_pot_gab;
        const double sf_pair_pot_e = sf_pair_pot.energy_gradient(pos_int_pow<2>(r), &sf_pair_pot_gab, r_sum);
        if (pair_pot_e < infinity) {
            // Here sf and f have to give the same result.
            EXPECT_LE(0, pair_pot_e);
            EXPECT_LE(0, sf_pair_pot_e);
            EXPECT_DOUBLE_EQ(pair_pot_e, sf_pair_pot_e);
            EXPECT_DOUBLE_EQ(pair_pot_gab, sf_pair_pot_gab);
        }
        // sf always has to agree with a simpler implementation given above.
        // Numerical differences can be present.
        const double alternative_sf_pair_pot_e = sf_other_implementation(r);
        EXPECT_LE(0, alternative_sf_pair_pot_e);
        EXPECT_NEAR_RELATIVE(sf_pair_pot_e, alternative_sf_pair_pot_e, 1e-10);
    }
}

TEST_F(HS_WCATest, EnergyGradient_AgreesWithNumerical){
    void* pots[6];
    pots[0] = new HS_WCA<3, 1>(eps, sca, radii);
    pots[1] = new HS_WCA<3, 2>(eps, sca, radii);
    pots[2] = new HS_WCA<3, 3>(eps, sca, radii);
    pots[3] = new HS_WCA<3, 4>(eps, sca, radii);
    pots[4] = new HS_WCA<3, 5>(eps, sca, radii);
    pots[5] = new HS_WCA<3, 6>(eps, sca, radii);
    for (int exp_ind = 0; exp_ind < 6; exp_ind++) {
        double e = ((struct pele::BasePotential*)pots[exp_ind])->get_energy_gradient(x, g);
        double ecomp = ((struct pele::BasePotential*)pots[exp_ind])->get_energy(x);
        ASSERT_NEAR(e, ecomp, 1e-10);
        ((struct pele::BasePotential*)pots[exp_ind])->numerical_gradient(x, gnum, 1e-6);
        for (size_t k=0; k<6; ++k){
            ASSERT_NEAR(g[k], gnum[k], 1e-6);
        }
        delete pots[exp_ind];
    }
}

TEST_F(HS_WCATest, EnergyGradientHessian_AgreesWithNumerical){
    void* pots[6];
    pots[0] = new HS_WCA<3, 1>(eps, sca, radii);
    pots[1] = new HS_WCA<3, 2>(eps, sca, radii);
    pots[2] = new HS_WCA<3, 3>(eps, sca, radii);
    pots[3] = new HS_WCA<3, 4>(eps, sca, radii);
    pots[4] = new HS_WCA<3, 5>(eps, sca, radii);
    pots[5] = new HS_WCA<3, 6>(eps, sca, radii);
    for (int exp_ind = 0; exp_ind < 6; exp_ind++) {
        Array<double> h(x.size()*x.size());
        Array<double> hnum(h.size());
        double e = ((struct pele::BasePotential*)pots[exp_ind])->get_energy_gradient_hessian(x, g, h);
        double ecomp = ((struct pele::BasePotential*)pots[exp_ind])->get_energy(x);
        ((struct pele::BasePotential*)pots[exp_ind])->numerical_gradient(x, gnum);
        ((struct pele::BasePotential*)pots[exp_ind])->numerical_hessian(x, hnum);

        EXPECT_NEAR(e, ecomp, 1e-10);
        for (size_t i=0; i<g.size(); ++i){
            ASSERT_NEAR(g[i], gnum[i], 1e-6);
        }
        for (size_t i=0; i<h.size(); ++i){
            ASSERT_NEAR(h[i], hnum[i], 1e-3);
        }
        delete pots[exp_ind];
    }
}

// r_hs = 1.0, eps = 1.0
double simple_energy(double r, double sca, int exp) {
    double C_ir_m = pow((2+sca)*sca / (r*r-1), exp);
    return C_ir_m*C_ir_m - 2*C_ir_m + 1;
}

TEST_F(HS_WCATest, Energy_SimpleTest){
    void* pots[6];
    pots[0] = new pele::sf_HS_WCA_interaction<1>(1.0, sca);
    pots[1] = new pele::sf_HS_WCA_interaction<2>(1.0, sca);
    pots[2] = new pele::sf_HS_WCA_interaction<3>(1.0, sca);
    pots[3] = new pele::sf_HS_WCA_interaction<4>(1.0, sca);
    pots[4] = new pele::sf_HS_WCA_interaction<5>(1.0, sca);
    pots[5] = new pele::sf_HS_WCA_interaction<6>(1.0, sca);
    // 2*r_hs = 1.0, eps = 1.0
    for (int exp = 1; exp <= 6; exp++) {
        for (double r = 1.01; r < 1.0 + sca; r += 0.01) {
            const double e = ((struct pele::BaseInteraction*)pots[exp-1])->energy(r*r, 1.0);
            double ecomp = simple_energy(r, sca, exp);
            EXPECT_NEAR(e, ecomp, e*1e-10);
        }
        delete pots[exp-1];
    }
}

TEST_F(HS_WCATest, Norm_SimpleTest){
    pele::HS_WCA<2, 6> pot(eps, sca, radii);

    pele::Array<double> x = {0.0, 1.0, 3.0, 4.0, 2.0, 2.0, 3.0, 0.0, 4.0, 1.0, 3.0, 3.0};
    double norm = pot.compute_norm(x);
    EXPECT_DOUBLE_EQ(norm, 5.0);
}

class HS_WCA_StabilityTest : public ::testing::Test {
public:
    size_t nparticles;
    size_t ndim;
    size_t ndof;
    double eps;
    double sca;
    double r_hs;
    Array<double> x;
    Array<double> radii;
    virtual void SetUp(){
        nparticles = 2;
        ndim = 2;
        ndof = nparticles * ndim;
        eps = 1;
        r_hs = 1;
        x = Array<double>(ndof);
        radii = Array<double>(nparticles);
        for (size_t i = 0; i < nparticles; ++i) {
            radii[i] = r_hs;
        }
        sca = 0.2;
    }
};

TEST_F(HS_WCA_StabilityTest, EnergyGradientHessian_threshold) {
    pele::HS_WCA<2, 6> pot(eps, sca, radii);
    pele::HS_WCA<2, 6> pot_dx(eps, sca, radii);

    x[1*ndim] = (1 + 0.95 * sca) * 2 * r_hs;
    pele::Array<double> g(x.size());
    pele::Array<double> h(x.size() * x.size());
    const double e = pot.get_energy_gradient_hessian(x, g, h);

    x[1*ndim] = std::nextafter(x[1*ndim], 1e3);
    pele::Array<double> g_dx(x.size());
    pele::Array<double> h_dx(h.size());
    const double e_dx = pot_dx.get_energy_gradient_hessian(x, g_dx, h_dx);

    ::testing::AssertionResult e_result = almostEqual(e, e_dx, 0);
    std::cout << "Energy: " << e_result.message() << std::endl;
    for (size_t i = 0; i < g.size(); ++i) {
        ::testing::AssertionResult g_result = almostEqual(g[i], g_dx[i], 0);
        if(!g_result) {
            std::cout << "Gradient[" << i << "]: " << g_result.message() << std::endl;
        }
    }
    for (size_t i = 0; i < h.size(); ++i) {
        ::testing::AssertionResult h_result = almostEqual(h[i], h_dx[i], 0);
        if(!h_result) {
            std::cout << "Hessian[" << i << "]: " << h_result.message() << std::endl;
        }
    }
}
