#include "pele/lj.hpp"
#include "pele/atlj.hpp"
#include "pele/lbfgs.hpp"
#include "pele/mxopt.hpp"
#include "pele/rosenbrock.hpp"
#include "pele/harmonic.hpp"
#include "pele/cvode.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>
#include "pele/gradient_descent.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <memory>

using pele::Array;
using std::cout;

TEST(LbfgsLJ, TwoAtom_Works){
    auto lj = std::make_shared<pele::LJ> (1., 1.);
    Array<double> x0(6, 0);
    std::cout << x0 << "\n";
    x0[0] = 2.;
    pele::LBFGS lbfgs(lj, x0);
    lbfgs.run();
    ASSERT_GT(lbfgs.get_nfev(), 1);
    ASSERT_GT(lbfgs.get_niter(), 1);
    ASSERT_LT(lbfgs.get_rms(), 1e-4);
    ASSERT_LT(lbfgs.get_rms(), 1e-4);
    ASSERT_NEAR(lbfgs.get_f(), -.25, 1e-10);
    Array<double> x = lbfgs.get_x();
    double dr, dr2 = 0;
    for (size_t i = 0; i < 3; ++i){
        dr = (x[i] - x[3+i]);
        dr2 += dr * dr;
    }
    dr = sqrt(dr2);
    ASSERT_NEAR(dr, pow(2., 1./6), 1e-5);
    Array<double> g = lbfgs.get_g();
    ASSERT_NEAR(g[0], -g[3], 1e-10);
    ASSERT_NEAR(g[1], -g[4], 1e-10);
    ASSERT_NEAR(g[2], -g[5], 1e-10);
    double rms = pele::norm(g) / sqrt(g.size());
    ASSERT_NEAR(rms, lbfgs.get_rms(), 1e-10);
}

TEST(LbfgsLJ, Reset_Works){
    auto lj = std::make_shared<pele::LJ> (1., 1.);
    Array<double> x0(6, 0);
    x0[0] = 2.;
    // lbfgs1 will minimize straight from x0
    pele::LBFGS lbfgs1(lj, x0);
    lbfgs1.run();

    // lbfgs2 will first minimize from x2 (!=x0) then reset from x0
    // it should end up exactly the same as lbfgs1
    Array<double> x2 = x0.copy();
    x2[1] = 2;
    pele::LBFGS lbfgs2(lj, x2);
    double H0 = lbfgs2.get_H0();
    lbfgs2.run();
    // now reset from x0
    lbfgs2.reset(x0);
    lbfgs2.set_H0(H0);
    lbfgs2.run();

    cout << lbfgs1.get_x() << "\n";
    cout << lbfgs2.get_x() << "\n";

    ASSERT_EQ(lbfgs1.get_nfev(), lbfgs2.get_nfev());
    ASSERT_EQ(lbfgs1.get_niter(), lbfgs2.get_niter());

    for (size_t i=0; i<x0.size(); ++i){
        ASSERT_DOUBLE_EQ(lbfgs1.get_x()[i], lbfgs2.get_x()[i]);
    }
    ASSERT_DOUBLE_EQ(lbfgs1.get_f(), lbfgs2.get_f());
    ASSERT_DOUBLE_EQ(lbfgs1.get_rms(), lbfgs2.get_rms());
//    ASSERT_EQ(lbfgs1.get_niter(), lbfgs1.get_niter());
//    ASSERT_GT(lbfgs.get_niter(), 1);
//    ASSERT_LT(lbfgs.get_rms(), 1e-4);
//    ASSERT_LT(lbfgs.get_rms(), 1e-4);
//    ASSERT_NEAR(lbfgs.get_f(), -.25, 1e-10);


}

TEST(LbfgsLJ, SetFuncGradientWorks){
    auto lj = std::make_shared<pele::LJ> (1., 1.);
    Array<double> x0(6, 0);
    x0[0] = 2.;
    pele::LBFGS lbfgs1(lj, x0);
    pele::LBFGS lbfgs2(lj, x0);
    auto grad = x0.copy();
    double e = lj->get_energy_gradient(x0, grad);

    // set the gradient for  lbfgs2.  It should have the same result, but
    // one fewer function evaluation.
    lbfgs2.set_func_gradient(e, grad);
    lbfgs1.run();
    lbfgs2.run();
    ASSERT_EQ(lbfgs1.get_nfev(), lbfgs2.get_nfev() + 1);
    ASSERT_EQ(lbfgs1.get_niter(), lbfgs2.get_niter());
    ASSERT_DOUBLE_EQ(lbfgs1.get_f(), lbfgs2.get_f());
}


TEST(LbfgsRosenbrock, Rosebrock_works){
    auto rosenbrock = std::make_shared<pele::RosenBrock> ();
    Array<double> x0(2, 0);
    pele::CVODEBDFOptimizer lbfgs(rosenbrock, x0);
    // pele::LBFGS lbfgs(rosenbrock, x0, 1e-4, 1, 1);
    // pele ::GradientDescent lbfgs(rosenbrock, x0);
    lbfgs.run(2000);
    Array<double> x = lbfgs.get_x();
    std::cout << x << "\n";
    cout << lbfgs.get_nfev() << " get_nfev() \n";
    cout << lbfgs.get_niter() << " get_niter() \n";
    cout << lbfgs.get_rms() << " get_rms() \n";
    cout << lbfgs.get_rms() << " get_rms() \n";
    std::cout << x0 << "\n" << " \n";
    std::cout << x << "\n";
    std::cout << "this is okay" << "\n";
}



// TEST(LbfgsSaddle, Saddle_works){
//     auto saddle = std::make_shared<pele::Saddle> ();
//     Array<double> x0(2, 0);
//     // Start from a point that ends on a saddle
//     x0[0] = 1.;
//     x0[1] = 0;
//     std::cout << x0 << "\n";
//     pele::LBFGS lbfgs(saddle, x0, 1e-4, 1, 1);
//     lbfgs.run();
//     Array<double> x = lbfgs.get_x();
//     std::cout << x << "x value \n";
//     cout << lbfgs.get_nfev() << " get_nfev() \n";
//     cout << lbfgs.get_niter() << " get_niter() \n";
//     cout << lbfgs.get_rms() << " get_rms() \n";
//     cout << lbfgs.get_rms() << " get_rms() \n";
//     std::cout << "this is okay" << "\n";
// }


// TEST(LbfgsATLJ, heavyconcaveworks){
//     auto saddle = std::make_shared<pele::ATLJ> (1., 1., 1.);
//     Array<double> x0 = {0.97303831232970894, 0, 0, 0, 0, 0, -1.8451195244938898, 0.66954397800263088, 0};
//     // Start from a point that ends on a saddle
//     std::cout << x0 << "\n";
//     pele::LBFGS lbfgs(saddle, x0, 1e-4, 1, 1);
//     lbfgs.run();
//     Array<double> x = lbfgs.get_x();
//     std::cout << x << "x value \n";
//     cout << lbfgs.get_nfev() << " get_nfev() \n";
//     cout << lbfgs.get_niter() << " get_niter() \n";
//     cout << lbfgs.get_rms() << " get_rms() \n";
//     cout << lbfgs.get_rms() << " get_rms() \n";
//     std::cout << "this is okay" << "\n";
// }



// TEST(LbfgsATLJ, why){
//     auto saddle = std::make_shared<pele::ATLJ> (1., 1., 2.);
//     Array<double> x0 = { 1.1596938257996146
//                          , 0.92275271036961359
//                          , 0
//                          , -1.03917561305288
//                          , 0.60339721865003093
//                          , 0
//                          , 0.11585897591687914
//                          , 0.5349653769730589
//                          , 0};
//     // Start from a point that ends on a saddle
//     std::cout << x0 << "\n";
//     pele::LBFGS lbfgs(saddle, x0, 1e-4, 1, 1);
//     lbfgs.run();
//     Array<double> x = lbfgs.get_x();
//     std::cout << x << "x value \n";
//     cout << lbfgs.get_nfev() << " get_nfev() \n";
//     cout << lbfgs.get_niter() << " get_niter() \n";
//     cout << lbfgs.get_rms() << " get_rms() \n";
//     cout << lbfgs.get_rms() << " get_rms() \n";
//     std::cout << "this is okay" << "\n";
// }



// TEST(LbfgsSaddlexcubed, Saddle_works){
//     auto saddlex3 = std::make_shared<pele::XCube> ();
//     Array<double> x0(1, 0);
//     // Start from a point that ends on a saddle
//     x0[0] = 1.;
//     std::cout << x0 << "\n";
//     pele::LBFGS lbfgs(saddlex3, x0, 1e-4, 1, 1);
//     lbfgs.run(30);
//     Array<double> x = lbfgs.get_x();
//     std::cout << x << "x value \n";
//     cout << lbfgs.get_nfev() << " get_nfev() \n";
//     cout << lbfgs.get_niter() << " get_niter() \n";
//     cout << lbfgs.get_rms() << " get_rms() \n";
//     cout << lbfgs.get_rms() << " get_rms() \n";
//     std::cout << "this is okay" << "\n";
// }




