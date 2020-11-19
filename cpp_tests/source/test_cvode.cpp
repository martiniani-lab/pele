/**
 * tests for the cvode solver and related functions note that the MPI Dependencies can mess these up
 */


#include "nvector/nvector_petsc.h"
#include "pele/base_potential.hpp"
#include "petscsys.h"
#include "petscsystypes.h"
#include "petscvec.h"
#include "petscviewer.h"
#include "sundials/sundials_nvector.h"
#include <memory>
#include <pele/lj.hpp>
#include <pele/cvode.hpp>
#include <pele/array.hpp>

#include <gtest/gtest.h>
using namespace pele;


TEST(CVODELJ, gradFunctionWorks){
    PetscInitializeNoArguments();
    std::shared_ptr<BasePotential> rosenbrock = std::make_shared<pele::LJ> (1, 1);
    UserData_ s1;
    Array<double> x0(6, 0);
    x0[0] = 2;
    Array<double> g(x0.size());
    rosenbrock->get_energy_gradient(x0,g);
    std::cout << g << "\n";
    std::cout << x0 << "\n";
    Vec x_petsc;
    Vec minus_grad_petsc;
    Vec direct_grad;
    VecCreateSeq(PETSC_COMM_SELF, x0.size(), &minus_grad_petsc);
    VecDuplicate(minus_grad_petsc, &direct_grad);
    rosenbrock->get_energy_gradient_sparse(x0, direct_grad);
    // set relevant user data
    s1.pot_ = rosenbrock;
    s1.neq = x0.size();
    s1.nfev =0;
    s1.stored_energy =0;
    double dummy;
    N_Vector x_nvec;
    PetsVec_eq_pele(x_petsc, x0);
    x_nvec = N_VMake_Petsc(x_petsc);
    N_Vector minus_grad_nvec;
    VecZeroEntries(minus_grad_petsc);
    minus_grad_nvec = N_VMake_Petsc(minus_grad_petsc);

    void * s1_void = &s1;
    f(dummy, x_nvec, minus_grad_nvec, (void *) &s1);


    // get values
    PetscInt ix[] = {0,1,2,3,4,5};
    double negative_grad_arr[6];
    double *direct_grad_arr;

    VecGetValues(minus_grad_petsc, 6, ix, negative_grad_arr);
    VecGetArray(direct_grad, &direct_grad_arr);

    // assert they're the same
    for (auto i = 0; i < 6; ++i) {
        ASSERT_NEAR(negative_grad_arr[i], -direct_grad_arr[i], 1e-10);
    }
    N_VDestroy(x_nvec);
    VecDestroy(&x_petsc);
    VecDestroy(&minus_grad_petsc);    
    PetscFinalize();
}


// TEST(CVODELJ, CVODESolverWorks){
//     auto rosenbrock = std::make_shared<pele::LJ> (1, 1);
//     Array<double> x0(6, 0);
//     pele::CVODEBDFOptimizer lbfgs(rosenbrock, x0);
//     // pele::LBFGS lbfgs(rosenbrock, x0, 1e-4, 1, 1);
//     // pele ::GradientDescent lbfgs(rosenbrock, x0);
//     lbfgs.run(2000);
//     Array<double> x = lbfgs.get_x();
//     std::cout << x << "\n";
//     cout << lbfgs.get_nfev() << " get_nfev() \n";
//     cout << lbfgs.get_niter() << " get_niter() \n";
//     cout << lbfgs.get_rms() << " get_rms() \n";
//     cout << lbfgs.get_rms() << " get_rms() \n";
//     std::cout << x0 << "\n" << " \n";
//     std::cout << x << "\n";
//     std::cout << "this is okay" << "\n";
//     Eigen::MatrixXf m(3,3);
//     double s2 = sqrt(2);
//     m(0,0) = 2;
//     m(0,1) = -1;
//     m(0,2) = 0;
//     m(1,0) = 1;
//     m(1,1) = 2;
//     m(1,2) = 0;
//     m(2,0) = 0;
//     m(2,1) = 0;
//     m(2,2) = 0;
//     std::cout << "here" << "\n";

//     Eigen::VectorXf b(3);
//     b << 1, 0, 0;
//     std::cout << m.colPivHouseholderQr().solve(b) << "solution \n";

//     std::cout << m << std::endl;
//     std::cout << b << std::endl;
// }


// TEST(CVODELJ, GradFunctionWorks){
//     auto rosenbrock = std::make_shared<pele::LJ> (1, 1);
//     Array<double> x0(6, 0);
//     pele::CVODEBDFOptimizer lbfgs(rosenbrock, x0);
//     // pele::LBFGS lbfgs(rosenbrock, x0, 1e-4, 1, 1);
//     // pele ::GradientDescent lbfgs(rosenbrock, x0);
//     lbfgs.run(2000);
//     Array<double> x = lbfgs.get_x();
//     std::cout << x << "\n";
//     cout << lbfgs.get_nfev() << " get_nfev() \n";
//     cout << lbfgs.get_niter() << " get_niter() \n";
//     cout << lbfgs.get_rms() << " get_rms() \n";
//     cout << lbfgs.get_rms() << " get_rms() \n";
//     std::cout << x0 << "\n" << " \n";
//     std::cout << x << "\n";
//     std::cout << "this is okay" << "\n";
//     Eigen::MatrixXf m(3,3);
//     double s2 = sqrt(2);
//     m(0,0) = 2;
//     m(0,1) = -1;
//     m(0,2) = 0;
//     m(1,0) = 1;
//     m(1,1) = 2;
//     m(1,2) = 0;
//     m(2,0) = 0;
//     m(2,1) = 0;
//     m(2,2) = 0;
//     std::cout << "here" << "\n";

//     Eigen::VectorXf b(3);
//     b << 1, 0, 0;
//     std::cout << m.colPivHouseholderQr().solve(b) << "solution \n";

//     std::cout << m << std::endl;
//     std::cout << b << std::endl;
// }




 





 