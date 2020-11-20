/**
 * tests for the cvode solver and related functions note that the MPI Dependencies can mess these up
 */


#include "nvector/nvector_petsc.h"
#include "pele/base_potential.hpp"
#include "pele/inversepower.hpp"
#include "pele/cell_lists.hpp"

#include "petscmat.h"
#include "petscsys.h"
#include "petscsystypes.h"
#include "petscvec.h"
#include "petscviewer.h"
#include <petsctao.h>
#include "sundials/sundials_nvector.h"
#include <memory>
#include <pele/lj.hpp>
#include <pele/cvode.hpp>
#include <pele/array.hpp>

#include <gtest/gtest.h>
using namespace pele;

typedef struct {
    PetscInt  n;          /* dimension */
    PetscReal alpha;   /* condition parameter */
    PetscBool chained;
} AppCtx;

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
    PetscVec_eq_pele(x_petsc, x0);
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
    std::cout << "this works" << "\n";    
    // N_VDestroy_Petsc(x_nvec);
    // N_VDestroy_Petsc(minus_grad_nvec);
    VecDestroy(&x_petsc);
    // VecDestroy(&minus_grad_petsc);
    PetscFinalize();
}

TEST(CVODEJac, negativehessianworks)
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

    
    // std::cout << grad-gradcell << "\n";

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

    // Sparse Negative hessian initialization
    Mat negative_hess;
    MatCreateSeqSBAIJ(PETSC_COMM_SELF, blocksize, nr_dof, nr_dof, hessav, NULL, &negative_hess);
    potcell_sparse->get_negative_hessian_sparse(x, negative_hess);
    MatView(negative_hess, PETSC_VIEWER_STDOUT_SELF);


    // SNES initialization
    SNES NLS;
    Vec coords;
    Mat test_jacobian;
    MatCreateSeqSBAIJ(PETSC_COMM_SELF, blocksize, nr_dof, nr_dof, hessav, NULL, &test_jacobian);
    Mat dummy;
    Vec x_petsc;
    PetscVec_eq_pele(x_petsc, x);

    // user data initialization
    UserData_ udata;
    udata.pot_ = potcell_sparse;
    
    // run negative hessian wrapper routine within SNES
    negative_hessian_wrapper(NLS, x_petsc, dummy, test_jacobian, (void *)(&udata));

    // // Viewers
    // MatView(test_jacobian, PETSC_VIEWER_STDOUT_SELF);
    // std::cout << "--------------------------------------------" << "\n";
    // MatView(negative_hess, PETSC_VIEWER_STDOUT_SELF);    

    double test_negative_hessian_val;
    double test_jacobian_val;
    for (int i =0; i < nr_dof; ++i) {
        for (int j = i; j < nr_dof; ++j) {
            MatGetValue(negative_hess, i, j, &test_negative_hessian_val);
            MatGetValue(test_jacobian, i, j, &test_jacobian_val);
            ASSERT_NEAR(test_negative_hessian_val, test_jacobian_val, 1e-10);
        }
    }
};


PetscErrorCode FormFunctionGradient(Tao tao,Vec X,PetscReal *f, Vec G,void *ptr)
{
    AppCtx            *user = (AppCtx *) ptr;
    PetscInt          i,nn=user->n/2;
    PetscErrorCode    ierr;
    PetscReal         ff=0,t1,t2,alpha=user->alpha;
    PetscScalar       *g;
    const PetscScalar *x;

    PetscFunctionBeginUser;
    /* Get pointers to vector data */
    ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
    ierr = VecGetArray(G,&g);CHKERRQ(ierr);

    /* Compute G(X) */
    if (user->chained) {
        g[0] = 0;
        for (i=0; i<user->n-1; i++) {
            t1 = x[i+1] - x[i]*x[i];
            ff += PetscSqr(1 - x[i]) + alpha*t1*t1;
            g[i] += -2*(1 - x[i]) + 2*alpha*t1*(-2*x[i]);
            g[i+1] = 2*alpha*t1;
        }
    } else {
        for (i=0; i<nn; i++){
            t1 = x[2*i+1]-x[2*i]*x[2*i]; t2= 1-x[2*i];
            ff += alpha*t1*t1 + t2*t2;
            g[2*i] = -4*alpha*t1*x[2*i]-2.0*t2;
            g[2*i+1] = 2*alpha*t1;
        }
    }

    /* Restore vectors */
    ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
    ierr = VecRestoreArray(G,&g);CHKERRQ(ierr);
    *f   = ff;

    ierr = PetscLogFlops(15.0*nn);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
PetscErrorCode FormHessian(Tao tao,Vec X,Mat H, Mat Hpre, void *ptr)
{
    AppCtx            *user = (AppCtx*)ptr;
    PetscErrorCode    ierr;
    PetscInt          i, ind[2];
    PetscReal         alpha=user->alpha;
    PetscReal         v[2][2];
    const PetscScalar *x;
    PetscBool         assembled;

    PetscFunctionBeginUser;
    /* Zero existing matrix entries */
    ierr = MatAssembled(H,&assembled);CHKERRQ(ierr);
    if (assembled){ierr = MatZeroEntries(H);CHKERRQ(ierr);}

    /* Get a pointer to vector data */
    ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);

    /* Compute H(X) entries */
    if (user->chained) {
        ierr = MatZeroEntries(H);CHKERRQ(ierr);
        for (i=0; i<user->n-1; i++) {
            PetscScalar t1 = x[i+1] - x[i]*x[i];
            v[0][0] = 2 + 2*alpha*(t1*(-2) - 2*x[i]);
            v[0][1] = 2*alpha*(-2*x[i]);
            v[1][0] = 2*alpha*(-2*x[i]);
            v[1][1] = 2*alpha*t1;
            ind[0] = i; ind[1] = i+1;
            ierr = MatSetValues(H,2,ind,2,ind,v[0],ADD_VALUES);CHKERRQ(ierr);
        }
    } else {
        for (i=0; i<user->n/2; i++){
            v[1][1] = 2*alpha;
            v[0][0] = -4*alpha*(x[2*i+1]-3*x[2*i]*x[2*i]) + 2;
            v[1][0] = v[0][1] = -4.0*alpha*x[2*i];
            ind[0]=2*i; ind[1]=2*i+1;
            ierr = MatSetValues(H,2,ind,2,ind,v[0],INSERT_VALUES);CHKERRQ(ierr);
        }
    }
    ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);

    /* Assemble matrix */
    ierr = MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = PetscLogFlops(9.0*user->n/2.0);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}



TEST(CVODESolverTest, CVODESolverWorks){
    auto rosenbrock = std::make_shared<pele::LJ> (1, 1);
    Array<double> x0(6, 0);
    pele::CVODEBDFOptimizer lbfgs(rosenbrock, x0);
    
}





TEST(TaoNewton, TaoNewtonstepswork) {
    PetscErrorCode     ierr;                  /* used to check for functions returning nonzeros */
    PetscReal          zero=0.0;
    Vec                x;                     /* solution vector */
    Mat                H;
    Tao                tao;                   /* Tao solver context */
    PetscBool          flg, test_lmvm = PETSC_FALSE;
    PetscMPIInt        size;                  /* number of processes running */
    AppCtx             user;                  /* user-defined application context */
    KSP                ksp;
    PC                 pc;
    Mat                M;
    Vec                in, out, out2;
    PetscReal mult_solve_dist;
    ierr = PetscInitializeNoArguments();
    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);
    // if (size > 1)
    //     SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE,
    //             "Incorrect number of processors");
    user.n = 2;
    user.alpha = 99.0;
    user.chained = PETSC_FALSE;
    ierr = VecCreateSeq(PETSC_COMM_SELF,user.n,&x);
    ierr = MatCreateSeqBAIJ(PETSC_COMM_SELF, 2, user.n, user.n, 1, NULL, &H);
    

    /* Create TAO solver with desired solution method */
    ierr = TaoCreate(PETSC_COMM_SELF,&tao);
    ierr = TaoSetType(tao, TAOLMVM);
    

    /* Set solution vec and an initial guess */
    ierr = VecSet(x, zero);
    ierr = TaoSetInitialVector(tao,x);

    ierr = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,&user);
    ierr = TaoSetHessianRoutine(tao, H, H, FormHessian, &user);
    

    /* Test the LMVM matrix */
    if (test_lmvm) {
        ierr = PetscOptionsSetValue(NULL, "-tao_type", "bqnktr");
    }

    ierr = TaoSolve(tao);    
}





 





 