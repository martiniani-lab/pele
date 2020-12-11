/**
 * tests for the cvode solver and related functions note that the MPI Dependencies can mess these up
 */


#include "nvector/nvector_petsc.h"
#include "pele/base_potential.hpp"
#include "pele/harmonic.hpp"
#include "pele/inversepower.hpp"
#include "pele/cell_lists.hpp"
#include <iostream>
#include <petscdm.h>
#include <petscdmda.h>

#include "pele/petsc_interface.hpp"
#include "petscmat.h"
#include "petscsnes.h"
#include "petscsys.h"
#include "petscsystypes.h"
#include "petscvec.h"
#include "pele/rosenbrock.hpp"
#include "petscviewer.h"
#include <petsctao.h>
#include "sundials/sundials_nvector.h"
#include <memory>
#include <pele/lj.hpp>
#include <pele/cvode.hpp>
#include <pele/array.hpp>



#include <cvode/cvode.h>
#include <nvector/nvector_petsc.h>
#include <sunnonlinsol/sunnonlinsol_petscsnes.h>



#include <gtest/gtest.h>
using namespace pele;

extern PetscErrorCode FormFunction(DM,PetscReal,Vec,Vec,void*);
extern PetscErrorCode FormInitialSolution(DM,Vec);
extern PetscErrorCode MySNESMonitor(SNES, PetscInt, PetscReal,
                                    PetscViewerAndFormat *);

/* f - computes f(t,x); this interfaces FormFunction to the CVODE expected format */
extern int fn(PetscReal t, N_Vector x, N_Vector xdot, void *ptr);
extern PetscErrorCode MyCVodeMonitor(PetscInt,PetscReal,Vec,void*);

/* private helper function for checking return value from SUNDIALS calls */
static int check_retval(void *value, const char *funcname, int opt);

TEST(CVODELJ, gradFunctionWorks){
    std::cout << "initializetion" << "\n";
    PetscInitializeNoArguments();
    // std::cout << "initialization done " << "\n";
    // std::shared_ptr<BasePotential> rosenbrock = std::make_shared<pele::LJ> (1, 1);
    // UserData_ s1;
    // Array<double> x0(6, 0);
    // x0[0] = 2;
    // Array<double> g(x0.size());
    // rosenbrock->get_energy_gradient(x0,g);
    // std::cout << g << "\n";
    // std::cout << x0 << "\n";
    // Vec x_petsc;
    // Vec minus_grad_petsc;
    // Vec direct_grad;
    // VecCreateSeq(PETSC_COMM_SELF, x0.size(), &minus_grad_petsc);
    // VecDuplicate(minus_grad_petsc, &direct_grad);
    // rosenbrock->get_energy_gradient_sparse(x0, direct_grad);
    // // set relevant user data
    // s1.pot_ = rosenbrock;
    // s1.neq = x0.size();
    // s1.nfev =0;
    // s1.stored_energy =0;
    // double dummy;
    // N_Vector x_nvec;
    // PetscVec_eq_pele(x_petsc, x0);
    // x_nvec = N_VMake_Petsc(x_petsc);
    // N_Vector minus_grad_nvec;
    // VecZeroEntries(minus_grad_petsc);
    // minus_grad_nvec = N_VMake_Petsc(minus_grad_petsc);
    // void * s1_void = &s1;
    // f(dummy, x_nvec, minus_grad_nvec, (void *) &s1);


    // // get values
    // PetscInt ix[] = {0,1,2,3,4,5};
    // double negative_grad_arr[6];
    // double *direct_grad_arr;

    // VecGetValues(minus_grad_petsc, 6, ix, negative_grad_arr);
    // VecGetArray(direct_grad, &direct_grad_arr);

    // // assert they're the same
    // for (auto i = 0; i < 6; ++i) {
    //     ASSERT_NEAR(negative_grad_arr[i], -direct_grad_arr[i], 1e-10);
    // }
    // std::cout << "this works" << "\n";    
    // // N_VDestroy_Petsc(x_nvec);
    // // N_VDestroy_Petsc(minus_grad_nvec);
    // VecDestroy(&x_petsc);
    // // VecDestroy(&minus_grad_petsc);
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
    Mat snes_jacobian;
    MatCreateSeqSBAIJ(PETSC_COMM_SELF, blocksize, nr_dof, nr_dof, hessav, NULL, &snes_jacobian);
    Vec x_petsc_dummy;
    PetscVec_eq_pele(x_petsc_dummy, x);
    potcell_sparse->get_hessian_petsc(x_petsc_dummy, snes_jacobian);
    MatView(snes_jacobian, PETSC_VIEWER_STDOUT_SELF);


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
    SNESJacobianWrapper(NLS, x_petsc, dummy, test_jacobian, (void *)(&udata));

    // // Viewers
    // MatView(test_jacobian, PETSC_VIEWER_STDOUT_SELF);
    // std::cout << "--------------------------------------------" << "\n";
    // MatView(negative_hess, PETSC_VIEWER_STDOUT_SELF);

    double test_negative_hessian_val;
    double test_jacobian_val;
    for (int i =0; i < nr_dof; ++i) {
        for (int j = i; j < nr_dof; ++j) {
            MatGetValue(snes_jacobian, i, j, &test_negative_hessian_val);
            MatGetValue(test_jacobian, i, j, &test_jacobian_val);
            ASSERT_NEAR(test_negative_hessian_val, test_jacobian_val, 1e-10);
        }
    }
};






TEST(CVST, CVODESolverWorksIP){
    static const size_t _ndim = 3;
    size_t nr_particles;
    size_t nr_dof;
    double eps;
    double sca;
    double rsca;
    double energy;
    //////////// parameters for inverse power potential at packing fraction 0.7 in 3d
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
    nr_particles = 32;
    nr_dof = nr_particles * _ndim;
    eps =1.0;
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
    // std::shared_ptr<pele::InversePowerPeriodic<_ndim> > pot = std::make_shared<pele::InversePowerPeriodic<_ndim> >(power, eps, radii, boxvec);

    
    pele::CVODEBDFOptimizer optimizer(potcell, x);
    optimizer.run();
    std::cout << optimizer.get_nfev() << "nfev \n";
    std::cout << optimizer.get_nhev() << "nhev \n";
    std::cout << optimizer.get_rms() << "rms \n";    
}





TEST(CVODESolverTest, CVODESolverWorks){
    // auto harmonic_pot = std::make_shared<pele::Harmonic>(origin, 1, 2);
    auto rosenbrock = std::make_shared<pele::RosenBrock>();
    // define particular x and y    
    Array<double> x0(2, 0);
    // x0[2] = 1.0;
    // constructor
    pele::CVODEBDFOptimizer lbfgs(rosenbrock, x0);
    
    lbfgs.run();
    std::cout << lbfgs.get_nfev() << "nfev \n";
    std::cout << lbfgs.get_nhev() << "nhev \n";
    std::cout << lbfgs.get_rms() << "rms \n";    
}

TEST(CVODEPETSCex, CVODEPETScexworks)
{
    /* SUNDIALS data structures */
    void*              cvode_mem;        /* integrator memory */
    N_Vector           nvecx;
    SUNNonlinearSolver NLS;
    long int           nsteps = 0;

    /* PETSc data structures */
    SNES                 snes;
    Vec                  x,r;            /* solution, residual vectors */
    Mat                  Jmf;
    PetscErrorCode       ierr;
    DM                   da;
    PetscViewerAndFormat *vf;
    PetscReal            T0, t, tf;

    /* set start and stop time */    
    T0 = 0.;
    t  = 0.;
    tf = 0.0005;


    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Initialize program
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = PetscInitializeNoArguments();
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Create distributed array (DMDA) to manage parallel grid and vectors
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,8,8,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da);
    ierr = DMSetFromOptions(da);
    ierr = DMSetUp(da);

    /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Extract global vectors from DMDA; then duplicate for remaining
        vectors that are the same types
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = DMCreateGlobalVector(da,&x);
    ierr = VecDuplicate(x,&r);



    
    /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Create N_Vector wrapper of petsc vector
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    nvecx = N_VMake_Petsc(x);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Create CVODE integrator
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    cvode_mem = CVodeCreate(CV_BDF);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Create nonlinear solver and set its options.

       Set Jacobian matrix data structure and default Jacobian evaluation
       routine. User can override with:
       -snes_mf : matrix-free Newton-Krylov method with no preconditioning
       (unless user explicitly sets preconditioner)
       -snes_mf_operator : form preconditioning matrix as set by the user,
       but use matrix-free approx for Jacobian-vector
       products within Newton-Krylov method

       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = SNESCreate(PETSC_COMM_WORLD,&snes);

    /* create SUNNonlinearSolver object which interfaces to SNES */
    NLS = SUNNonlinSol_PetscSNES(nvecx,snes); /* This will call SNESSetFunction appropriately */

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Customize nonlinear solver
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_DEFAULT,&vf);
    ierr = SNESMonitorSet(snes,(PetscErrorCode (*)(SNES,PetscInt,PetscReal,void*))MySNESMonitor,vf,(PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy);

    /* use matrix free */
    ierr = MatCreateSNESMF(snes,&Jmf);
    ierr = SNESSetJacobian(snes,Jmf,Jmf,MatMFFDComputeJacobian,0);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Set runtime options
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = SNESSetFromOptions(snes);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Set initial conditions and integrator options
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = FormInitialSolution(da,x);
    ierr = CVodeInit(cvode_mem, fn, T0, nvecx);
    

    /* provide the DM context as user data so we can access it in the RHS */
    ierr = CVodeSetUserData(cvode_mem,(void *)da);

    /* use the PETSc TS default tolerances */
    ierr = CVodeSStolerances(cvode_mem,1e-4,1e-4);

    /* set the max order to 1 for Backward Euler */
    ierr = CVodeSetMaxOrd(cvode_mem,1);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Set the nonlinear solver
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = CVodeSetNonlinearSolver(cvode_mem,NLS);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Solve nonlinear system
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    while (t<tf) {
        /* CV_ONE_STEP mode causes CVODE to return after every time step.
           We use it here to demonstrate how to print monitoring information
           at every time step. */
        MyCVodeMonitor(nsteps,t,x,NULL);
        ierr = CVode(cvode_mem,tf,nvecx,&t,CV_ONE_STEP);
        ierr = CVodeGetNumSteps(cvode_mem, &nsteps);
    }
    MyCVodeMonitor(nsteps,t,x,NULL);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Free work space.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    CVodeFree(&cvode_mem);
    /* ierr = VecDestroy(&x); */
    /* N_VDestroy_Petsc(nvecx); */
    
    N_VDestroy(nvecx);
    ierr = SUNNonlinSolFree(NLS);
    ierr = VecDestroy(&x);
    ierr = VecDestroy(&r);
    ierr = MatDestroy(&Jmf);
    ierr = DMDestroy(&da);

    ierr = PetscFinalize();
}



int fn(PetscReal t, N_Vector x, N_Vector xdot, void *ptr)
{
    PetscErrorCode ierr;
    ierr = FormFunction((DM)ptr,t,N_VGetVector_Petsc(x),N_VGetVector_Petsc(xdot),NULL);
    return ierr;
}

/*
  FormFunction - Evaluates nonlinear function, F(x).

  Input Parameters:
  .  DM - the DM context
  .  X - input vector
  .  ptr - optional user-defined context, as set by SNESSetFunction()

  Output Parameter:
  .  F - function vector
*/
PetscErrorCode FormFunction(DM da,PetscReal ftime,Vec X,Vec F,void *ptr)
{
    PetscErrorCode ierr;
    PetscInt       i,j,Mx,My,xs,ys,xm,ym;
    PetscReal      two = 2.0,hx,hy,sx,sy;
    PetscScalar    u,uxx,uyy,**x,**f;
    Vec            localX;

    PetscFunctionBeginUser;
    ierr = DMGetLocalVector(da,&localX);
    ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

    hx = 1.0/(PetscReal)(Mx-1); sx = 1.0/(hx*hx);
    hy = 1.0/(PetscReal)(My-1); sy = 1.0/(hy*hy);

    /*
      Scatter ghost points to local vector,using the 2-step process
      DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
      By placing code between these two statements, computations can be
      done while messages are in transition.
    */
    ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX);
    ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX);

    /*
      Get pointers to vector data
    */
    ierr = DMDAVecGetArrayRead(da,localX,&x);
    ierr = DMDAVecGetArray(da,F,&f);

    /*
      Get local grid boundaries
    */
    ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);

    /*
      Compute function over the locally owned part of the grid
    */
    for (j=ys; j<ys+ym; j++) {
        for (i=xs; i<xs+xm; i++) {
            if (i == 0 || j == 0 || i == Mx-1 || j == My-1) {
                f[j][i] = x[j][i];
                continue;
            }
            u   = x[j][i];
            uxx = (two*u - x[j][i-1] - x[j][i+1])*sx;
            uyy = (two*u - x[j-1][i] - x[j+1][i])*sy;
            /*      f[j][i] = -(uxx + uyy); */
            f[j][i] = -u*(uxx + uyy) - (4.0 - 1.0)*((x[j][i+1] - x[j][i-1])*(x[j][i+1] - x[j][i-1])*.25*sx +
                                                    (x[j+1][i] - x[j-1][i])*(x[j+1][i] - x[j-1][i])*.25*sy);
        }
    }

    /*
      Restore vectors
    */
    ierr = DMDAVecRestoreArrayRead(da,localX,&x);
    ierr = DMDAVecRestoreArray(da,F,&f);
    ierr = DMRestoreLocalVector(da,&localX);
    ierr = PetscLogFlops(11.0*ym*xm);
    PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
PetscErrorCode FormInitialSolution(DM da,Vec U)
{
    PetscErrorCode ierr;
    PetscInt       i,j,xs,ys,xm,ym,Mx,My;
    PetscScalar    **u;
    PetscReal      hx,hy,x,y,r;

    PetscFunctionBeginUser;
    ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

    hx = 1.0/(PetscReal)(Mx-1);
    hy = 1.0/(PetscReal)(My-1);

    /*
      Get pointers to vector data
    */
    ierr = DMDAVecGetArray(da,U,&u);

    /*
      Get local grid boundaries
    */
    ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);

    /*
      Compute function over the locally owned part of the grid
    */
    for (j=ys; j<ys+ym; j++) {
        y = j*hy;
        for (i=xs; i<xs+xm; i++) {
            x = i*hx;
            r = PetscSqrtReal((x-.5)*(x-.5) + (y-.5)*(y-.5));
            if (r < .125) u[j][i] = PetscExpReal(-30.0*r*r*r);
            else          u[j][i] = 0.0;
        }
    }

    /*
      Restore vectors
    */
    ierr = DMDAVecRestoreArray(da,U,&u);
    PetscFunctionReturn(0);
}

PetscErrorCode MyCVodeMonitor(PetscInt step,PetscReal ptime,Vec v,void *ctx)
{
    PetscErrorCode ierr;
    PetscReal      norm;

    PetscFunctionBeginUser;
    ierr = VecNorm(v,NORM_2,&norm);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"timestep %D time %g norm %g\n",step,(double)ptime,(double)norm);
    PetscFunctionReturn(0);
}

/*
  MySNESMonitor - illustrate how to set user-defined monitoring routine for SNES.
  Input Parameters:
  snes - the SNES context
  its - iteration number
  fnorm - 2-norm function value (may be estimated)
  ctx - optional user-defined context for private data for the
  monitor routine, as set by SNESMonitorSet()
*/
PetscErrorCode MySNESMonitor(SNES snes,PetscInt its,PetscReal fnorm,PetscViewerAndFormat *vf)
{
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    std::cout << "-------- solution " << "\n";

    ierr = SNESMonitorSolution(snes, its, fnorm, vf);
    std::cout << "------------" << "\n";
    ierr = SNESMonitorDefault(snes,its,fnorm,vf);
    PetscFunctionReturn(0);
}

/* Check function return value...
   opt == 0 means SUNDIALS function allocates memory so check if
   returned NULL pointer
   opt == 1 means SUNDIALS function returns a retval so check if
   retval >= 0
   opt == 2 means function allocates memory so check if returned
   NULL pointer
*/





 





 