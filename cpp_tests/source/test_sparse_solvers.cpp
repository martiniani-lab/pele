/**
 * Tests for sparse krylov solver methods using petsc gradients and hessians
 */



#include "pele/inversepower.hpp"
#include "pele/cell_lists.hpp"
#include "petscviewer.h"
#include "sundials/sundials_nvector.h"
#include <cstddef>
#include <petscerror.h>
#include <petscsys.h>
#include <petscmat.h>
#include <petscvec.h>
#include <petscksp.h>




// sundials imports for petsc
#include <sundials/sundials_types.h>
#include <nvector/nvector_petsc.h>
#include <nvector/nvector_serial.h>
#include <sunnonlinsol/sunnonlinsol_petscsnes.h>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <gtest/gtest.h>
#include <cmath>
#include <memory>


#if defined(SUNDIALS_EXTENDED_PRECISION)
#define GSYM "Lg"
#define ESYM "Le"
#define FSYM "Lf"
#else
#define GSYM "g"
#define ESYM "e"
#define FSYM "f"
#endif

#define NEQ   3                /* number of equations        */
#define TOL   RCONST(1.0e-2)   /* nonlinear solver tolerance */
#define MAXIT 10               /* max nonlinear iterations   */

#define ZERO  RCONST(0.0)  /* real 0.0 */
#define HALF  RCONST(0.5)  /* real 0.5 */
#define ONE   RCONST(1.0)  /* real 1.0 */
#define TWO   RCONST(2.0)  /* real 2.0 */
#define THREE RCONST(3.0)  /* real 3.0 */
#define FOUR  RCONST(4.0)  /* real 4.0 */
#define SIX   RCONST(6.0)  /* real 6.0 */

/* approximate solution */
#define Y1 0.785196933062355226
#define Y2 0.496611392944656396
#define Y3 0.369922830745872357


int Res(N_Vector y, N_Vector f, void *mem)
{
    Vec yvec, fvec;
    realtype vals[3];
    realtype y1, y2, y3;

    yvec = N_VGetVector_Petsc(y);
    fvec = N_VGetVector_Petsc(f);

    /* set vector indices */
    sunindextype indc[3] = {0, 1, 2};

    /* get y vector values */
    VecGetValues(yvec, 3, indc, vals);
    y1 = vals[0]; y2 = vals[1]; y3 = vals[2];

    /* set f vector values */
    vals[0] = y1*y1 + y2*y2 + y3*y3 - ONE;
    vals[1] = TWO * y1*y1 + y2*y2 - FOUR * y3;
    vals[2] = THREE * (y1*y1) - FOUR * y2 + y3*y3;
    VecSetValues(fvec, 3, indc, vals, INSERT_VALUES);

    /* assemble the f vector */
    VecAssemblyBegin(fvec);
    VecAssemblyEnd(fvec);

    return(0);
}


// test version /////////////////////////////////////////////



N_Vector N_VNewEmpty_test()
{   std::cout << "we are here" << "\n";

    N_Vector     v;
    N_Vector_Ops ops;
    std::cout << "we are here 2.0" << "\n";
 
    /* create vector object */
    v = NULL;
    std::cout << "malloc" << "\n";
 
    v = (N_Vector) malloc(sizeof *v);
    std::cout << "malloc done for vector" << "\n";
 
    if (v == NULL) return(NULL);
    std::cout << "we are here 3.0" << "\n";
 
    /* create vector ops structure */
    ops = NULL;
    ops = (N_Vector_Ops) malloc(sizeof *ops);
    if (ops == NULL) { free(v); return(NULL); }
    std::cout << "we are here 4.0" << "\n";
 
    /* initialize operations to NULL */

    /* constructors, destructors, and utility operations */
    ops->nvgetvectorid     = NULL;
    ops->nvclone           = NULL;
    ops->nvcloneempty      = NULL;
    ops->nvdestroy         = NULL;
    ops->nvspace           = NULL;
    ops->nvgetarraypointer = NULL;
    ops->nvsetarraypointer = NULL;
    ops->nvgetcommunicator = NULL;
    ops->nvgetlength       = NULL;

    /* standard vector operations */
    ops->nvlinearsum    = NULL;
    ops->nvconst        = NULL;
    ops->nvprod         = NULL;
    ops->nvdiv          = NULL;
    ops->nvscale        = NULL;
    ops->nvabs          = NULL;
    ops->nvinv          = NULL;
    ops->nvaddconst     = NULL;
    ops->nvdotprod      = NULL;
    ops->nvmaxnorm      = NULL;
    ops->nvwrmsnormmask = NULL;
    ops->nvwrmsnorm     = NULL;
    ops->nvmin          = NULL;
    ops->nvwl2norm      = NULL;
    ops->nvl1norm       = NULL;
    ops->nvcompare      = NULL;
    ops->nvinvtest      = NULL;
    ops->nvconstrmask   = NULL;
    ops->nvminquotient  = NULL;

    /* fused vector operations (optional) */
    ops->nvlinearcombination = NULL;
    ops->nvscaleaddmulti     = NULL;
    ops->nvdotprodmulti      = NULL;

    /* vector array operations (optional) */
    ops->nvlinearsumvectorarray         = NULL;
    ops->nvscalevectorarray             = NULL;
    ops->nvconstvectorarray             = NULL;
    ops->nvwrmsnormvectorarray          = NULL;
    ops->nvwrmsnormmaskvectorarray      = NULL;
    ops->nvscaleaddmultivectorarray     = NULL;
    ops->nvlinearcombinationvectorarray = NULL;

    /* local reduction operations (optional) */
    ops->nvdotprodlocal     = NULL;
    ops->nvmaxnormlocal     = NULL;
    ops->nvminlocal         = NULL;
    ops->nvl1normlocal      = NULL;
    ops->nvinvtestlocal     = NULL;
    ops->nvconstrmasklocal  = NULL;
    ops->nvminquotientlocal = NULL;
    ops->nvwsqrsumlocal     = NULL;
    ops->nvwsqrsummasklocal = NULL;

    /* XBraid interface operations */
    ops->nvbufsize   = NULL;
    ops->nvbufpack   = NULL;
    ops->nvbufunpack = NULL;

    /* debugging functions (called when SUNDIALS_DEBUG_PRINTVEC is defined) */
    ops->nvprint     = NULL;
    ops->nvprintfile = NULL;

    /* attach ops and initialize content to NULL */
    v->ops     = ops;
    v->content = NULL;

    return(v);
}
#define NV_CONTENT_PTC(v)    ( (N_VectorContent_Petsc)(v->content) )

#define NV_LOCLENGTH_PTC(v)  ( NV_CONTENT_PTC(v)->local_length )

#define NV_GLOBLENGTH_PTC(v) ( NV_CONTENT_PTC(v)->global_length )

#define NV_OWN_DATA_PTC(v)   ( NV_CONTENT_PTC(v)->own_data )

#define NV_PVEC_PTC(v)       ( NV_CONTENT_PTC(v)->pvec )

#define NV_COMM_PTC(v) (NV_CONTENT_PTC(v)->comm)


#define ZERO   RCONST(0.0)
#define HALF   RCONST(0.5)
#define ONE    RCONST(1.0)
#define ONEPT5 RCONST(1.5)

/* Error Message */
#define BAD_N1 "N_VNewEmpty_Petsc -- Sum of local vector lengths differs from "
#define BAD_N2 "input global length. \n\n"
#define BAD_N BAD_N1 BAD_N2
N_Vector N_VNewEmpty_Petsc_test(MPI_Comm comm,
                                sunindextype local_length,
                                sunindextype global_length)
{
    std::cout << "starts" << "\n";
 
    N_Vector v;
    N_VectorContent_Petsc content;
    sunindextype n, Nsum;
    PetscErrorCode ierr;

    /* Compute global length as sum of local lengths */
    n = local_length;
    std::cout << "are we even here" << "\n";
 
    ierr = MPI_Allreduce(&n, &Nsum, 1, MPI_SUNINDEXTYPE, MPI_SUM, comm);
    std::cout << "ieerrr" << "\n";
 
    CHKERRABORT(comm,ierr);
    std::cout << "this is it" << "\n";
 
    if (Nsum != global_length) {
        fprintf(stderr, BAD_N);
        return(NULL);
    }

    /* Create an empty vector object */
    v = NULL;
    std::cout << "new empty initialized" << "\n";
 
    v = N_VNewEmpty_test();
    std::cout << "malloc error here" << "\n";

    if (v == NULL) return(NULL);

    /* Attach operations */
    std::cout << "constructors start" << "\n";
 
    /* constructors, destructors, and utility operations */
    v->ops->nvgetvectorid     = N_VGetVectorID_Petsc;
    v->ops->nvclone           = N_VClone_Petsc;
    v->ops->nvcloneempty      = N_VCloneEmpty_Petsc;
    v->ops->nvdestroy         = N_VDestroy_Petsc;
    v->ops->nvspace           = N_VSpace_Petsc;
    v->ops->nvgetarraypointer = N_VGetArrayPointer_Petsc;
    v->ops->nvsetarraypointer = N_VSetArrayPointer_Petsc;
    v->ops->nvgetcommunicator = N_VGetCommunicator_Petsc;
    v->ops->nvgetlength       = N_VGetLength_Petsc;
    std::cout << "vector operators out"
              << "\n";    
 
    /* standard vector operations */
    v->ops->nvlinearsum    = N_VLinearSum_Petsc;
    v->ops->nvconst        = N_VConst_Petsc;
    v->ops->nvprod         = N_VProd_Petsc;
    v->ops->nvdiv          = N_VDiv_Petsc;
    v->ops->nvscale        = N_VScale_Petsc;
    v->ops->nvabs          = N_VAbs_Petsc;
    v->ops->nvinv          = N_VInv_Petsc;
    v->ops->nvaddconst     = N_VAddConst_Petsc;
    v->ops->nvdotprod      = N_VDotProd_Petsc;
    v->ops->nvmaxnorm      = N_VMaxNorm_Petsc;
    v->ops->nvwrmsnormmask = N_VWrmsNormMask_Petsc;
    v->ops->nvwrmsnorm     = N_VWrmsNorm_Petsc;
    v->ops->nvmin          = N_VMin_Petsc;
    v->ops->nvwl2norm      = N_VWL2Norm_Petsc;
    v->ops->nvl1norm       = N_VL1Norm_Petsc;
    v->ops->nvcompare      = N_VCompare_Petsc;
    v->ops->nvinvtest      = N_VInvTest_Petsc;
    v->ops->nvconstrmask   = N_VConstrMask_Petsc;
    v->ops->nvminquotient  = N_VMinQuotient_Petsc;

    /* fused and vector array operations are disabled (NULL) by default */
    std::cout << "reduction operations" << "\n";
 
    /* local reduction operations */
    v->ops->nvdotprodlocal     = N_VDotProdLocal_Petsc;
    v->ops->nvmaxnormlocal     = N_VMaxNormLocal_Petsc;
    v->ops->nvminlocal         = N_VMinLocal_Petsc;
    v->ops->nvl1normlocal      = N_VL1NormLocal_Petsc;
    v->ops->nvinvtestlocal     = N_VInvTestLocal_Petsc;
    v->ops->nvconstrmasklocal  = N_VConstrMaskLocal_Petsc;
    v->ops->nvminquotientlocal = N_VMinQuotientLocal_Petsc;
    v->ops->nvwsqrsumlocal     = N_VWSqrSumLocal_Petsc;
    v->ops->nvwsqrsummasklocal = N_VWSqrSumMaskLocal_Petsc;
std::cout << "xbraid operations" << "\n";
 
    /* XBraid interface operations */
    v->ops->nvbufsize   = N_VBufSize_Petsc;
    v->ops->nvbufpack   = N_VBufPack_Petsc;
    v->ops->nvbufunpack = N_VBufUnpack_Petsc;
    std::cout << "XBRaid" << "\n";

    /* Create content */
    content = NULL;
    std::cout << (sizeof *content) << "\n";
    content = (N_VectorContent_Petsc) malloc(sizeof *content);
    std::cout << (sizeof *content) << "\n";
    std::cout << "malloc issue 2.0" << "\n";
    if (content == NULL) { N_VDestroy(v); return(NULL); }
    // std::cout << "content is not null" << "\n";

    /* Attach content */
 v->content = content;

    /* Initialize content */
    content->local_length  = local_length;
    std::cout << local_length << "\n";
    std::cout << content->local_length << "\n";


    content->global_length = global_length;
    std::cout << global_length << "\n";
    std::cout << content->global_length << "\n";


    content->comm          = comm;
    content->own_data      = SUNFALSE;
    content->pvec          = NULL;

    return(v);}

N_Vector N_VMake_Petsc_test(Vec pvec)
{
    N_Vector v = NULL;
    MPI_Comm comm;
    PetscInt local_length;
    PetscInt global_length;
    VecGetLocalSize(pvec, &local_length);
    VecGetSize(pvec, &global_length);
    std::cout << "petsc object comm" << "\n";
 
    PetscObjectGetComm((PetscObject) pvec, &comm);
    std::cout << "petsc object comm works" << "\n";
    v = N_VNewEmpty_Petsc_test(comm, local_length, global_length);
    std::cout << "object initialized" << "\n";
    if (v == NULL)
        return(NULL);

    /* Attach data */
    NV_OWN_DATA_PTC(v) = SUNFALSE;
    NV_PVEC_PTC(v)     = pvec;

    return(v);
}

// end test

int Jac(SNES snes, Vec y, Mat J, Mat Jpre, void *ctx)
{
    realtype y1, y2, y3;
    realtype yvals[3];

    /* set vector indices */
    sunindextype indc[3] = { 0, 1, 2 };

    /* get y vector values */
    VecGetValues(y, 3, indc, yvals);
    y1 = yvals[0]; y2 = yvals[1]; y3 = yvals[2];

    /* set the Jacobian values */
    realtype jvals[3][3] =  { { TWO*y1,  TWO*y2,  TWO*y3 },
                              { FOUR*y1, TWO*y2, -FOUR   },
                              { SIX*y1, -FOUR,    TWO*y3 } };
    MatSetValues(J, 3, indc, 3, indc, &jvals[0][0], INSERT_VALUES);

    /* assemble the matrix */
    if (J != Jpre) {
        MatAssemblyBegin(Jpre, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(Jpre, MAT_FINAL_ASSEMBLY);
    }
    MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);

    return(0);
}

static int check_retval(void *flagvalue, const char *funcname, int opt)
{
    int *errflag;

    /* Check if the function returned a NULL pointer -- no memory allocated */
    if (opt == 0) {
        if (flagvalue == NULL) {
            fprintf(stderr, "\nERROR: %s() failed -- returned NULL\n\n", funcname);
            return(1);
        } else {
            return(0);
        }
    }

    /* Check if the function returned an non-zero value -- internal failure */
    if (opt == 1) {
        errflag = (int *) flagvalue;
        if (*errflag != 0) {
            fprintf(stderr, "\nERROR: %s() failed -- returned %d\n\n", funcname, *errflag);
            return(1);
        } else {
            return(0);
        }
    }

    /* if we make it here then opt was not 0 or 1 */
    fprintf(stderr, "\nERROR: check_retval failed -- Invalid opt value\n\n");
    return(1);
}


#define TOL   RCONST(1.0e-2)   /* nonlinear solver tolerance */
TEST(SparseSolver, SparseSolverWorks){
    PetscInitializeNoArguments();
    Vec            x, b, u;      /* approx solution, RHS, exact solution */
    Mat            A;            /* linear system matrix */
    KSP            ksp;          /* linear solver context */
    PC             pc;           /* preconditioner context */
    PetscReal      norm;         /* norm of solution error */
    PetscErrorCode ierr;
    PetscInt       i,n = 10,col[3],its;
    PetscMPIInt    size;
    PetscScalar    value[3];

    ierr = VecCreateSeq(PETSC_COMM_SELF, n, &x);
    ierr = VecDuplicate(x, &b);
    ierr = VecDuplicate(x, &u);
    // ierr = MatCreateSeqAIJ(PETSC_COMM_SELF, n, n, 10, NULL, &A);
    ierr = MatCreateSeqSBAIJ(PETSC_COMM_SELF,1, n, n, 10, NULL, &A);
    // MatCreateDense(PETSC_COMM_SELF, n, n, PETSC_DECIDE, PETSC_DECIDE, NULL, &A);
    ierr = MatSetUp(A);    
    /////// Assembly //////
    value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
    for (i=1; i<n-1; i++) {
        col[0] = i-1; col[1] = i; col[2] = i+1;
        ierr   = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);
    }
    i    = n - 1; col[0] = n - 2; col[1] = n - 1;
    ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);
    i    = 0; col[0] = 0; col[1] = 1; value[0] = 2.0; value[1] = -1.0;
    ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
    //////// End Assembly /////
    MatView(A, PETSC_VIEWER_STDOUT_SELF);
    VecSet(u, 1.0);
    MatMult(A, u, b);
    VecView(b, PETSC_VIEWER_STDOUT_SELF);

    // KSPCREATE
    KSPCreate(PETSC_COMM_WORLD,&ksp);
    KSPSetType(ksp, KSPMINRES);
    // operators
    KSPSetOperators(ksp, A, A);

    // preconditioner
    ierr = KSPGetPC(ksp,&pc);
    ierr = PCSetType(pc,PCJACOBI);
    ierr = KSPSetTolerances(ksp,1.e-5,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);

    // solver
    ierr = KSPSolve(ksp,b,x);
    KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD);
    /////// prec

    VecAXPY(x, -1.0, u);
    VecNorm(x, NORM_2, &norm);
    ierr = KSPGetIterationNumber(ksp,&its);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g, Iterations %D\n",(double)norm,its);
    //////
    PetscFinalize();
}



/**
 * SNES cvode tests
 */
TEST(SparseSNES, SparseSNESWorks){

    int                retval = 0;
    N_Vector           y, y0, w;
    SUNNonlinearSolver NLS;
    long int           niters;

    SNES snes;
    Vec X, Y0, Y, W;
    Mat J;
    retval = PetscInitializeNoArguments(); CHKERRV(retval);
    VecCreate(PETSC_COMM_WORLD, &X);
    VecSetSizes(X, PETSC_DECIDE, 3);
    VecSetFromOptions(X);
    VecDuplicate(X, &Y0);
    VecDuplicate(X, &Y);
    VecDuplicate(X, &W);
    // create nvector wrappers
    y0 = N_VMake_Petsc_test(Y0);
    std::cout << "here" << "\n";
    N_VPrint_Petsc(y0);
    std::cout << "here2" << "\n";

    // y0 = N_VNewEmpty_Serial(10);
    
    y  = N_VMake_Petsc(Y);
    
    // y = N_VNewEmpty_Serial(10);
    
    std::cout << "here 2.0" << "\n";

    w = N_VMake_Petsc_test(W);
    // w = N_VNewEmpty_Serial(10);
    std::cout << "w" << "\n";


    /* create Jacobian matrix */
    std::cout << "mat create" << "\n";
    MatCreate(PETSC_COMM_WORLD, &J);
    std::cout << "mat create done" << "\n";

    
    MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, NEQ, NEQ);
    MatSetFromOptions(J);
    MatSetUp(J);

    /* set initial guess */
    VecSet(Y0, 0);
    VecSet(Y, 0.5);
    VecSet(W, 1);
    std::cout << "here is the issue" << "\n";

    /* create SNES context */
    SNESCreate(PETSC_COMM_WORLD, &snes);
    std::cout << "snes creation issue" << "\n";

    SNESSetJacobian(snes, J, J, Jac, NULL);
    std::cout << "here 3" << "\n";
    /* set the maximum number of nonlinear iterations */
    SNESSetTolerances(snes,
                      TOL,
                      PETSC_DEFAULT,
                      PETSC_DEFAULT,
                      MAXIT,
                      PETSC_DEFAULT);
    SNESSetFromOptions(snes);


    /* create nonlinear solver */
    NLS = SUNNonlinSol_PetscSNES(y, snes);


    // if (check_retval((void *)NLS, "SUNNonlinSol_PetscSNES", 0)) return(1);

    /* set the nonlinear residual function */
    retval = SUNNonlinSolSetSysFn(NLS, Res);
    // if (check_retval(&retval, "SUNNonlinSolSetSysFn", 1)) return(1);

    /* solve the nonlinear system */
    retval = SUNNonlinSolSolve(NLS, y0, y, w, TOL, SUNFALSE, NULL);
    // if (check_retval(&retval, "SUNNonlinSolSolve", 1)) return(1);
    std::cout << "here 4" << "\n";

    /* get the solution */
    realtype yvals[3];
    sunindextype indc[3] = {0, 1, 2};
    VecGetValues(Y, 3, indc, yvals);

    /* print the solution */
    printf("Solution:\n");
    printf("y1 = %"GSYM"\n", yvals[0]);
    printf("y2 = %"GSYM"\n", yvals[1]);
    printf("y3 = %"GSYM"\n", yvals[2]);

    /* print the solution error */
    printf("Solution Error:\n");
    printf("e1 = %"GSYM"\n", yvals[0] - Y1);
    printf("e2 = %"GSYM"\n", yvals[1] - Y2);
    printf("e3 = %"GSYM"\n", yvals[2] - Y3);

    /* get the number of linear iterations */
    retval = SUNNonlinSolGetNumIters(NLS, &niters);
    // if (check_retval(&retval, "SUNNonlinSolGetNumIters", 1)) return(1);

    printf("Number of nonlinear iterations: %ld\n",niters);

    /* Free vector, matrix, and nonlinear solver */
    VecDestroy(&X);
    VecDestroy(&Y0);
    VecDestroy(&Y);
    VecDestroy(&W);
    MatDestroy(&J);
    SNESDestroy(&snes);
    N_VDestroy(y);
    N_VDestroy(y0);
    N_VDestroy(w);
    SUNNonlinSolFree(NLS);

    /* Print result */
    if (retval) {
        printf("FAIL\n");
    } else {
        printf("SUCCESS\n");
    }

    std::cout << retval << "retvall \n";

}
