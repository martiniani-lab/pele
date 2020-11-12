/*
 * Implementation of PETSC code for sparse symmetric matrices
 * Written with reference to petsc matrix and N_Vector PETSC implementations
 */


#ifndef _SUNMATRIX_PETSC_SPARSE_SYMMETRIC_H
#define _SUNMATRIX_PETSC_SPARSE_SYMMETRIC_H


#include "sundials/sundials_export.h"
#include "sunmatrix/sunmatrix_dense.h"
#include <stdio.h>
#include <sundials/sundials_matrix.h>
#include <petscmat.h>
#include <sundials/sundials_mpi_types.h>

#ifdef __cplusplus  /* wrapper to enable C++ usage */
extern "C" {
#endif



/**
 * @brief      PETSC SEQSBAIJ WRAPPER
 */
struct _SUNMatrixContent_PETSc {
    Mat pmat;                  /* petsc matrix */
    booleantype own_data;      /* whether the data is owned*/
};
typedef struct _SUNMatrixContent_PETSc *SUNMatrixContent_PETSc;


/**
 * @brief  Constructor for creating PETSc SEQSBAIJ Matrix owned by SUNMatrix_PETSc. constructor arguments are same as for PETSc
 */
SUNDIALS_EXPORT SUNMatrix SUNMatPETScSeqSBAIJ();
/**
 * @brief      Constructor for creating PETSc SEQSBAIJ Matrix not owned by SUNMatrix_PETSc,
 *             it assumes creation and destruction of PETSc object are handled elsewhere
 * @param      Mat A ()
 *
 * @return     return type
 */
SUNDIALS_EXPORT SUNMatrix SUNMatMake_PETScSeqSBAIJ(Mat A);


/**
 * @brief      gets the number of rows
 */
SUNDIALS_EXPORT sunindextype SUNMatPETScSeqSBAIJ_Rows(SUNMatrix A);
/**
 * @brief      gets the number of columns
 */
SUNDIALS_EXPORT sunindextype SUNMatPETScSeqSBAIJ_Columns(SUNMatrix A);
SUNDIALS_EXPORT SUNMatrix SUNMatClone_PETScSeqSBAIJ(SUNMatrix A);
SUNDIALS_EXPORT void SUNMatDestroy_PETScSeqSBAIJ(SUNMatrix A);
SUNDIALS_EXPORT SUNMatrix_ID SUNMatGetID_PETScSeqSBAIJ(SUNMatrix A);
#ifdef __cplusplus
}
#endif
#endif