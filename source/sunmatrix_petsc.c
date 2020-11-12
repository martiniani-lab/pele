/* -----------------------------------------------------------------
 * Programmer(s): Praharsh
 * -----------------------------------------------------------------
 * Based on N_Vector_Petsc and SunMatrix_dense by Scott D. Cohen, Alan C.
 * Hindmarsh, Radu Serban, and Aaron Collier @ LLNL
 * -----------------------------------------------------------------
 * This is the implementation file for a PETSc implementation
 * of the SUNMatrix package.
 * -----------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>

#include "pele/sunmatrix_petsc.h"
#include "petscmat.h"
#include "sundials/sundials_matrix.h"
#include "sundials/sundials_nvector.h"
#include <sundials/sundials_math.h>

#define ZERO RCONST(0.0)
#define HALF RCONST(0.5)
#define ONE RCONST(1.0)
#define ONEPT5 RCONST(1.5)

#define MAT_CONTENT_PTC(sunmat) ((SUNMatrixContent_PETSc)(sunmat->content))
#define MAT_OWN_DATA(sunmat) (MAT_CONTENT_PTC(sunmat)->own_data)
#define MAT_PETSC_MAT(sunmat) (MAT_CONTENT_PTC(sunmat)->pmat)

SUNMatrix SUNMatPETScSeqSBAIJ()
{
  SUNMatrix A;
  /* we're going to be deferring the errors to PETSc */
  SUNMatrixContent_PETSc content;
  A = NULL;
  A = SUNMatNewEmpty();
  if (A == NULL) {
      return NULL;
  };
  /* attach operations */
  A->ops->getid = SUNMatGetID_PETScSeqSBAIJ;
  A->ops->clone = SUNMatClone_PETScSeqSBAIJ;
  A->ops->destroy = SUNMatDestroy_PETScSeqSBAIJ;

  /* create content */
  content = NULL;
  content = (SUNMatrixContent_PETSc)malloc(sizeof *content);
  if (content == NULL) {
    SUNMatDestroy(A);
    return (NULL);
  }
  /* attach content*/
  /* note that the matrix details are only stored within the PETSC object */
  A->content = content;

  /* initialize content */
  content->own_data = SUNFALSE;
  content->pmat = NULL;
  return A;
};

SUNMatrix SUNMatMake_PETScSeqSBAIJ(Mat pmat) {
  SUNMatrix smat = NULL;
  smat = SUNMatPETScSeqSBAIJ();
  if (smat == NULL)
    return NULL;
  /* Attach data */
  MAT_OWN_DATA(smat) = SUNFALSE;
  MAT_PETSC_MAT(smat) = pmat;
  return (smat);
}

SUNMatrix SUNMatClone_PETScSeqSBAIJ(SUNMatrix A)
{
  Mat A_PETSc = MAT_PETSC_MAT(A);
  SUNMatrix B = SUNMatPETScSeqSBAIJ();
  Mat B_PETSc;
  MatDuplicate(A_PETSc, MAT_COPY_VALUES, &B_PETSc);
  /* Attach data */
  MAT_OWN_DATA(B) = SUNTRUE;
  MAT_PETSC_MAT(B) = B_PETSc;
};

void SUNMatDestroy_PETScSeqSBAIJ(SUNMatrix A)
{
  if (A == NULL)
    return;

  /* free content */
  if (A->content != NULL) {
    if (MAT_OWN_DATA(A) && MAT_PETSC_MAT(A) != NULL) {
      MatDestroy(&(MAT_PETSC_MAT(A)));
      MAT_PETSC_MAT(A) = NULL;
    }
    free(A->content);
    A->content = NULL;
  }

  /* free ops */
  if (A->ops != NULL) {
    free(A->ops);
    A->ops = NULL;
  }
  /* free matrix */
  free(A);
  A = NULL;
};



SUNMatrix_ID SUNMatGetID_PETScSeqSBAIJ(SUNMatrix A)
{
  return SUNMATRIX_CUSTOM;
}
