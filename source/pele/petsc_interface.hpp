/**
 * Wrapper functions for Pele arrays interfacing with Vec and Mat from Petsc and
 * SUNDIALS
 */


#ifndef _PELE_PETSC_CVODE_INTERFACE
#define _PELE_PETSC_CVODE_INTERFACE
#include "nvector/nvector_petsc.h"
#include "pele/array.hpp"
#include "sundials/sundials_nvector.h"
#include "petscvec.h"




/**
 * creates a new N_vector_petsc array array that wraps around the pele array
 * data
 */
inline N_Vector N_Vector_eq_pele(pele::Array<double> x) {
  N_Vector y;
  Vec y_petsc;
  VecCreateSeq(PETSC_COMM_SELF, x.size(), &y_petsc);
  for (size_t i = 0; i < x.size(); ++i) {
    VecSetValue(y_petsc, i, x[i], INSERT_VALUES);
  }
  VecAssemblyBegin(y_petsc);
  VecAssemblyEnd(y_petsc);
  y = N_VMake_Petsc(y_petsc);
  return y;
}


/**
 * Creates a new Vec object with data from pele array (cannot wrap data due to
 * PETSc data encapsulation)
 * involves creation of new stuff
 */
inline void PetscVec_eq_pele(Vec &x_petsc, pele::Array<double> x) {
    VecCreateSeq(PETSC_COMM_SELF, x.size(), &x_petsc);
  for (auto i = 0; i < x.size(); ++i) {
    VecSetValue(x_petsc, i, x[i], INSERT_VALUES);
  }
  VecAssemblyBegin(x_petsc);
  VecAssemblyEnd(x_petsc);
}

/**
 * Creates a new pele array with data belonging to an N_Vector petsc arra
 */

inline pele::Array<double> pele_eq_N_Vector(N_Vector x) {
  Vec x_petsc = N_VGetVector_Petsc(x);
  double *x_data;
  VecGetArray(x_petsc, &x_data);
  return pele::Array<double>(x_data, N_VGetLength(x)).copy();
}

/**
 * gets the array data and wraps it into a pele Array.
 * Note: pele arrays are single processor only Note: does not
 * work if restore array is not called later
 */
inline pele::Array<double> pele_eq_PetscVec(Vec x) {
  double *x_arr;
  VecGetArray(x, &x_arr);
  PetscInt length;
  VecGetLocalSize(x, &length);
  return pele::Array<double>(x_arr, (int)length);
}


#endif  // end _PELE_PETSC_CVODE_INTERFACE