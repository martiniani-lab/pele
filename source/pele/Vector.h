/*
 *
 * MCSim - Monte Carlo simulation of lattice polymers and proteins
 *
 * Copyright (C) 2006 - 2019 Thomas Wuest <monte.carlo.coder@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 */

#ifndef VECTOR_H
#define VECTOR_H


// Type of coordinate variables:
// on-lattice --> long int
// off-lattice --> double

//typedef double CoordsType;
typedef long int CoordsType;


// see below...
class Vector;


/*
  Matrix class:

  This class provides a data structure to store m x n matrices of any
  size (dynamically allocated) and various routines for the handling
  of matrix-matrix and matrix-vector operations, e.g. matrix-matrix
  addition/multiplication, matrix-vector multiplication, etc.
*/

class Matrix {

public:

  // constructors:

  // generates a new matrix
  // 1. number of rows
  // 2. number of columns
  // 3. initial value for all matrix elements
  Matrix(int = 1, int = 1, CoordsType = 0);
  // copy constructor
  Matrix(const Matrix&);
  // destructor
  ~Matrix();

  // returns a reference to the [i,j]'th matrix element
  // --> "inline" for efficient reading and writing
  CoordsType& Elem(int i, int j) {
    return e[i][j];
  }

  // the subsequent routines are mostly self-explanatory...

  void Set(int, int, const CoordsType);
  CoordsType Get(int, int);
  void Print();

  void Copy(const Matrix&);
  void Add(const Matrix&);
  void Sub(const Matrix&);
  void ScalarMul(const CoordsType);
  void MatrixMatrixMul(const Matrix&, Matrix&);

  void MatrixVectorMul(const Vector&, Vector&);

  // special matrix-vector multiplication that performs
  // the "pivot operation"
  // 1. reference to vector which defines the "pivot center"
  // 2. reference to vector on which the operation is performed
  // 3. reference to vector which stores the result
  void PivotOperation(const Vector&, const Vector&, Vector&);

  // checks the overlap between two vectors after pivot operation
  // 1. reference to vector which defines the "pivot center"
  // 2. reference to vector on which the operation is performed
  // 3. reference to vector with which the overlap is checked
  // returns 1 if overlap, 0 otherwise
  int RelativeOverlap(const Vector&, const Vector&, const Vector&);

private:

  int rows;        // number of rows
  int cols;        // number of columns
  CoordsType** e;  // pointer to pointer to array with matrix elements

};


/*
  Vector class:

  This class provides a data structure to store vectors of any size
  (dynamically allocated) and various routines for the handling of
  vector operations, e.g. vector-vector addition, scalar product,
  distance between two vectors, etc.
*/

class Vector {

  // These routines are defined in the class "Matrix". They are
  // declared "friend" of the class "Vector" for efficient
  // computation.
  friend void Matrix::MatrixVectorMul(const Vector&, Vector&);
  friend void Matrix::PivotOperation(const Vector&, const Vector&, Vector&);
  friend int Matrix::RelativeOverlap(const Vector&, const Vector&, const Vector&);

public:

  // constructors:

  // generates a new vector
  // 1. number of vector components
  // 2. initial value for all coordinates
  Vector(int = 1, CoordsType = 0);
  // copy constructor
  Vector(const Vector&);
  // destructor
  ~Vector();

  // returns a reference to the i'th vector component
  // --> "inline" for efficient reading and writing
  CoordsType& Elem(int i) {
    return c[i];
  }

  // the subsequent routines are mostly self-explanatory

  void Set(int, const CoordsType);
  CoordsType Get(int);
  void Print();

  void Copy(const Vector&);
  void Copy(const Vector&, const Vector&);
  void Add(const Vector&);
  void Add(const Vector&, const Vector&);
  void Sum(const Vector&, const Vector&);
  void Sub(const Vector&);
  void Sub(const Vector&, const Vector&);
  void ScalarMul(const CoordsType);
  CoordsType Norm2();
  CoordsType ScalarProduct(const Vector&);
  CoordsType ScalarProduct(const Vector&, const Vector&);
  CoordsType Distance2(const Vector&);
  CoordsType L1Distance(const Vector&);

  // checks the overlap between two vectors (on-lattice)
  // returns 1 if overlap, 0 otherwise
  int Overlap(const Vector&);

  // checks the overlap between two vectors
  // off-lattice --> 2. threshold
  // returns 1 if overlap, 0 otherwise
  int Overlap(const Vector&, const CoordsType);

  // returns dimension of first non-equal component between two
  // vectors
  int GetNormalDim(const Vector&);

private:

  int dim;        // number of vector components
  CoordsType* c;  // pointer to 1D array with vector coordinates

};


#endif
