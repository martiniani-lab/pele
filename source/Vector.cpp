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

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "pele/Vector.h"


Vector::Vector(int d, CoordsType s)
{
  int i;
  dim = d;
  c = new CoordsType[dim];
  for (i = 0; i < dim; i++)
    c[i] = s;
}

Vector::Vector(const Vector& v)
{
  int i;
  dim = v.dim;
  c = new CoordsType[dim];
  for (i = 0; i < dim; i++)
    c[i] = v.c[i];
}

Vector::~Vector()
{
  delete[] c;
}

void Vector::Set(int i, const CoordsType s)
{
  c[i] = s;
}

CoordsType Vector::Get(int i)
{
  return c[i];
}

void Vector::Print()
{
  int i;
  for (i = 0; i < dim; i++)
    printf(" %5ld", c[i]);
  printf("\n");
}

void Vector::Copy(const Vector& v)
{
  int i;
  for (i = 0; i < dim; i++)
    c[i] = v.c[i];
}

void Vector::Copy(const Vector& v1, const Vector& v2)
{
  int i;
  for (i = 0; i < dim; i++)
    c[i] = v2.c[i] - v1.c[i];
}

void Vector::Add(const Vector& v)
{
  int i;
  for (i = 0; i < dim; i++)
    c[i] += v.c[i];
}

void Vector::Add(const Vector& v1, const Vector& v2)
{
  int i;
  for (i = 0; i < dim; i++)
    c[i] += v2.c[i] - v1.c[i];
}

void Vector::Sum(const Vector& v1, const Vector& v2)
{
  int i;
  for (i = 0; i < dim; i++)
    c[i] = v1.c[i] + v2.c[i] - c[i];
}

void Vector::Sub(const Vector& v)
{
  int i;
  for (i = 0; i < dim; i++)
    c[i] -= v.c[i];
}

void Vector::Sub(const Vector& v1, const Vector& v2)
{
  int i;
  for (i = 0; i < dim; i++)
    c[i] -= v2.c[i] - v1.c[i];
}

void Vector::ScalarMul(const CoordsType s)
{
  int i;
  for (i = 0; i < dim; i++)
    c[i] *= s;
}

CoordsType Vector::Norm2()
{
  int i;
  CoordsType s = 0;
  for (i = 0; i < dim; i++)
    s += c[i] * c[i];
  return s;
}

CoordsType Vector::ScalarProduct(const Vector& v)
{
  int i;
  CoordsType s = 0;
  for (i = 0; i < dim; i++)
    s += c[i] * v.c[i];
  return s;
}

CoordsType Vector::ScalarProduct(const Vector& v1, const Vector& v2)
{
  int i;
  CoordsType s = 0;
  for (i = 0; i < dim; i++)
    s += (v1.c[i] - c[i]) * (v2.c[i] - c[i]);
  return s;
}

CoordsType Vector::Distance2(const Vector& v)
{
  int i;
  CoordsType d;
  CoordsType s = 0;
  for (i = 0; i < dim; i++) {
    d = v.c[i] - c[i];
    s += d * d;
  }
  return s;
}

CoordsType Vector::L1Distance(const Vector& v)
{
  int i;
  CoordsType s = 0;
  for (i = 0; i < dim; i++)
    if (v.c[i] < c[i]) s += c[i] - v.c[i];
    else s += v.c[i] - c[i];
  return s;
}

int Vector::Overlap(const Vector& v)
{
  int i;
  for (i = 0; i < dim; i++)
    if (v.c[i] != c[i]) return 0;
  return 1;
}

int Vector::Overlap(const Vector& v, const CoordsType r)
{
  int i;
  CoordsType d;
  CoordsType s = 0;
  for (i = 0; i < dim; i++) {
    d = v.c[i] - c[i];
    s += d * d;
  }
  if (s <= (r*r)) return 1;
  else return 0;
}

int Vector::GetNormalDim(const Vector& v)
{
  int i;
  for (i = 0; i < dim; i++)
    if (v.c[i] != c[i]) return i;
  return -1;
}


Matrix::Matrix(int r, int c, CoordsType s)
{
  int i, j;
  rows = r;
  cols = c;
  e = new CoordsType*[rows];
  for (i = 0; i < rows; i++) {
    e[i] = new CoordsType[cols];
    for (j = 0; j < cols; j++)
      e[i][j] = s;
  }
}

Matrix::Matrix(const Matrix& m)
{
  int i, j;
  rows = m.rows;
  cols = m.cols;
  e = new CoordsType*[rows];
  for (i = 0; i < rows; i++) {
    e[i] = new CoordsType[cols];
    for (j = 0; j < cols; j++)
      e[i][j] = m.e[i][j];
  }
}

Matrix::~Matrix()
{
  int i;
  for (i = 0; i < rows; i++)
    delete[] e[i];
  delete[] e;
}

void Matrix::Set(int i, int j, const CoordsType s)
{
  e[i][j] = s;
}

CoordsType Matrix::Get(int i, int j)
{
  return e[i][j];
}

void Matrix::Print()
{
  int i, j;
  printf(" %d  %d\n", rows, cols);
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++)
      printf(" %4ld", e[i][j]);
    printf("\n");
  }
}

void Matrix::Copy(const Matrix& m)
{
  int i, j;
  for (i = 0; i < rows; i++)
    for (j = 0; j < cols; j++)
      e[i][j] = m.e[i][j];
}

void Matrix::Add(const Matrix& m)
{
  int i, j;
  for (i = 0; i < rows; i++)
    for (j = 0; j < cols; j++)
      e[i][j] += m.e[i][j];
}

void Matrix::Sub(const Matrix& m)
{
  int i, j;
  for (i = 0; i < rows; i++)
    for (j = 0; j < cols; j++)
      e[i][j] -= m.e[i][j];
}

void Matrix::ScalarMul(const CoordsType s)
{
  int i, j;
  for (i = 0; i < rows; i++)
    for (j = 0; j < cols; j++)
      e[i][j] *= s;
}

void Matrix::MatrixMatrixMul(const Matrix& m, Matrix& mr)
{
  int i, j, k;
  for (i = 0; i < mr.rows; i++)
    for (j = 0; j < mr.cols; j++) {
      mr.e[i][j] = 0;
      for (k = 0; k < cols; k++)
	mr.e[i][j] += e[i][k] * m.e[k][j];
    }
}

void Matrix::MatrixVectorMul(const Vector& v, Vector& vr)
{
  int i, j;
  for (i = 0; i < vr.dim; i++) {
    vr.c[i] = 0;
    for (j = 0; j < cols; j++)
      vr.c[i] += e[i][j] * v.c[j];
  }
}

void Matrix::PivotOperation(const Vector& p, const Vector& v, Vector& vr)
{
  int i, j;
  for (i = 0; i < vr.dim; i++) {
    vr.c[i] = 0;
    for (j = 0; j < cols; j++)
      vr.c[i] += e[i][j] * (v.c[j] - p.c[j]);
    vr.c[i] += p.c[i];
  }
}

int Matrix::RelativeOverlap(const Vector& p, const Vector& v, const Vector& vref)
{
  int i, j;
  CoordsType d;
  for (i = 0; i < vref.dim; i++) {
    d = 0;
    for (j = 0; j < cols; j++)
      d += e[i][j] * (v.c[j] - p.c[j]);
    if (d != vref.c[i]) return 0;
  }
  return 1;
}
