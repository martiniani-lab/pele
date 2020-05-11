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

#include "pele/Globals.h"


int RestartFlag = 0;

gsl_rng* rng = NULL;


unsigned long int Nfac(unsigned long int n)
{
  unsigned long int k, v;
  v = 1;
  for (k = 2; k <= n; k++)
    v *= k;
  return v;
}


unsigned long int NtoM(unsigned long int n, unsigned long int m)
{
  unsigned long int k, v;
  v = 1;
  for (k = 1; k <= m; k++)
    v *= n;
  return v;
}


int DeltaFunc(long int i, long int j)
{
  if (i == j) return 1;
  else return 0;
}
