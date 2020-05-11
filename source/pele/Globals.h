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

#ifndef GLOBALS_H
#define GLOBALS_H

#include <gsl/gsl_rng.h>


// Boltzmann constant
const double kB = 1.0;

// state variable indicating whether the current simulation has been
// restarted from a previous simulation (0 = no; 1 = yes)
extern int RestartFlag;

// pointer to an instance of a random number generator (see GSL docs
// at https://www.gnu.org/software/gsl/doc/html/rng.html)
extern gsl_rng* rng;


// calculates the factorial (n!) of an unsigned long int
unsigned long int Nfac(unsigned long int);
// calculates the power of N to M (N^M)
unsigned long int NtoM(unsigned long int, unsigned long int);
// calculates the Kronecker delta function of two long integers
int DeltaFunc(long int, long int);


#endif
