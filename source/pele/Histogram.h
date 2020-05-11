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

#ifndef HISTOGRAM_H
#define HISTOGRAM_H


// Type of measured quantities:
// for discrete variables --> long int
// for continuous variables --> double

//typedef double ObservableType;
typedef long int ObservableType;


/*
  Histogram class:

  This class provides data structures and routines for the storing and
  modifying of multi-dimensional histogram data (i.e. DOS, histogram,
  and mask) used for Wang-Landau sampling. Internally, this data is
  stored in dynamically allocated one-dimensional arrays.

  Notes:

  - Non-physical dimensions follow AFTER physical dimensions.

  - Non-physical observables and corresponding histogram intervals
    (dimensions) must ALWAYS be defined such that each non-physical
    observable value which maps to the first entry/bin (zero's index)
    of its corresponding non-physical interval (dimension) represents
    the "physical state" of the system.
*/

class Histogram {

public:

  // constructors:

  // generates a "Histogram" class instance either from an input file
  // or from a "Histogram" class state file
  // 1. file name
  // 2. flag: 0 = any file type (usually, the input file)
  //          1 = only 'Histogram' class state file (HSF) type
  Histogram(const char*, int = 0);

  // generates a "Histogram" class instance
  // 1. number of dimensions
  // 2. number of physical dimensions
  // 3. array with the boundary specifications for each dimension
  //    i.e. minimum, maximum of the interval and bin size
  Histogram(int, int, ObservableType[][3]);

  // destructor
  ~Histogram();

  // writes the entire histogram state
  // (i.e. specs. and data) to a file
  // 1. a counter
  // 2. file name
  // 3. flag: true = shift DOS to min. DOS; false = don't
  void SaveState(const unsigned long int = 0, const char* = NULL, const bool = true);

  // returns min. and max. of the DOS
  // 1. flag: 0 = all dim.; 1 = only physical dim.
  double GetDOSMin(int = 0);
  double GetDOSMax(int = 0);

  // returns min. and max. of the histogram
  // 1. flag: 0 = all dim.; 1 = only physical dim.
  // 2. flag: 0 = incl. "zeros"; 1 = excl. "zeros"
  unsigned long int GetHistMin(int = 0, int = 1);
  unsigned long int GetHistMax(int = 0, int = 1);

  // prints histogram specifications (e.g. number of dimensions and
  // bins, boundaries, etc.)
  void PrintSpecs();

  // prints normalized (to unity) DOS
  // 1. file name
  // 2. flag: 0 = all dim.; 1 = only physical dim.
  void PrintNormDOS(const char* = NULL, int = 1);

  // checks the histogram for Wang-Landau sampling
  // 1. check type: 0 = flatness check (flatness criterion)
  //                1 = minimal number of hits per bin
  // 2. flatness criterion (0 - 1)
  // 3. minimal number of hits per bin
  int CheckHistogram(int, double, unsigned long int);

  // masks all bins
  void MaskAll();

  // resets the histogram
  void ResetHist();

  // returns the histogram array index or -1 if out of boundaries
  // 1. 1D array of observables
  // 2. flag: 0 = all bins; 1 = only within masked bins
  long int GetIndex(ObservableType[], int = 0);

  // updates the DOS, histogram and mask for Wang-Landau sampling
  // 1. histogram array index
  // 2. reference to the counter of the histogram check interval
  // 3. update type
  // 4. DOS modification factor
  void Update(long int, unsigned long int&, int, double);

  // gets DOS, histogram and mask ("inline")
  // 1. histogram array index
  double GetDOS(const long int index) {
    return dos[index];
  }
  int GetMask(const long int index) {
    return mask[index];
  }
  unsigned long int GetHist(const long int index) {
    return hist[index];
  }

  // records and prints the "up" and "down" round trip times
  // returns 1 in case of an up-walk, otherwise returns 0
  // 1. histogram array index
  // 2. current number of Monte Carlo steps
  // 3. reference to a variable that stores the MC steps
  // 4. dimension for which the round trips are to be checked
  // 5. file name (optional)
  int CheckItinerancy(long int, const unsigned long int, unsigned long int&,
		      int = 0, const char* = NULL);

private:

  int Dim;      // total number of dimensions
  int DimPhys;  // number of physical dimensions

  long int Bins;             // total number of bins
  long int BinsPhys;         // number of physical bins
  long int BinsNonPhys;      // Bins / BinsPhys
  long int BinsMasked;       // total number of masked bins
  long int BinsMaskedPhys;   // number of physical masked bins
  long int BinsVisited;      // total number of currently visited bins
  long int BinsVisitedPhys;  // number of currently physical visited bins

  unsigned long int Hits;      // sum of histogram entries
  unsigned long int HitsPhys;  // sum of physical histogram entries

  ObservableType* HMin;   // pointer to 1D array with histogram boundary minima
  ObservableType* HMax;   // pointer to 1D array with histogram boundary maxima
  ObservableType* HSize;  // pointer to 1D array with histogram bin sizes

  long int* HBins;  // pointer to 1D array with histogram sizes (for each dim.)
  long int* HInt;   // pointer to 1D array with histogram "size offsets"

  int* mask;                // pointer to the 'mask' (1D array)
  double* dos;              // pointer to the DOS (1D array)
  unsigned long int* hist;  // pointer to the histogram (1D array)

  int YoYo;  // flag used by CheckItinerancy: 0 = "down walk"; 1 = "up walk"

  // prints error messages on standard error stream
  // 1. error code
  // 2. string argument (optional)
  void ErrorMsg(int, const char* = NULL);

};


#endif
