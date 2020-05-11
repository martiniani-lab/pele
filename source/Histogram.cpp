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
#include <string.h>

#include "pele/Histogram.h"


Histogram::Histogram(const char* filename, int OnlyHSF)
{

  char line[200];
  char pname[51];
  FILE* f;

  int i, j;
  long int n;
  int flag = 0;

  // initialize 'Histogram' class parameters
  Dim = 1;
  DimPhys = 1;
  Bins = 1;
  BinsPhys = 1;
  BinsNonPhys = 1;
  BinsMasked = 0;
  BinsMaskedPhys = 0;
  BinsVisited = 0;
  BinsVisitedPhys = 0;
  Hits = 0;
  HitsPhys = 0;
  YoYo = 1;

  // read 'Histogram' class parameters...

  if (OnlyHSF) {
    f = fopen(filename, "r");
    if (f == NULL) ErrorMsg(6, filename);
  }
  else {
    // --> first, check "hdata_current.dat"
    f = fopen("hdata_current.dat", "r");
    // if not available..., check "hdata_init.dat"
    if (f == NULL) f = fopen("hdata_init.dat", "r");
  }

  // read 'Histogram' class parameters from input file
  if (f == NULL) {

    f = fopen(filename, "r");
    if (f == NULL) ErrorMsg(1, filename);

    while (fgets(line, sizeof(line), f) != NULL) {

      if ((line[0] != '#') && (line[0] != '\n')) {

	if (sscanf(line, "%50s", pname) == 1) {

	  if (!strcmp(pname, "HistogramDims")) {
	    if (sscanf(line, "%*50s %d %d", &Dim, &DimPhys) != 2) ErrorMsg(2, pname);
	    if (DimPhys < 1) ErrorMsg(3, pname);
	    if (Dim < DimPhys) ErrorMsg(3, pname);
	    flag = 1;
	    i = 0;
	    HMin  = new ObservableType[Dim];
	    HMax  = new ObservableType[Dim];
	    HSize = new ObservableType[Dim];
	    HBins = new long int[Dim];
	    HInt  = new long int[Dim];
	    continue;
	  }

	  if (!strcmp(pname, "HistogramSpecs")) {
	    if (flag) {
	      if (i == Dim) ErrorMsg(5);
	      if (sscanf(line, "%*50s %ld %ld %ld",
			 &HMin[i],
			 &HMax[i],
			 &HSize[i]) != 3) ErrorMsg(2, pname);
	      if (HMin[i] >= HMax[i]) ErrorMsg(3, pname);
	      if (HSize[i] <= (ObservableType)(0)) ErrorMsg(3, pname);
	      HBins[i] = (long int)((HMax[i] - HMin[i]) / HSize[i]);
	      // required to keep within array boundaries:
	      HMax[i] = HMin[i] + (ObservableType)(HBins[i]) * HSize[i];
	      Bins *= HBins[i];
	      if (i < DimPhys) BinsPhys *= HBins[i];
	      i++;
	    }
	    else {
	      fprintf(stderr, "\nWarning in 'Histogram': ");
	      fprintf(stderr, "Parameter '%s' ignored.\n", pname);
	    }
	    continue;
	  }

	}

	else ErrorMsg(4);

      }

    }

    fclose(f);

    if ((flag == 0) || (i < Dim)) ErrorMsg(5);

    BinsNonPhys = Bins / BinsPhys;

    for (i = Dim - 1; i >= 0; i--) {
      HInt[i] = 1;
      for (j = i + 1; j < Dim; j++)
	HInt[i] *= HBins[j];
    }

    dos  = new double[Bins];
    mask = new int[Bins];
    hist = new unsigned long int[Bins];

    for (n = 0; n < Bins; n++) {
      dos[n] = 0.0;
      mask[n] = 0;
      hist[n] = 0;
    }

  }

  // read 'Histogram' class parameters from Histogram state file
  else {

    fscanf(f, "%*[^\n]"); fscanf(f, "%*c");
    fscanf(f, "%*[^\n]"); fscanf(f, "%*c");

    if (fscanf(f, "%*s %d", &Dim) != 1) ErrorMsg(6, filename);
    if (fscanf(f, "%*s %d", &DimPhys) != 1) ErrorMsg(6, filename);

    if (fscanf(f, "%*s %ld", &Bins) != 1) ErrorMsg(6, filename);
    if (fscanf(f, "%*s %ld", &BinsPhys) != 1) ErrorMsg(6, filename);
    if (fscanf(f, "%*s %ld", &BinsMasked) != 1) ErrorMsg(6, filename);
    if (fscanf(f, "%*s %ld", &BinsMaskedPhys) != 1) ErrorMsg(6, filename);
    if (fscanf(f, "%*s %ld", &BinsVisited) != 1) ErrorMsg(6, filename);
    if (fscanf(f, "%*s %ld", &BinsVisitedPhys) != 1) ErrorMsg(6, filename);

    if (fscanf(f, "%*s %lu", &Hits) != 1) ErrorMsg(6, filename);
    if (fscanf(f, "%*s %lu", &HitsPhys) != 1) ErrorMsg(6, filename);

    // currently, error checking disabled for 'YoYo' to guarantee
    // compatibility with old Histogram state files
    if (fscanf(f, "%*s %d", &YoYo) != 1) {
      YoYo = 1;
      fprintf(stdout, "Warning: 'YoYo' set to default value (1)\n");
    }
    else {
      fscanf(f, "%*c");
      fscanf(f, "%*[^\n]"); fscanf(f, "%*c");
    }

    fscanf(f, "%*[^\n]"); fscanf(f, "%*c");

    BinsNonPhys = Bins / BinsPhys;

    HMin  = new ObservableType[Dim];
    HMax  = new ObservableType[Dim];
    HSize = new ObservableType[Dim];
    HBins = new long int[Dim];
    HInt  = new long int[Dim];

    dos  = new double[Bins];
    mask = new int[Bins];
    hist = new unsigned long int[Bins];

    for (i = 0; i < Dim; i++) {
      if (fscanf(f, "%*d %ld %ld %ld %ld",
		 //      if (fscanf(f, "%*d %lf %lf %ld %ld",
		 &HMin[i], &HSize[i], &HBins[i], &HInt[i]) != 4)
	ErrorMsg(6, filename);
      HMax[i] = HMin[i] + (ObservableType)(HBins[i]) * HSize[i];
    }

    fscanf(f, "%*c");
    fscanf(f, "%*[^\n]"); fscanf(f, "%*c");
    fscanf(f, "%*[^\n]"); fscanf(f, "%*c");

    for (n = 0; n < Bins; n++) {
      if (fscanf(f, "%d %lf %lu", &mask[n], &dos[n], &hist[n]) != 3) ErrorMsg(6, filename);
      if (!mask[n]) dos[n] = 0.0;
    }

    fclose(f);

  }

  PrintSpecs();

}


Histogram::Histogram(int d, int dphys, ObservableType limits[][3])
{

  int i, j;
  long int n;

  // initialize 'Histogram' class parameters
  Dim = 1;
  DimPhys = 1;
  Bins = 1;
  BinsPhys = 1;
  BinsNonPhys = 1;
  BinsMasked = 0;
  BinsMaskedPhys = 0;
  BinsVisited = 0;
  BinsVisitedPhys = 0;
  Hits = 0;
  HitsPhys = 0;
  YoYo = 1;

  Dim = d;
  DimPhys = dphys;
  if (DimPhys < 1) ErrorMsg(3);
  if (Dim < DimPhys) ErrorMsg(3);

  HMin  = new ObservableType[Dim];
  HMax  = new ObservableType[Dim];
  HSize = new ObservableType[Dim];
  HBins = new long int[Dim];
  HInt  = new long int[Dim];

  for (i = 0; i < Dim; i++) {
    HMin[i]  = limits[i][0];
    HMax[i]  = limits[i][1];
    HSize[i] = limits[i][2];
    if (HMin[i] >= HMax[i]) ErrorMsg(3);
    if (HSize[i] <= (ObservableType)(0)) ErrorMsg(3);
    HBins[i] = (long int)((HMax[i] - HMin[i]) / HSize[i]);
    // required to keep within array boundaries:
    HMax[i] = HMin[i] + (ObservableType)(HBins[i]) * HSize[i];
    Bins *= HBins[i];
    if (i < DimPhys) BinsPhys *= HBins[i];
  }

  BinsNonPhys = Bins / BinsPhys;

  for (i = Dim - 1; i >= 0; i--) {
    HInt[i] = 1;
    for (j = i + 1; j < Dim; j++)
      HInt[i] *= HBins[j];
  }

  dos  = new double[Bins];
  mask = new int[Bins];
  hist = new unsigned long int[Bins];

  for (n = 0; n < Bins; n++) {
    dos[n] = 0.0;
    mask[n] = 0;
    hist[n] = 0;
  }

  PrintSpecs();

}


Histogram::~Histogram()
{

  delete[] dos;
  delete[] mask;
  delete[] hist;

  delete[] HMin;
  delete[] HMax;
  delete[] HSize;
  delete[] HBins;
  delete[] HInt;

}


void Histogram::PrintSpecs()
{

  int i;

  printf("\n### Histogram Specifications ###########################\n\n");
  printf(" Dim  %d\n", Dim);
  printf(" DimPhys  %d\n\n", DimPhys);
  printf(" Bins  %ld\n", Bins);
  printf(" BinsPhys  %ld\n", BinsPhys);
  printf(" BinsNonPhys  %ld\n\n", BinsNonPhys);
  printf(" Sequence: Dim, HMin, HMax, HSize, HBins, HInt\n");
  for (i = 0; i < Dim; i++) {
    printf("  %2d %8ld %8ld %8ld %5ld %8ld\n",
	   i, HMin[i], HMax[i], HSize[i], HBins[i], HInt[i]);
  }
  printf("\n########################################################\n");

}


void Histogram::ErrorMsg(int message, const char* arg)
{

  fprintf(stderr, "\n");
  fprintf(stderr, "Error in 'Histogram': ");

  switch (message) {

  case  1 : {
    fprintf(stderr, "Input file '%s' not found.\n", arg);
    break;
  }

  case  2 : {
    fprintf(stderr, "Parameter '%s' has invalid format.\n", arg);
    break;
  }

  case  3 : {
    fprintf(stderr, "Parameter '%s' is out of range.\n", arg);
    break;
  }

  case  4 : {
    fprintf(stderr, "Unreadable parameter.\n");
    break;
  }

  case  5 : {
    fprintf(stderr, "Initialization failed.\n");
    break;
  }

  case  6 : {
    fprintf(stderr, "Class state file '%s' unreadable.\n", arg);
    break;
  }

  default : fprintf(stderr, "%s.\n", arg);

  }

  fprintf(stderr, "Program aborted.\n\n");
  exit(1);

}


void Histogram::MaskAll()
{
  long int n;
  for (n = 0; n < Bins; n++)
    mask[n] = 1;
  BinsMasked = Bins;
  BinsMaskedPhys = BinsPhys;
}


void Histogram::ResetHist()
{
  long int n;
  for (n = 0; n < Bins; n++)
    hist[n] = 0;
  BinsVisited = 0;
  BinsVisitedPhys = 0;
  Hits = 0;
  HitsPhys = 0;
}


double Histogram::GetDOSMin(int OnlyPhys)
{

  long int n, incr;
  int flag = 1;
  double min = 0.0;

  if (OnlyPhys) incr = BinsNonPhys;
  else incr = 1;

  for (n = 0; n < Bins; n += incr)
    if (mask[n]) {
      if (flag) {
	flag = 0;
	min = dos[n];
      }
      else if (dos[n] < min) min = dos[n];
    }

  return min;

}


double Histogram::GetDOSMax(int OnlyPhys)
{

  long int n, incr;
  int flag = 1;
  double max = 0.0;

  if (OnlyPhys) incr = BinsNonPhys;
  else incr = 1;

  for (n = 0; n < Bins; n += incr)
    if (mask[n]) {
      if (flag) {
	flag = 0;
	max = dos[n];
      }
      else if (dos[n] > max) max = dos[n];
    }

  return max;

}


unsigned long int Histogram::GetHistMin(int OnlyPhys, int OnlyCur)
{

  long int n, incr;
  int flag = 1;
  unsigned long int min = 0;

  if (OnlyPhys) incr = BinsNonPhys;
  else incr = 1;

  if (OnlyCur) {
    for (n = 0; n < Bins; n += incr)
      if (hist[n]) {
	if (flag) {
	  flag = 0;
	  min = hist[n];
	}
	else if (hist[n] < min) min = hist[n];
      }
  }

  else {
    for (n = 0; n < Bins; n += incr)
      if (mask[n]) {
	if (flag) {
	  flag = 0;
	  min = hist[n];
	}
	else if (hist[n] < min) min = hist[n];
      }
  }

  return min;

}


unsigned long int Histogram::GetHistMax(int OnlyPhys, int OnlyCur)
{

  long int n, incr;
  int flag = 1;
  unsigned long int max = 0;

  if (OnlyPhys) incr = BinsNonPhys;
  else incr = 1;

  if (OnlyCur) {
    for (n = 0; n < Bins; n += incr)
      if (hist[n]) {
	if (flag) {
	  flag = 0;
	  max = hist[n];
	}
	else if (hist[n] > max) max = hist[n];
      }
  }

  else {
    for (n = 0; n < Bins; n += incr)
      if (mask[n]) {
	if (flag) {
	  flag = 0;
	  max = hist[n];
	}
	else if (hist[n] > max) max = hist[n];
      }
  }

  return max;

}


int Histogram::CheckHistogram(int CheckType, double p, unsigned long int bmin)
{

  long int n;
  double threshold;

  switch (CheckType) {

  case 0 : {

    threshold = p * (double)(Hits) / (double)(BinsMasked);

    for (n = 0; n < Bins; n++)
      if ((mask[n]) && ((double)(hist[n]) < threshold)) return 0;

    return 1;

  }

  case 1 : {

    for (n = 0; n < Bins; n++)
      if ((mask[n]) && (hist[n] < bmin)) return 0;

    return 1;

  }

  }

  return 0;

}


void Histogram::PrintNormDOS(const char* filename, int OnlyPhys)
{

  const double dosmax = GetDOSMax(OnlyPhys);

  long int m, n, incr;
  int d, i;
  double c = 0.0;

  FILE* f;
  if (filename != NULL) f = fopen(filename, "w");
  else f = stdout;

  if (OnlyPhys) {
    incr = BinsNonPhys;
    d = DimPhys;
  }
  else {
    incr = 1;
    d = Dim;
  }

  for (n = 0; n < Bins; n += incr)
    if (mask[n]) c += exp(dos[n] - dosmax);

  for (n = 0; n < Bins; n += incr)
    if (mask[n]) {
      m = n;
      for (i = 0; i < d; i++) {
	fprintf(f, " %4ld", HMin[i] + (m / HInt[i]) * HSize[i]);
	m %= HInt[i];
      }
      fprintf(f, "  %18.10e\n", exp(dos[n] - dosmax) / c);
    }

  if (filename != NULL) fclose(f);

}


long int Histogram::GetIndex(ObservableType v[], int MaskCheck)
{

  int i;
  long int index = 0;

  for (i = 0; i < Dim; i++) {
    if ((v[i] < HMin[i]) || (v[i] >= HMax[i])) return -1;
    index += HInt[i] * (long int)((v[i] - HMin[i]) / HSize[i]);
  }

  if ((MaskCheck) && (!mask[index])) return -1;
  else return index;

}


void Histogram::Update(long int index,
		       unsigned long int& counter,
		       int UpdateType, double f)
{

  int i, PhysFlag;

  if ((index % BinsNonPhys) == 0) PhysFlag = 1;
  else PhysFlag = 0;

  switch (UpdateType) {

  case  1 : {
    if (mask[index]) dos[index] += f;
    else {
      ResetHist();
      counter = 0;
      dos[index] = GetDOSMin();
    }
    break;
  }

  default : {
    dos[index] += f;
  }

  }

  if (!hist[index]) {
    BinsVisited++;
    if (PhysFlag) BinsVisitedPhys++;
  }

  hist[index]++;
  Hits++;
  if (PhysFlag) HitsPhys++;

  if (!mask[index]) {
    mask[index] = 1;
    BinsMasked++;
    if (PhysFlag) BinsMaskedPhys++;
    printf("New bin masked:");
    for (i = 0; i < Dim; i++) {
      printf(" %4ld", HMin[i] + (index / HInt[i]) * HSize[i]);
      index %= HInt[i];
    }
    printf("\n");
    printf("Actual DOS minimum  = %15.8e\n", GetDOSMin());
    printf("Actual mod. factor  = %15.8e\n", f);
    printf("Ratio (Min DOS / f) = %15.8e\n", GetDOSMin() / f);
  }

}


int Histogram::CheckItinerancy(long int index,
			       const unsigned long int MCMoves,
			       unsigned long int& MCMovesMem,
			       int dim, const char* filename)
{

  int i;
  long int n;
  FILE* f;

  if ((index % BinsNonPhys) == 0) {  // only "physical" round-trips are considered

    for (i = 0; i < dim; i++) index %= HInt[i];
    index /= HInt[dim];

    if (YoYo == 1) {  // up-walk

      for (n = Bins - BinsNonPhys; n >= 0; n -= BinsNonPhys)
	if (mask[n]) {
	  for (i = 0; i < dim; i++) n %= HInt[i];
	  n /= HInt[dim];
	  break;
	}

      if (index == n) {
	if (filename != NULL) f = fopen(filename, "a");
	else f = stdout;
	fprintf(f, "Up itinerancy = %lu\n", MCMoves - MCMovesMem);
	fflush(f);
	if (filename != NULL) fclose(f);
	MCMovesMem = MCMoves;
	YoYo = 0;
	return 1;
      }

    }

    else {  // down-walk

      for (n = 0; n < Bins; n += BinsNonPhys)
	if (mask[n]) {
	  for (i = 0; i < dim; i++) n %= HInt[i];
	  n /= HInt[dim];
	  break;
	}

      if (index == n) {
	if (filename != NULL) f = fopen(filename, "a");
	else f = stdout;
	fprintf(f, "Down itinerancy = %lu\n", MCMoves - MCMovesMem);
	fflush(f);
	if (filename != NULL) fclose(f);
	MCMovesMem = MCMoves;
	YoYo = 1;
      }

    }

  }

  return 0;

}


void Histogram::SaveState(const unsigned long int c, const char* s, const bool ShiftDOS)
{

  int i;
  long int n;
  char filename[50];
  double dosmin;

  if (ShiftDOS) dosmin = GetDOSMin();
  else dosmin = 0.0;

  if (c == 0)
    if (s == NULL) sprintf(filename, "hdata_current.dat");
    else sprintf(filename, "%s", s);
  else
    if (s == NULL) sprintf(filename, "hdata%08lu.dat", c);
    else sprintf(filename, "%s%08lu.dat", s, c);

  FILE* f = fopen(filename, "w");

  fprintf(f, "# State file for 'Histogram' class instance\n");
  fprintf(f, "# Do not edit this file!\n");

  fprintf(f, "Dim  %d\n", Dim);
  fprintf(f, "DimPhys  %d\n", DimPhys);

  fprintf(f, "Bins  %ld\n", Bins);
  fprintf(f, "BinsPhys  %ld\n", BinsPhys);
  fprintf(f, "BinsMasked  %ld\n", BinsMasked);
  fprintf(f, "BinsMaskedPhys  %ld\n", BinsMaskedPhys);
  fprintf(f, "BinsVisited  %ld\n", BinsVisited);
  fprintf(f, "BinsVisitedPhys  %ld\n", BinsVisitedPhys);

  fprintf(f, "Hits  %lu\n", Hits);
  fprintf(f, "HitsPhys  %lu\n", HitsPhys);

  fprintf(f, "YoYo  %d\n", YoYo);

  fprintf(f, "\n");
  fprintf(f, "# Sequence: Dim, HMin, HSize, HBins, HInt\n");
  for (i = 0; i < Dim; i++)
    fprintf(f, " %2d %8ld %8ld %5ld %8ld\n", i, HMin[i], HSize[i], HBins[i], HInt[i]);
  //  fprintf(f, " %2d %12.5e %12.5e %5ld %8ld\n", i, HMin[i], HSize[i], HBins[i], HInt[i]);

  fprintf(f, "\n");
  fprintf(f, "# Sequence: Mask, Density of States (DOS), Histogram\n");
  for (n = 0; n < Bins; n++)
    if (mask[n])
      fprintf(f, " %d  %.16e  %lu\n", mask[n], dos[n] - dosmin, hist[n]);
    else
      fprintf(f, " %d  %.16e  %lu\n", mask[n], dos[n], hist[n]);

  fclose(f);

}
