// Pele Helper functions 


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
#include <time.h>
#include <gsl/gsl_permutation.h>

#include "pele/Globals.h"
#include "pele/HPModel.h"

namespace pele {
    HPModel::HPModel(const char* filename)
    {

        const int MaxSeqHP = 200000;

        char line[200];
        char pname[51];
        FILE* f;

        char s[101];
        char* sHP;
        int flag = 0;

        int i, n;
        char c;

        // initialize 'HPModel' class parameters
        PolymerDim = 2;
        OccupancyFieldType = 0;
        MoveFraction[0] = 1.0;
        MoveFraction[1] = 1.0;

        NumberOfMonomers = 0;
        sHP = new char[MaxSeqHP + 1];
        sHP[0] = '\0';

        // read 'HPModel' class parameters from input file
        f = fopen(filename, "r");
        if (f == NULL) ErrorMsg(1, filename);

        while (fgets(line, sizeof(line), f) != NULL) {

            if ((line[0] != '#') && (line[0] != '\n')) {

                if (sscanf(line, "%50s", pname) == 1) {

                    if (!strcmp(pname, "HPModelDimension")) {
                        if (sscanf(line, "%*50s %d", &PolymerDim) != 1) ErrorMsg(2, pname);
                        if (PolymerDim < 2) ErrorMsg(3, pname);
                        continue;
                    }

                    if (!strcmp(pname, "OccupancyFieldType")) {
                        if (sscanf(line, "%*50s %d", &OccupancyFieldType) != 1) ErrorMsg(2, pname);
                        if ((OccupancyFieldType < 0) || (OccupancyFieldType > 1)) ErrorMsg(3, pname);
                        continue;
                    }

                    if (!strcmp(pname, "MoveFractions")) {
                        if (sscanf(line, "%*50s %lf %lf",
                                   &MoveFraction[0], &MoveFraction[1]) != 2) ErrorMsg(2, pname);
                        continue;
                    }

                    if (!strcmp(pname, "HPSequence")) {
                        // check that 'NumberOfMonomers' has not been declared in input file
                        if (flag == 2) ErrorMsg(0, "'HPSequence' and 'NumberOfMonomers' declared simultaneously");
                        if (sscanf(line, "%*50s %100s", s) != 1) ErrorMsg(2, pname);
                        if ((int)(strlen(sHP) + strlen(s)) <= MaxSeqHP) {
                            strcpy(sHP + strlen(sHP), s);
                        NumberOfMonomers = strlen(sHP);
                    }
                    else ErrorMsg(5);
                    flag = 1;
                    continue;
                }

                if (!strcmp(pname, "NumberOfMonomers")) {
                    // check that 'HPSequence' has not been declared in input file
                    if (flag == 1) ErrorMsg(0, "'HPSequence' and 'NumberOfMonomers' declared simultaneously");
                    if (sscanf(line, "%*50s %d", &NumberOfMonomers) != 1) ErrorMsg(2, pname);
                    if ((NumberOfMonomers < 1) || (NumberOfMonomers > MaxSeqHP)) ErrorMsg(3, pname);
                    for (n = 0; n < NumberOfMonomers; n++) sHP[n] = 'H';
                    flag = 2;
                    continue;
                }

            }

            else ErrorMsg(4);

        }

    }

    fclose(f);

    if (NumberOfMonomers == 0) ErrorMsg(0, "No HP sequence specified");
    if (NumberOfMonomers < 3) ErrorMsg(0, "HP sequence too short");

    // now, construct class instance...

    // pull-2 moves may shift the polymer by max. 2 lattice units
    // --> LSize must be >= polymer chain length + 2 in order to
    //     allow for a "free moving" of the polymer in space
    LSize = NumberOfMonomers + 2;
    LOffset = new unsigned long int[PolymerDim];
    for (n = 0; n < PolymerDim; n++)
        LOffset[n] = NtoM(LSize, PolymerDim - 1 - n);

    seq = new MonomerType[NumberOfMonomers];
    coords = new Vector*[NumberOfMonomers];
    tmpcoords = new Vector*[NumberOfMonomers];

    for (n = 0; n < NumberOfMonomers; n++) {
        coords[n] = new Vector(PolymerDim);
        tmpcoords[n] = new Vector(PolymerDim);
        coords[n] -> Elem(0) = n;
        switch (sHP[n]) {
        case 'P' : {seq[n] = polar; break;}
        case 'H' : {seq[n] = hydrophobic; break;}
        default  : {ErrorMsg(6, &sHP[n]);}
        }
    }

    // read coordinates file...
    // --> first, check "coords_current.xyz"
    f = fopen("coords_current.xyz", "r");
    // if not available..., check "coords_init.xyz"
    if (f == NULL) f = fopen("coords_init.xyz", "r");
    if (f != NULL) {
        for (n = 0; n < NumberOfMonomers; n++) {
            if (fscanf(f, "%c", &c) != 1) ErrorMsg(7);
            switch (c) {
            case 'P' : {seq[n] = polar; break;}
            case 'H' : {seq[n] = hydrophobic; break;}
            default  : {ErrorMsg(6, &c);}
            }
            for (i = 0; i < PolymerDim; i++)
                if (fscanf(f, "%ld", &coords[n] -> Elem(i)) != 1)
                    ErrorMsg(7);
            fscanf(f, "%*c");
        }
        fclose(f);
    }

    if (OccupancyFieldType == 0)
        OccupancyField = new BitTable(NtoM(LSize, PolymerDim));
    else
        // good choice for hash table size:
        //   --> a prime number distant from a power of 2
        //   --> e.g. 47,12203,1600033,50000017
        OccupancyField = new HashTable(1600033);

    site = new unsigned long int[NumberOfMonomers];
    tmpsite = new unsigned long int[NumberOfMonomers];

    for (n = 0; n < NumberOfMonomers; n++) {
        site[n] = GetSite(*coords[n]);
        tmpsite[n] = site[n];
        OccupancyField -> Insert(site[n], n);
    }

    NumberOfHMonomers = 0;
    for (n = 0; n < NumberOfMonomers; n++)
    if (seq[n] == hydrophobic) NumberOfHMonomers++;

  InternalHHContacts = 0;
  for (n = 0; n < NumberOfMonomers - 1; n++)
    if ((seq[n] == hydrophobic) &&
	(seq[n + 1] == hydrophobic)) InternalHHContacts++;

  CheckModelIntegrity();

  // pull-2 move:
  InitPull2Set();

  // bond-rebridging move: for relabeling/restoring
  CoordsPointer = new Vector*[NumberOfMonomers];
  CoordsIndex = new int[NumberOfMonomers];

  // pivot move:
  InitPivotMatrices();

  // observables:
  // measured during Monte Carlo sampling
  // 0  : number of non-bonded HH contacts (principal observable)

  // additional observables could be defined and sampled, e.g. for a
  // 2D Wang-Landau simulation

  InitObservables(1);

  // evaluate "initial" principal observables...
  if (NumberOfHMonomers != NumberOfMonomers) {
    // HP model Hamiltonian
    HPModelFlag = 1;
    GetObservablesHP();
  }
  else {
    // ISAW model Hamiltonian
    HPModelFlag = 0;
    GetObservablesISAW();
  }

  // calculating the number of non-bonded HH contacts (principal
  // observable) with 'GetObservablesPart' (HP/ISAW polymer fragment)
  // takes approx. 4 times more operations per H monomer than with
  // 'GetObservables' (entire HP/ISAW polymer)

  // --> if (BeadsInterval < BeadsIntervalThreshold) use 'GetObservablesPart'
  //     otherwise use 'GetObservables'
  BeadsIntervalThreshold = NumberOfMonomers / 4;

  // required for Monte Carlo moves statistics
  for (i = 0; i < 3; i++) {
    MoveTrials[i] = 0;
    MoveRejects[i] = 0;
    MoveDiscards[i] = 0;
    BeadsTrials[i] = 0;
    BeadsRejects[i] = 0;
  }
  // I haaate globals
  rng = gsl_rng_alloc(gsl_rng_mt19937);
  gsl_rng_set(rng, 5);
  WriteState(0);

  delete[] sHP;

}


HPModel::~HPModel()
{

  int n;
  long int i;

  CheckModelIntegrity();

  delete[] seq;

  for (n = 0; n < NumberOfMonomers; n++) {
    delete coords[n];
    delete tmpcoords[n];
  }
  delete[] coords;
  delete[] tmpcoords;

  delete OccupancyField;

  delete[] site;
  delete[] tmpsite;

  delete[] Pull2Set;

  delete[] CoordsPointer;
  delete[] CoordsIndex;

  DeleteObservables();

  for (i = 0; i < NPiOps; i++)
    delete pivot[i];
  delete[] pivot;

  delete[] LOffset;

  WriteState(5);

  printf("Ciao HPModel...\n");

}


void HPModel::WriteState(int format, const char* filename)
{

  const char symbol[2] = {'H', 'P'};

  int i, n;

  FILE* f;
  if (filename != NULL) f = fopen(filename, "a");
  else f = stdout;

  switch (format) {

    // mol2 format output
  case  1 : {

    double* ctr = new double[PolymerDim];
    for (i = 0; i < PolymerDim; i++) ctr[i] = 0.0;
    for (n = 0; n < NumberOfMonomers; n++)
      for (i = 0; i < PolymerDim; i++)
	ctr[i] += (double)(coords[n] -> Elem(i));
    for (i = 0; i < PolymerDim; i++) ctr[i] /= (double)(NumberOfMonomers);

    fprintf(f, "\n");
    fprintf(f, "@<TRIPOS>MOLECULE\n");
    fprintf(f, "HPMODEL\n");
    fprintf(f, " %d %d\n", NumberOfMonomers, NumberOfMonomers - 1);
    fprintf(f, "PROTEIN\n");
    fprintf(f, "NO_CHARGES\n");
    fprintf(f, "@<TRIPOS>ATOM\n");
    for (n = 0; n < NumberOfMonomers; n++) {
      fprintf(f, "%5d %c", n + 1, symbol[seq[n]]);
      if (PolymerDim == 2) {
	fprintf(f, " %9.3f %9.3f %9.3f",
		(double)(coords[n] -> Elem(0)) - ctr[0],
		(double)(coords[n] -> Elem(1)) - ctr[1],
		0.0);
      }
      else {
	fprintf(f, " %9.3f %9.3f %9.3f",
		(double)(coords[n] -> Elem(0)) - ctr[0],
		(double)(coords[n] -> Elem(1)) - ctr[1],
		(double)(coords[n] -> Elem(2)) - ctr[2]);
      }
      fprintf(f, " %c\n", symbol[seq[n]]);
    }
    fprintf(f, "@<TRIPOS>BOND\n");
    for (n = 1; n < NumberOfMonomers; n++)
      fprintf(f, "%5d %5d %5d 1\n", n, n, n + 1);

    delete[] ctr;

    break;

  }

    // pdb format output
  case  2 : {

    fprintf(f, "MODEL\n");
    for (n = 0; n < NumberOfMonomers; n++) {
      if (seq[n] == polar)
	fprintf(f, "ATOM   %4d  O           0", n + 1);
      else
	fprintf(f, "ATOM   %4d  P           0", n + 1);
      if (PolymerDim == 2) {
	fprintf(f, " %9.2f %9.2f %9.2f\n",
		10.0 * (double)(coords[n] -> Get(0) - coords[0] -> Get(0)),
		10.0 * (double)(coords[n] -> Get(1) - coords[0] -> Get(1)),
		0.0);
      }
      else {
	fprintf(f, " %9.2f %9.2f %9.2f\n",
		10.0 * (double)(coords[n] -> Get(0) - coords[0] -> Get(0)),
		10.0 * (double)(coords[n] -> Get(1) - coords[0] -> Get(1)),
		10.0 * (double)(coords[n] -> Get(2) - coords[0] -> Get(2)));
      }
    }
    fprintf(f, "CONECT %4d %4d\n", 1, 2);
    for (n = 2; n < NumberOfMonomers; n++)
      fprintf(f, "CONECT %4d %4d %4d\n", n, n - 1, n + 1);
    fprintf(f, "CONECT %4d %4d\n", NumberOfMonomers, NumberOfMonomers - 1);
    fprintf(f, "ENDMDL\n\n");

    break;

  }

    // current model state and observables output
  case  3 : {

    fprintf(f, "\n");
    fprintf(f, "HP/ISAW Model : %d\n", NumberOfMonomers);
    PrintObservables(f);
    for (n = 0; n < NumberOfMonomers; n++) {
      fprintf(f, "%c", symbol[seq[n]]);
      for (i = 0; i < PolymerDim; i++)
	fprintf(f, " %ld", coords[n] -> Get(i) - coords[0] -> Get(i));
      fprintf(f, "\n");
    }
    break;

  }

    // write current model state
  case  4 : {

    if (filename != NULL) {
      fclose(f);
      f = fopen(filename, "w");
    }
    for (n = 0; n < NumberOfMonomers; n++) {
      fprintf(f, "%c", symbol[seq[n]]);
      for (i = 0; i < PolymerDim; i++)
	fprintf(f, " %ld", coords[n] -> Get(i));
      fprintf(f, "\n");
    }
    break;

  }

    // print move statistics
  case  5 : {

    fprintf(f, "\nMove statistics:\n\n");
    fprintf(f, " Move type       | # Attempts   | Valid moves | Acceptance ratio | Polymer fragment | Polymer fragment\n");
    fprintf(f, "                 |              | (ratio)     | (valid moves)    | (valid moves)    | (rejected valid moves)\n");
    fprintf(f, "-----------------|--------------|-------------|------------------|------------------|-----------------------\n");
    for (i = 0; i < 3; i++) {
      switch (i) {
      case 0 : {fprintf(f, " pull-2          |"); break;}
      case 1 : {fprintf(f, " bond-rebridging |"); break;}
      case 2 : {fprintf(f, " pivot           |"); break;}
      }
      fprintf(f, " %12ld |  %10.4e |       %10.4e |       %10.4e |             %10.4e\n",
	      MoveTrials[i] + MoveDiscards[i],
	      double(MoveTrials[i]) / double(MoveTrials[i] + MoveDiscards[i]),
	      double(MoveTrials[i] - MoveRejects[i]) / double(MoveTrials[i]),
	      double(BeadsTrials[i]) / double(MoveTrials[i]),
	      double(BeadsRejects[i]) / double(MoveRejects[i]));
    }
    fprintf(f, "\n");
    break;

  }

    // print model specifications
  case  0 : {

    printf("\n### Physical Model #####################################\n\n");
    printf("HP Model in d dimensions\n\n");
    printf("Specifications:\n\n");
    printf(" Dimensions = %d\n", PolymerDim);
    printf(" Number of monomers = %d (%d)\n", NumberOfMonomers, NumberOfHMonomers);
    printf(" Internal HH contacts = %d\n", InternalHHContacts);
    printf(" Observables type = ");
    if (HPModelFlag) printf("HP model\n");
    else printf("ISAW model\n");
    printf(" Polymer fragment threshold = %d\n", BeadsIntervalThreshold);
    printf(" Lattice size = %ld (%lu)\n", LSize, NtoM(LSize, PolymerDim));
    printf(" Move fractions = %6.4f / %6.4f / %6.4f\n",
	   MoveFraction[0],
	   MoveFraction[1] - MoveFraction[0],
	   1.0 - MoveFraction[1]);
    printf("\nInitial state:\n\n");
    for (n = 0; n < NumberOfMonomers; n++) {
      printf(" %c ", symbol[seq[n]]);
      coords[n] -> Print();
    }
    printf("\n########################################################\n");
    break;

  }

  }

  if (filename != NULL) fclose(f);

}


void HPModel::CheckModelIntegrity()
{

  int m, n;

  for (n = 0; n < NumberOfMonomers - 1; n++)
    for (m = n + 1; m < NumberOfMonomers; m++)
      if (coords[n] -> Overlap(*coords[m])) ErrorMsg(8);

  m = 0;
  for (n = 0; n < NumberOfMonomers - 1; n++) {
    if (coords[n] -> Distance2(*coords[n + 1]) != 1) ErrorMsg(8);
    if ((seq[n] == hydrophobic) &&
	(seq[n + 1] == hydrophobic)) m++;
  }

  if (InternalHHContacts != m) ErrorMsg(8);

  m = 0;
  for (n = 0; n < NumberOfMonomers; n++) {
    if (site[n] != GetSite(*coords[n])) ErrorMsg(8);
    if (OccupancyField -> CheckKey(site[n]) != n) ErrorMsg(8);
    if ((seq[n] != hydrophobic) && (seq[n] != polar)) ErrorMsg(8);
    if (seq[n] == hydrophobic) m++;
  }

  if (NumberOfHMonomers != m) ErrorMsg(8);

  if (OccupancyField -> GetFillSize() != (unsigned long int)(NumberOfMonomers)) ErrorMsg(8);

  OccupancyField -> GetInfo();
  printf("\nModel integrity ok!\n");

}


void HPModel::ErrorMsg(int message, const char* arg)
{

  fprintf(stderr, "\n");
  fprintf(stderr, "Error in 'HPModel': ");

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
    fprintf(stderr, "HP sequence too long.\n");
    break;
  }

  case  6 : {
    fprintf(stderr, "Invalid monomer type '%c'.\n", *arg);
    break;
  }

  case  7 : {
    fprintf(stderr, "Coordinates file unreadable.\n");
    break;
  }

  case  8 : {
    fprintf(stderr, "Model integrity broken.\n");
    break;
  }

  default : fprintf(stderr, "%s.\n", arg);

  }

  fprintf(stderr, "Program aborted.\n\n");
  exit(1);

}


void HPModel::GetObservablesHP()
{

  int i, n, nn;
  long int HHContacts;

  HHContacts = 0;

  for (n = 0; n < NumberOfMonomers; n++) {

    if (seq[n] == hydrophobic) {

      for (i = 0; i < PolymerDim; i++) {

	if ((coords[n] -> Elem(i) % LSize) != 0)
	  nn = OccupancyField -> Query(site[n] - LOffset[i]);
	else
	  nn = OccupancyField -> Query(site[n] + (LSize - 1) * LOffset[i]);

	if ((nn != -1) && (seq[nn] == hydrophobic)) HHContacts++;

      }

    }

  }

  observable[0] = HHContacts - InternalHHContacts;

}



void HPModel::GetObservablesPartHP(int nmin, int nmax, long int sign)
{

  int i, n, nn;
  long int HHint, HHext;

  // in order to get the correct number of HH contacts for the
  // evaluated chain fragment, one needs to distinguish between
  // "internal" (i.e. within the chain fragment) and "external"
  // (i.e. with monomers outside of the chain fragment) HH contacts

  // --> "internal" HH contacts are counted twice (divide by 2)

  // note that "intra-chain" HH contacts cancel out when determining
  // the relative change of HH contacts (i.e. difference of HH
  // contacts between the "new" and "old" chain fragments)

  HHint = 0;  // "internal" HH contacts
  HHext = 0;  // "external" HH contacts

  if (nmin > nmax) {
    n = nmax;
    nmax = nmin;
    nmin = n;
  }
  else n = nmin;

  for (; n <= nmax; n++) {

    if (seq[n] == hydrophobic) {

      for (i = 0; i < PolymerDim; i++) {

	// check positive direction of coordinate...
	if (((coords[n] -> Elem(i) + 1) % LSize) != 0)
	  nn = OccupancyField -> Query(site[n] + LOffset[i]);
	else
	  nn = OccupancyField -> Query(site[n] - (LSize - 1) * LOffset[i]);

	if ((nn != -1) && (seq[nn] == hydrophobic)) {
	  if ((nn < nmin) || (nn > nmax)) HHext++;
	  else HHint++;
	}

	// check negative direction of coordinate...
	if ((coords[n] -> Elem(i) % LSize) != 0)
	  nn = OccupancyField -> Query(site[n] - LOffset[i]);
	else
	  nn = OccupancyField -> Query(site[n] + (LSize - 1) * LOffset[i]);

	if ((nn != -1) && (seq[nn] == hydrophobic)) {
	  if ((nn < nmin) || (nn > nmax)) HHext++;
	  else HHint++;
	}

      }

    }

  }

  observable[0] += sign * (HHint / 2 + HHext);

}


void HPModel::GetObservablesISAW()
{

  int i, n, nn;
  long int HHContacts;

  HHContacts = 0;

  for (n = 0; n < NumberOfMonomers; n++) {

    for (i = 0; i < PolymerDim; i++) {

      if ((coords[n] -> Elem(i) % LSize) != 0)
	nn = OccupancyField -> Query(site[n] - LOffset[i]);
      else
	nn = OccupancyField -> Query(site[n] + (LSize - 1) * LOffset[i]);

      if (nn != -1) HHContacts++;

    }

  }

  observable[0] = HHContacts - InternalHHContacts;

}


void HPModel::GetObservablesPartISAW(int nmin, int nmax, long int sign)
{

  int i, n, nn;
  long int HHint, HHext;

  // see comments in 'GetObservablesPartHP' (above)

  HHint = 0;  // "internal" HH contacts
  HHext = 0;  // "external" HH contacts

  if (nmin > nmax) {
    n = nmax;
    nmax = nmin;
    nmin = n;
  }
  else n = nmin;

  for (; n <= nmax; n++) {

    for (i = 0; i < PolymerDim; i++) {

      // check positive direction of coordinate...
      if (((coords[n] -> Elem(i) + 1) % LSize) != 0)
	nn = OccupancyField -> Query(site[n] + LOffset[i]);
      else
	nn = OccupancyField -> Query(site[n] - (LSize - 1) * LOffset[i]);

      if (nn != -1) {
	if ((nn < nmin) || (nn > nmax)) HHext++;
	else HHint++;
      }

      // check negative direction of coordinate...
      if ((coords[n] -> Elem(i) % LSize) != 0)
	nn = OccupancyField -> Query(site[n] - LOffset[i]);
      else
	nn = OccupancyField -> Query(site[n] + (LSize - 1) * LOffset[i]);

      if (nn != -1) {
	if ((nn < nmin) || (nn > nmax)) HHext++;
	else HHint++;
      }

    }

  }

  observable[0] += sign * (HHint / 2 + HHext);

}


void HPModel::DoMCMove()
{

    double r = gsl_rng_uniform(rng);

    // backup "old" principal observable...
    tmp_observable[0] = observable[0];

    MoveProposal = 0;

    if (r < MoveFraction[0]) {       // pull-2 move
        MCMoveType = 0;
        Pull2Move();
    }

    else if (r < MoveFraction[1]) {  // bond-rebridging move
        MCMoveType = 1;
        BondRebridgingMove();
    }

    else {                           // pivot move
        MCMoveType = 2;
        PivotMove();
    }

    if (MoveProposal) {
        MoveTrials[MCMoveType]++;
        BeadsTrials[MCMoveType] += BeadsInterval;
    }
    else MoveDiscards[MCMoveType]++;

  //  printf("Do MC move %d ...\n", MCMoveType);
  //  CheckModelIntegrity();

}


void HPModel::UnDoMCMove()
{

  int n;
  Vector* t;
  Vector** tt;
  unsigned long int* s;

  //  printf("Undo MC move %d ...\n", MCMoveType);

  MoveRejects[MCMoveType]++;
  BeadsRejects[MCMoveType] += BeadsInterval;

  // restore "old" principal observable...
  observable[0] = tmp_observable[0];

  // pull-2 move
  if (MCMoveType == 0) {

    n = BeadsEnd;

    while (1) {

      OccupancyField -> Delete(site[n]);
      site[n] = tmpsite[n];
      OccupancyField -> Insert(site[n], n);

      t = coords[n];
      coords[n] = tmpcoords[n];
      tmpcoords[n] = t;

      if (n == BeadsStart) break;

      n -= BeadsIncr;

    }

  }

  // bond-rebridging move
  else if (MCMoveType == 1) {

    for (n = 0; n < NumberOfMonomers; n++)
      if (CoordsIndex[n] != n)
	OccupancyField -> Replace(site[n], CoordsIndex[n]);

    tt = coords;
    coords = CoordsPointer;
    CoordsPointer = tt;

    s = site;
    site = tmpsite;
    tmpsite = s;

  }

  // pivot move
  else {

    n = BeadsStart;
    while (1) {
      OccupancyField -> Delete(site[n]);
      t = coords[n];
      coords[n] = tmpcoords[n];
      tmpcoords[n] = t;
      if (n == BeadsEnd) break;
      n += BeadsIncr;
    }

    n = BeadsStart;
    while (1) {
      site[n] = tmpsite[n];
      OccupancyField -> Insert(site[n], n);
      if (n == BeadsEnd) break;
      n += BeadsIncr;
    }

  }

  //  CheckModelIntegrity();

}


// in 2D the 8 pivot matrices generated by InitPivotMatrices()
// correspond to the following symmetry operations
// on the square lattice (relative to a given pivot point):

// pivot[0] = identity
// pivot[1] = reflexion on principal diagonal
// pivot[2] = reflexion on y-axis
// pivot[3] = +90 degree rotation (counter-clockwise)
// pivot[4] = reflexion on x-axis
// pivot[5] = -90 degree rotation (clockwise)
// pivot[6] = 180 degree rotation
// pivot[7] = reflexion on secondary diagonal

// in d dimensions, the total number of
// pivot operations is 2^d * d!

void HPModel::InitPivotMatrices()
{

  const long int c = NtoM(2, PolymerDim);

  long int k, n, v;
  int i, j;
  Matrix* t;

  long int* s = new long int[PolymerDim];
  gsl_permutation* p = gsl_permutation_alloc(PolymerDim);

  printf("\n\nInitializing Pivot matrices...");

  NPiOps = c * Nfac(PolymerDim);
  pivot = new Matrix*[NPiOps];
  for (n = 0; n < NPiOps; n++)
    pivot[n] = new Matrix(PolymerDim, PolymerDim);

  n = 0;
  for (k = 0; k < c; k++) {

    v = k;
    for (i = PolymerDim - 1; i >= 0; i--) {
      s[i] = v / NtoM(2, i);
      s[i] = 1 - 2 * s[i];
      v %= NtoM(2, i);
    }

    gsl_permutation_init(p);

    do {

      for (i = 0; i < PolymerDim; i++)
	for (j = 0; j < PolymerDim; j++)
	  pivot[n] ->
	    Set(i, j, s[i] * DeltaFunc(i, gsl_permutation_get(p, j)));

      //      pivot[n] -> Print();
      n++;

    } while (gsl_permutation_next(p) == GSL_SUCCESS);

  }

  if (n != NPiOps) ErrorMsg(8);

  printf("done\n");
  printf("--> number of Pivot matrices = %ld\n\n", NPiOps);

  gsl_permutation_free(p);
  delete[] s;

  // identity operation (0) does not generate
  // a "new" configuration
  // --> shift to the end of array of pivot matrices
  t = pivot[0];
  pivot[0] = pivot[n - 1];
  pivot[n - 1] = t;

}


void HPModel::InitPull2Set()
{

  int n, d, i, u, j, v;
  long int m = 0;

  // max. number of pull-2 moves:
  // --> 2 * (max. number of end pull-2 moves)
  //     + (n - 2) * (max. number of internal pull-2 moves)

  // note: the square lattice in d dimensions has
  //       2 * d nearest neighbor sites and
  //       2 * (d - 1) nearest neighbor sites in the hyper-plane
  //       perpendicular to a given direction

  printf("\n\nInitializing pull-2 moves...");

  MaxPull2Moves = 0;

  // internal pull-2 moves
  MaxPull2Moves += (NumberOfMonomers-2) * 2 * 2*(PolymerDim-1);

  // end pull-2 moves
  MaxPull2Moves += 2 * (2*(PolymerDim-1) + 1);
  MaxPull2Moves += 2 * (2*(PolymerDim-1) * (2*(PolymerDim-2) + 1 + 1));

  Pull2Set = new int[MaxPull2Moves][6];

  for (n = 1; n < NumberOfMonomers - 1; n++) {
    for (d = -1; d <= 1; d += 2) {
      for (i = 1; i <= PolymerDim - 1; i++) {
	for (u = -1; u <= 1; u += 2) {

	  Pull2Set[m][0] = n;
	  Pull2Set[m][1] = d;
	  Pull2Set[m][2] = i;
	  Pull2Set[m][3] = u;
	  Pull2Set[m][4] = -1;
	  Pull2Set[m][5] = -1;

// 	  printf("set %3ld : %3d %3d %3d %3d %3d %3d\n",
// 		 m,
// 		 Pull2Set[m][0],
// 		 Pull2Set[m][1],
// 		 Pull2Set[m][2],
// 		 Pull2Set[m][3],
// 		 Pull2Set[m][4],
// 		 Pull2Set[m][5]);

	  m++;

	}
      }
    }
  }

  for (n = 0; n < 2; n++) {
    for (i = 0; i <= PolymerDim - 1; i++) {
      for (u = -1; u <= 1; u += 2) {
	for (j = 0; j <= PolymerDim - 1; j++) {
	  for (v = -1; v <= 1; v += 2) {

	    if ((i == 0) && (u == -1)) continue;
	    if ((j == 0) && (v == -1)) continue;
	    if ((i == j) && (u != v)) continue;

	    Pull2Set[m][0] = (n == 0) ? 0 : NumberOfMonomers - 1;
	    Pull2Set[m][1] = (n == 0) ? 1 : -1;
	    Pull2Set[m][2] = i;
	    Pull2Set[m][3] = u;
	    Pull2Set[m][4] = j;
	    Pull2Set[m][5] = v;

// 	    printf("set %3ld : %3d %3d %3d %3d %3d %3d\n",
// 		   m,
// 		   Pull2Set[m][0],
// 		   Pull2Set[m][1],
// 		   Pull2Set[m][2],
// 		   Pull2Set[m][3],
// 		   Pull2Set[m][4],
// 		   Pull2Set[m][5]);

	    m++;

	  }
	}
      }
    }
  }

  if (m != MaxPull2Moves) ErrorMsg(8);

  printf("done\n");
  printf("--> max. number of pull-2 moves = %ld\n\n", MaxPull2Moves);

}


unsigned long int HPModel::GetSite(Vector& v)
{

  int i;
  long int s;
  unsigned long int index = 0;

  for (i = 0; i < PolymerDim; i++) {
    s = v.Elem(i) % LSize;
    index += LOffset[i] * (unsigned long int)((s >= 0) ? s : s + LSize);
  }

  return index;

}


void HPModel::Pull2Move()
{

  int n, d, i, u, j, v;
  int dim, c, k, end, flag = 1;
  CoordsType dir;
  Vector* t;
  unsigned long int s;

  long int m = (long int)(gsl_rng_uniform(rng) * MaxPull2Moves);

  n = Pull2Set[m][0];
  d = Pull2Set[m][1];
  i = Pull2Set[m][2];
  u = Pull2Set[m][3];
  j = Pull2Set[m][4];

  // internal pull-2 move
  if (j == -1) {

    dim = coords[n] -> GetNormalDim(*coords[n - d]);
    i = (dim + i) % PolymerDim;

    // site L
    tmpcoords[n] -> Copy(*coords[n - d]);
    tmpcoords[n] -> Elem(i) += u;
    tmpsite[n] = GetSite(*tmpcoords[n]);
    if (OccupancyField -> Query(tmpsite[n]) != -1) return;

    // site C
    k = n + d;
    tmpcoords[k] -> Copy(*coords[n]);
    tmpcoords[k] -> Elem(i) += u;
    tmpsite[k] = GetSite(*tmpcoords[k]);
    c = OccupancyField -> Query(tmpsite[k]);
    if (c != -1) {
      if (c != k) return;
      else flag = 0;
    }

  }

  // end pull-2 move
  else {

    v = Pull2Set[m][5];
    k = n + d;
    dim = coords[n] -> GetNormalDim(*coords[k]);
    dir = coords[n] -> Elem(dim) - coords[k] -> Elem(dim);

    // site C
    tmpcoords[k] -> Copy(*coords[n]);
    tmpcoords[k] -> Elem((dim + i) % PolymerDim) += u * dir;
    tmpsite[k] = GetSite(*tmpcoords[k]);
    if (OccupancyField -> Query(tmpsite[k]) != -1) return;

    // site L
    tmpcoords[n] -> Copy(*tmpcoords[k]);
    tmpcoords[n] -> Elem((dim + j) % PolymerDim) += v * dir;
    tmpsite[n] = GetSite(*tmpcoords[n]);
    if (OccupancyField -> Query(tmpsite[n]) != -1) return;

  }

  BeadsIncr = d;
  BeadsStart = n;

  if (flag) {

    if (d == 1) end = NumberOfMonomers - 1;
    else end = 0;

    n += d;
    while ((n != end) && (tmpcoords[n] -> Distance2(*coords[n + d]) != 1)) {
      n += d;
      tmpsite[n] = site[n - 2*d];
      tmpcoords[n] -> Copy(*coords[n - 2*d]);
    }

  }

  BeadsEnd = n;
  BeadsInterval = BeadsIncr * (BeadsEnd - BeadsStart) + 1;

  // change of observables from "old" polymer fragment...
  if (BeadsInterval < BeadsIntervalThreshold) {
    if (HPModelFlag)
      GetObservablesPartHP(BeadsStart, BeadsEnd, -1);
    else
      GetObservablesPartISAW(BeadsStart, BeadsEnd, -1);
  }

  // apply pull-2 move...
  n = BeadsStart;

  while (1) {

    OccupancyField -> Delete(site[n]);
    s = site[n];
    site[n] = tmpsite[n];
    tmpsite[n] = s;
    OccupancyField -> Insert(site[n], n);

    t = coords[n];
    coords[n] = tmpcoords[n];
    tmpcoords[n] = t;

    if (n == BeadsEnd) break;

    n += BeadsIncr;

  }

  // change of observables from "new" polymer fragment...
  if (BeadsInterval < BeadsIntervalThreshold) {
    if (HPModelFlag)
      GetObservablesPartHP(BeadsStart, BeadsEnd, +1);
    else
      GetObservablesPartISAW(BeadsStart, BeadsEnd, +1);
  }
  else {
    if (HPModelFlag)
      GetObservablesHP();
    else
      GetObservablesISAW();
  }

  MoveProposal = 1;

}


void HPModel::BondRebridgingMove()
{

  int i, n;
  int c1(-1), c2(-1), c3(-1), c4(-1), cdir(-1);
  int j1(-1), j2(-1), j3(-1), j4(-1), jdir(-1);
  int cmin, cmax;
  int ndim, d, dim, dir;
  Vector** tt;
  unsigned long int* s;

  n = (int)(gsl_rng_uniform(rng) * (NumberOfMonomers + 1));

  if (n > 1) {  // chain internal bond-rebridging

    c1 = n - 2;
    c2 = n - 1;

    ndim = coords[c1] -> GetNormalDim(*coords[c2]);
    d = (int)(gsl_rng_uniform(rng) * (2 * PolymerDim - 2));
    dim = d / 2;
    if (dim >= ndim) dim++;
    dir = 1 - 2 * (d % 2);

    coords[c1] -> Elem(dim) += dir;
    c3 = OccupancyField -> Query(GetSite(*coords[c1]));
    coords[c1] -> Elem(dim) -= dir;
    if (c3 == -1) return;

    coords[c2] -> Elem(dim) += dir;
    c4 = OccupancyField -> Query(GetSite(*coords[c2]));
    coords[c2] -> Elem(dim) -= dir;
    if (c4 == -1) return;

    if (abs(c4 - c3) != 1) return;

    if ((c4 - c3) == -1) {  // anti-parallel strands

      cdir = 1;

      if (c1 < c3) {
	cmin = c2;
	cmax = c4;
      }
      else {
	cmin = c3;
	cmax = c1;
      }

      if ((cmax - cmin) != 1) {
	j1 = cmin + (int)(gsl_rng_uniform(rng) * (cmax - cmin + 1));
	if (j1 != cmax) j2 = j1 + 1;
	else {
	  j2 = cmin;
	  if (cmin == c2) {
	    c2 = -1;
	    c4 = -1;
	  }
	  else {
	    c3 = -1;
	    c1 = -1;
	  }
	}
      }
      else {
	j1 = cmin;
	j2 = cmax;
      }

      ndim = coords[j1] -> GetNormalDim(*coords[j2]);
      d = (int)(gsl_rng_uniform(rng) * (2 * PolymerDim - 2));
      dim = d / 2;
      if (dim >= ndim) dim++;
      dir = 1 - 2 * (d % 2);

      coords[j1] -> Elem(dim) += dir;
      j3 = OccupancyField -> Query(GetSite(*coords[j1]));
      coords[j1] -> Elem(dim) -= dir;
      if ((j3 == -1) || ((cmin <= j3) && (j3 <= cmax))) return;

      coords[j2] -> Elem(dim) += dir;
      j4 = OccupancyField -> Query(GetSite(*coords[j2]));
      coords[j2] -> Elem(dim) -= dir;
      if ((j4 == -1) || ((cmin <= j4) && (j4 <= cmax))) return;

      if (abs(j4 - j3) != 1) return;

      if ((j4 - j3) == -1) jdir = 1;  // anti-parallel strands

    }

  }

  else {  // chain terminals bond-rebridging

    if (n == 0) {
      c1 = 0;
      c3 = 1;
    }
    else {
      c1 = NumberOfMonomers - 1;
      c3 = NumberOfMonomers - 2;
    }

    ndim = coords[c1] -> GetNormalDim(*coords[c3]);
    d = (int)(gsl_rng_uniform(rng) * (2 * PolymerDim - 1));
    if (d != (2 * PolymerDim - 2)) {
      dim = d / 2;
      if (dim >= ndim) dim++;
      dir = 1 - 2 * (d % 2);
    }
    else {
      dim = ndim;
      dir = coords[c1] -> Elem(ndim) - coords[c3] -> Elem(ndim);
    }

    coords[c1] -> Elem(dim) += dir;
    c3 = OccupancyField -> Query(GetSite(*coords[c1]));
    coords[c1] -> Elem(dim) -= dir;
    if (c3 == -1) return;

  }

  if (n == 0) {
    i = c3 - 1;
    d = -1;
  }
  else {
    i = 0;
    d = 1;
  }

  //  printf("c1-4, cdir : %3d %3d %3d %3d %3d\n", c1, c2, c3, c4, cdir);
  //  printf("j1-4, jdir : %3d %3d %3d %3d %3d\n", j1, j2, j3, j4, jdir);

  // apply bond-rebridging move...
  BeadsInterval = 0;
  for (n = 0; n < NumberOfMonomers; n++) {

    CoordsIndex[n] = i;
    CoordsPointer[n] = coords[i];
    tmpsite[n] = site[i];

    //    printf("Relabel: %3d --> %3d\n", n, i);

    if (n != i) {
      OccupancyField -> Replace(site[i], n);
      BeadsInterval++;
    }

    if (i == c1) {
      i = c3;
      c3 = -1;
      d *= cdir;
    }
    else if (i == c2) {
      i = c4;
      c4 = -1;
      d *= cdir;
    }
    else if (i == c3) {
      i = c1;
      c1 = -1;
      d *= cdir;
    }
    else if (i == c4) {
      i = c2;
      c2 = -1;
      d *= cdir;
    }
    else if (i == j1) {
      i = j3;
      j3 = -1;
      d *= jdir;
    }
    else if (i == j2) {
      i = j4;
      j4 = -1;
      d *= jdir;
    }
    else if (i == j3) {
      i = j1;
      j1 = -1;
      d *= jdir;
    }
    else if (i == j4) {
      i = j2;
      j2 = -1;
      d *= jdir;
    }
    else i += d;

  }

  tt = coords;
  coords = CoordsPointer;
  CoordsPointer = tt;

  s = site;
  site = tmpsite;
  tmpsite = s;

  // get "new" observables...
  // note: bond-rebridging moves do NOT alter
  //       the number of HH contacts of ISAWs
  if (HPModelFlag) GetObservablesHP();

  MoveProposal = 1;

}


void HPModel::PivotMove()
{

  int m, n, pp;
  long int po;
  Vector* t;
  unsigned long int s;

  // identity operation is excluded
  // because it does not generate a "new" configuration
  po = (long int)(gsl_rng_uniform(rng) * (NPiOps - 1));
  pp = (int)(gsl_rng_uniform(rng) * (NumberOfMonomers - 2)) + 1;

  if (pp < (NumberOfMonomers - 1 - pp)) {
    BeadsStart = pp - 1;
    BeadsIncr = -1;
    BeadsEnd = 0;
  }
  else {
    BeadsStart = pp + 1;
    BeadsIncr = +1;
    BeadsEnd = NumberOfMonomers - 1;
  }

  n = BeadsStart;
  while (1) {
    pivot[po] -> PivotOperation(*coords[pp], *coords[n], *tmpcoords[n]);
    tmpsite[n] = GetSite(*tmpcoords[n]);
    m = OccupancyField -> Query(tmpsite[n]);
    // if (m == "occupied") -->
    //  if (BeadsIncr == -1) AND (m > "pivot-point") --> return
    //    OR
    //  if (BeadsIncr == +1) AND (m < "pivot-point") --> return
    if (m != -1) {
      if (BeadsIncr == -1) {
	if (m > pp) return;
      }
      else if (m < pp) return;
    }
    if (n == BeadsEnd) break;
    n += BeadsIncr;
  }

  BeadsInterval = BeadsIncr * (BeadsEnd - BeadsStart) + 1;

  // change of observables from "old" polymer fragment...
  if (BeadsInterval < BeadsIntervalThreshold) {
    if (HPModelFlag)
      GetObservablesPartHP(BeadsStart, BeadsEnd, -1);
    else
      GetObservablesPartISAW(BeadsStart, BeadsEnd, -1);
  }

  // apply pivot move...
  n = BeadsStart;
  while (1) {
    OccupancyField -> Delete(site[n]);
    t = coords[n];
    coords[n] = tmpcoords[n];
    tmpcoords[n] = t;
    if (n == BeadsEnd) break;
    n += BeadsIncr;
  }

  n = BeadsStart;
  while (1) {
    s = site[n];
    site[n] = tmpsite[n];
    tmpsite[n] = s;
    OccupancyField -> Insert(site[n], n);
    if (n == BeadsEnd) break;
    n += BeadsIncr;
  }

  // change of observables from "new" polymer fragment...
  if (BeadsInterval < BeadsIntervalThreshold) {
    if (HPModelFlag)
      GetObservablesPartHP(BeadsStart, BeadsEnd, +1);
    else
      GetObservablesPartISAW(BeadsStart, BeadsEnd, +1);
  }
  else {
    if (HPModelFlag)
      GetObservablesHP();
    else
      GetObservablesISAW();
  }

  MoveProposal = 1;

}



//  PELE HELPER functions

Vector** HPModel::pele_array_to_vector(Array<double> x) {
    Vector** v;
    v = new Vector*[NumberOfMonomers];
    int j, blah;
    for (blah = 0; blah < NumberOfMonomers; ++blah) {
        v[blah] = new Vector(PolymerDim);
#pragma unroll
        for (j=0; j < PolymerDim; ++j) {
            v[blah]->Elem(j) = x[PolymerDim*blah + j] ;
        }
    }

    return v;
}


Array<double> HPModel::vector_to_pele_array(Vector **v) {
    Array<double> x(NumberOfMonomers*PolymerDim);
    int j, blah;
    for (blah = 0; blah < NumberOfMonomers; ++blah) {
#pragma unroll
        for (j=0; j < PolymerDim; ++j) {
            x[PolymerDim*blah + j] = v[blah]->Elem(j) ;
        }
    }
}

// Returns the energy
double HPModel::get_energy(const Array<double> &x) {
    return observable[0];
}


} // namespace pele