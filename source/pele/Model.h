// TODO: Include modification clauses for gnu gpl3


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

#ifndef MODEL_H
#define MODEL_H

#include <stdlib.h>
#include <stdio.h>

#include "Histogram.h"


/*
  Model class:

  This is the basis class for all derived "physical model" classes. It
  declares the principal (virtual) functions and procedures which are
  required for any of the Monte Carlo algorithms. Furthermore, it
  provides (largely self-explanatory) functions for the handling of
  physical observables variables.
*/

class Model {

public:

  virtual ~Model() {}

  // returns useful information of the "physical model" (e.g., in case
  // of the "HP model", prints the HP protein configuration in various
  // formats)
  // 1. type of information
  // 2. output stream (e.g. file name)
  virtual void WriteState(int = 0, const char* = NULL) = 0;
  // perfoms a Monte Carlo trial move
  virtual void DoMCMove() = 0;
  // undoes a Monte Carlo trial move in case of rejection due to the
  // Wang-Landau / Metropolis acceptance criterion
  virtual void UnDoMCMove() = 0;

  void PrintObservables(FILE* f = stdout) {

    int i;

    fprintf(f, "Observables:");
    for (i = 0; i < NumberOfObservables; i++)
      fprintf(f, " %ld", observable[i]);
    fprintf(f, "\n");

  }

  // number of defined observables for the "physical model"; for the
  // "HPModel" class, there is currently only one observable defined
  // (number of non-bonded HH contacts)
  int NumberOfObservables;
  // pointer to the array of observables
  ObservableType* observable;

  // state variable indicating whether a particular Monte Carlo trial
  // move has been applied successfully (i.e. no self-overlap
  // violation), (0 = not successful; 1 = successful); in case of not
  // successful, the time consuming Wang-Landau / Metropolis
  // acceptance criterion check can be skipped
  int MoveProposal;

protected:

  ObservableType* tmp_observable;

  void InitObservables(int n) {

    int i;

    NumberOfObservables = n;

    observable = new ObservableType[NumberOfObservables];
    tmp_observable = new ObservableType[NumberOfObservables];

    for (i = 0; i < NumberOfObservables; i++) {
      observable[i] = (ObservableType)(0);
      tmp_observable[i] = (ObservableType)(0);
    }

  }


  void BackupObservables() {

    int i;

    for (i = 0; i < NumberOfObservables; i++)
      tmp_observable[i] = observable[i];

  }


  void ResetObservables() {

    int i;

    for (i = 0; i < NumberOfObservables; i++) {
      tmp_observable[i] = observable[i];
      observable[i] = (ObservableType)(0);
    }

  }


  void RestoreObservables() {

    ObservableType* t = observable;
    observable = tmp_observable;
    tmp_observable = t;

  }


  void DeleteObservables() {

    delete[] observable;
    delete[] tmp_observable;

  }


};


#endif
