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

#ifndef HPMODEL_H
#define HPMODEL_H

#include "Vector.h"
#include "Field.h"
#include "Model.h"
#include "base_potential.h"
#include "array.h"


/*
  HPModel class:

  This is a derived "physical model" class from the basis class
  "Model" and contains all relevant data structures and routines to
  run Monte Carlo simulations for the hydrophobic-polar (HP) protein
  model or the interacting self-avoiding walk (ISAW) polymer model on
  the simple (hyper-)cubic lattice in dimension d >= 2. For efficiency
  reasons (e.g. performing Monte Carlo trial moves, checking
  self-overlaps, counting non-bonded HH contacts), the configuration
  of the protein/polymer is stored both as a set of d-dimensional
  vectors for the monomer coordinates and as entries in a
  d-dimensional occupancy field. Currently, the following three types
  of Monte Carlo trial moves are implemented: pull moves,
  bond-rebridging moves, pivot moves. The principal observable is the
  number of non-bonded HH contacts (the energy of these models is just
  the negative of this quantity). To each Monte Carlo trial move
  belongs the updating of this observable.

*/

namespace pele  {

    class HPModel : public Model, public BasePotential {

    public:

        // constructor
        // generates a "HPModel" class instance
        // 1. input file name
        HPModel(const char*);
        // destructor
        ~HPModel();

        // see comments for the basis class "Model" for the following three
        // routines
        void WriteState(int = 0, const char* = NULL);
        void DoMCMove();
        void UnDoMCMove();

    private:

        // the two monomer types: "H" and "P"
        enum MonomerType {hydrophobic = 0, polar = 1};

        // physical dimension d of the protein/polymer (d >= 2)
        int PolymerDim;
        // the occupancy field can be either a bit table or a hash table,
        // see class "Field"
        int OccupancyFieldType;
        // array specifying the relative frequency of each type of Monte
        // Carlo trial move; as an example: [0.5, 0.8] means 50% pull moves,
        // 30% bond-rebridging moves and 20% pivot moves
        double MoveFraction[2];

        // various arrays to collect statistics for the three (3) types of
        // Monte Carlo trial moves
        unsigned long int MoveTrials[3];
        unsigned long int MoveRejects[3];
        unsigned long int MoveDiscards[3];
        unsigned long int BeadsTrials[3];
        unsigned long int BeadsRejects[3];

        // two parameters specifying the d-dimensional occupancy field
        long int LSize;
        unsigned long int* LOffset;

        // total number of pivot operations in d dimensions and pointer to
        // the corresponding pivot matrices
        long int NPiOps;
        Matrix** pivot;

        // total number of monomers (protein/polymer chain length)
        int NumberOfMonomers;
        // number of "H" monomers (equals 'NumberOfMonomers' in case of ISAW
        // model)
        int NumberOfHMonomers;
        // number of chain-internal "HH" contacts (equals
        // 'NumberOfMonomers-1' in case of ISAW model)
        int InternalHHContacts;

        // flag indicating whether the current system is an "HP model" or an
        // "ISAW model"; it is an "HP model" if there is at least one "P"
        // monomer in the sequence; 0 = "ISAW model", 1 = "HP model"
        int HPModelFlag;

        // pointer to "Field" class instance (either "BitTable" or
        // "HashTable")
        Field* OccupancyField;

        // pointer to sequence array of "H"'s and "P"'s
        MonomerType* seq;

        // pointer to array of vectors for the monomer coordinates
        Vector** coords;
        // same for temporary storage
        Vector** tmpcoords;

        // pointers to arrays of monomer locations in the d-dimensional
        // occupancy field
        unsigned long int* site;
        unsigned long int* tmpsite;

        // index for the type of Monte Carlo trial move; 0 = pull move; 1 =
        // bond-rebridging move; 2 = pivot move
        int MCMoveType;

        // "first" and "last" monomers (and "direction") of the chain
        // fragment whose monomer positions have changed after a Monte Carlo
        // trial move (only used for pull moves and pivot moves)
        int BeadsStart, BeadsEnd, BeadsIncr;
        // length of chain fragment whose monomer positions have changed
        // after a Monte Carlo trial move (for all types of Monte Carlo
        // trial moves); "threshold" length of chain fragment determining
        // which method to use to evaluate the number of non-bonded HH
        // contacts (either for the entire chain or only for the chain
        // fragment)
        int BeadsInterval, BeadsIntervalThreshold;

        // maximal number of all possible pull moves
        long int MaxPull2Moves;
        // pointer to a "look-up" table with the relative displacements for
        // all possible pull moves
        int (*Pull2Set)[6];

        // pointer to temporary "buffer array" to keep vector addresses for
        // the monomer coordinates
        Vector** CoordsPointer;
        // pointer to temporary "buffer array" to keep monomer indices
        int* CoordsIndex;

        // initializes the pivot matrices in d dimensions
        void InitPivotMatrices();
        // generates a "look-up" table with the relative displacements for
        // all possible pull moves (labeled "pull-2" moves because there are
        // generally two monomers which are pulled at a time)
        void InitPull2Set();
        // does various model integrity checks (in particular, checks the
        // consistency between the two representations of storing the
        // protein/polymer configuration, i.e. d-dimensional vectors for the
        // monomer coordinates vs entries in a d-dimensional occupancy
        // field)
        void CheckModelIntegrity();
        // returns the monomer location in the d-dimensional occupancy field
        // from the d-dimensional vector for the monomer coordinates
        // 1. reference to a "Vector" object (monomer coordinates)
        unsigned long int GetSite(Vector&);

        // performs a pull move (labeled a "pull-2" move because there are
        // generally two monomers which are pulled at a time)
        void Pull2Move();
        // performs a bond-rebridging move
        void BondRebridgingMove();
        // performs a pivot move
        void PivotMove();

        // evaluates the number of non-bonded HH contacts (principal
        // observable) for the HP model running over the entire protein
        // chain
        void GetObservablesHP();
        // evaluates the number of non-bonded HH contacts (principal
        // observable) for the ISAW model running over the entire polymer
        // chain
        void GetObservablesISAW();

        // evaluates the number of non-bonded HH contacts (principal
        // observable) for the HP model running only over the protein chain
        // fragment whose monomer locations have changed after a Monte Carlo
        // trial move
        void GetObservablesPartHP(int, int, long int);
        // evaluates the number of non-bonded HH contacts (principal
        // observable) for the ISAW model running only over the polymer
        // chain fragment whose monomer locations have changed after a Monte
        // Carlo trial move
        void GetObservablesPartISAW(int, int, long int);

        // prints error messages on standard error stream
        // 1. error code
        // 2. string argument (optional)
        void ErrorMsg(int, const char* = NULL);


        // Pele McPele interfacing functions

        double get_energy(Array<double> const & x);

        // Conversion interface function for coordinates
        Vector** pele_array_to_vector(Array<double> x);

        Array<double> vector_to_pele_array(Vector** v);

        // 
        
        // takestep
        
        
    };

};
#endif
