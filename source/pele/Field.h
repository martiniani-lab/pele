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

#ifndef FIELD_H
#define FIELD_H


/*
  Field class:

  This is the basis class for the derived classes "BitTable" and
  "HashTable" which provide data structures and generic routines (such
  as e.g. query, insert, delete) for the handling of bit and hash
  tables, respectively; for further reading, see e.g. T.H. Cormen et
  al., "Introduction to Algorithms", MIT Press (2009).
*/

class Field {

public:

  virtual ~Field() {}

  virtual void Reset() = 0;
  virtual int Query(unsigned long int) = 0;
  virtual void Insert(unsigned long int, int) = 0;
  virtual void Replace(unsigned long int, int) = 0;
  virtual void Delete(unsigned long int) = 0;
  virtual int CheckKey(unsigned long int) = 0;
  virtual unsigned long int GetFillSize() = 0;
  virtual void GetInfo(bool = false) = 0;

protected:

  bool IsStatic;
  unsigned long int size;

};


/*
  BitTable class:

  This class provides a bit table data structure.
*/

class BitTable : public Field {

public:

  // constructor
  BitTable(unsigned long int, bool = true);

  // destructor
  ~BitTable();

  void Reset();
  int Query(unsigned long int);
  void Insert(unsigned long int, int);
  void Replace(unsigned long int, int);
  void Delete(unsigned long int);
  int CheckKey(unsigned long int);
  unsigned long int GetFillSize();
  void GetInfo(bool = false);

private:

  static unsigned int StaticBitTables;
  static unsigned long int static_size;
  static int* static_bt;

  int* bt;

};


/*
  HashTable class:

  This class provides a hash table data structure using the "modulo by
  prime" rule (division rule) as hash function. Collision resolution
  is treated by means of linked lists (separate chaining).
*/

class HashTable : public Field {

public:

  // constructor
  HashTable(unsigned long int, bool = true);

  // destructor
  ~HashTable();

  void Reset();
  int Query(unsigned long int);
  void Insert(unsigned long int, int);
  void Replace(unsigned long int, int);
  void Delete(unsigned long int);
  int CheckKey(unsigned long int);
  unsigned long int GetFillSize();
  void GetInfo(bool = false);

private:

  struct ListEntry {
    unsigned long int key;
    int index;
    ListEntry* next;
  };

  static unsigned int StaticHashTables;
  static unsigned long int static_size;
  static ListEntry** static_ht;

  ListEntry** ht;

  inline unsigned long int HashFunction(unsigned long int);

};


#endif
