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

#include "pele/Field.h"


unsigned int BitTable::StaticBitTables = 0;
unsigned long int BitTable::static_size = 0;
int* BitTable::static_bt = NULL;


unsigned int HashTable::StaticHashTables = 0;
unsigned long int HashTable::static_size = 0;
HashTable::ListEntry** HashTable::static_ht = NULL;


BitTable::BitTable(unsigned long int s, bool flag)
{

  unsigned long int i;

  IsStatic = flag;

  if (IsStatic) {

    if (static_bt == NULL) {
      static_size = s;
      static_bt = new int[static_size];
      for (i = 0; i < static_size; i++) static_bt[i] = -1;
    }

    StaticBitTables++;
    size = static_size;
    bt = static_bt;

  }

  else {

    size = s;
    bt = new int[size];
    for (i = 0; i < size; i++) bt[i] = -1;

  }

}


BitTable::~BitTable()
{

  if (IsStatic) StaticBitTables--;

  if ((!IsStatic) || (StaticBitTables == 0)) delete[] bt;

  if ((IsStatic) && (StaticBitTables == 0)) {
    static_size = 0;
    static_bt = NULL;
  }

}


void BitTable::Reset()
{

  unsigned long int i;

  for (i = 0; i < size; i++) bt[i] = -1;

}


int BitTable::Query(unsigned long int key)
{

  return bt[key];

}


void BitTable::Insert(unsigned long int key, int index)
{

  bt[key] = index;

}


void BitTable::Replace(unsigned long int key, int index)
{

  bt[key] = index;

}


void BitTable::Delete(unsigned long int key)
{

  bt[key] = -1;

}


int BitTable::CheckKey(unsigned long int key)
{

  return bt[key];

}


unsigned long int BitTable::GetFillSize()
{

  unsigned long int i, n;

  n = 0;
  for (i = 0; i < size; i++)
    if (bt[i] != -1) n++;

  return n;

}


void BitTable::GetInfo(bool flag)
{

  unsigned long int i;

  printf("\n### Field Specifications ###############################\n\n");

  printf(" Type: Bit table, ");
  if (IsStatic) printf("static (%d)", StaticBitTables);
  else printf("non-static");
  printf("\n\n");

  printf(" Size = %ld\n", size);
  printf(" Total number of items = %ld\n", GetFillSize());

  if (flag) {
    printf("\n");
    printf(" Bit table content:\n");
    for (i = 0; i < size; i++)
      if (bt[i] != -1) printf("  %8ld : %5d\n", i, bt[i]);
  }

  printf("\n########################################################\n");

}


HashTable::HashTable(unsigned long int s, bool flag)
{

  unsigned long int i;

  IsStatic = flag;

  if (IsStatic) {

    if (static_ht == NULL) {
      static_size = s;
      static_ht = new ListEntry*[static_size];
      for (i = 0; i < static_size; i++) static_ht[i] = NULL;
    }

    StaticHashTables++;
    size = static_size;
    ht = static_ht;

  }

  else {

    size = s;
    ht = new ListEntry*[size];
    for (i = 0; i < size; i++) ht[i] = NULL;

  }

}


HashTable::~HashTable()
{

  unsigned long int i;
  ListEntry* p;

  if (IsStatic) StaticHashTables--;

  if ((!IsStatic) || (StaticHashTables == 0)) {

    for (i = 0; i < size; i++) {
      while (ht[i] != NULL) {
	p = ht[i] -> next;
	delete ht[i];
	ht[i] = p;
      }
    }

    delete[] ht;

  }

  if ((IsStatic) && (StaticHashTables == 0)) {
    static_size = 0;
    static_ht = NULL;
  }

}


unsigned long int HashTable::HashFunction(unsigned long int key)
{

  return key % size;

}


void HashTable::Reset()
{

  unsigned long int i;
  ListEntry* p;

  for (i = 0; i < size; i++) {

    while (ht[i] != NULL) {
      p = ht[i] -> next;
      delete ht[i];
      ht[i] = p;
    }

  }

}


int HashTable::Query(unsigned long int key)
{

  ListEntry* p = ht[HashFunction(key)];

  while (p != NULL) {
    if (p -> key == key) return p -> index;
    p = p -> next;
  }

  return -1;

}


void HashTable::Insert(unsigned long int key, int index)
{

  unsigned long int i = HashFunction(key);

  ListEntry* p = new ListEntry;

  p -> key = key;
  p -> index = index;
  p -> next = ht[i];

  ht[i] = p;

}


void HashTable::Replace(unsigned long int key, int index)
{

  ListEntry* p = ht[HashFunction(key)];

  while (p != NULL) {
    if (p -> key == key) {
      p -> index = index;
      return;
    }
    p = p -> next;
  }

}


void HashTable::Delete(unsigned long int key)
{

  unsigned long int i = HashFunction(key);

  ListEntry* p = ht[i];
  ListEntry* prev;

  while (p != NULL) {

    if (p -> key == key) {
      if (p == ht[i]) ht[i] = p -> next;
      else prev -> next = p -> next;
      delete p;
      return;
    }

    else {
      prev = p;
      p = p -> next;
    }

  }

}


int HashTable::CheckKey(unsigned long int key)
{

  unsigned long int i, n;
  ListEntry* p;

  n = 0;
  for (i = 0; i < size; i++) {
    p = ht[i];
    while (p != NULL) {
      if (p -> key == key) n++;
      p = p -> next;
    }
  }

  if (n == 1) {
    p = ht[HashFunction(key)];
    while (p != NULL) {
      if (p -> key == key) return p -> index;
      p = p -> next;
    }
  }

  return -1;

}


unsigned long int HashTable::GetFillSize()
{

  unsigned long int i, n;
  ListEntry* p;

  n = 0;
  for (i = 0; i < size; i++) {
    p = ht[i];
    while (p != NULL) {
      n++;
      p = p -> next;
    }
  }

  return n;

}


void HashTable::GetInfo(bool flag)
{

  unsigned long int i, n;
  ListEntry* p;

  unsigned long int total_items;
  unsigned long int non_empty_lists;
  unsigned long int max_items_list;

  unsigned long int* h;

  total_items = GetFillSize();
  h = new unsigned long int[total_items + 1];
  for (i = 0; i < total_items + 1; i++) h[i] = 0;

  max_items_list = 0;
  for (i = 0; i < size; i++) {
    p = ht[i];
    n = 0;
    while (p != NULL) {
      n++;
      p = p -> next;
    }
    h[n]++;
    if (n > max_items_list) max_items_list = n;
  }

  non_empty_lists = size - h[0];

  printf("\n### Field Specifications ###############################\n\n");

  printf(" Type: Hash table, ");
  if (IsStatic) printf("static (%d)", StaticHashTables);
  else printf("non-static");
  printf("\n\n");

  printf(" Size = %ld\n", size);
  printf(" Total number of items = %ld\n", total_items);
  printf(" Number of non-empty lists = %ld\n", non_empty_lists);
  printf(" Average number of collisions = ");
  if (non_empty_lists > 0)
    printf("%5.3f", double(total_items) / double(non_empty_lists));
  else
    printf("%5.3f", 0.0);
  printf("\n");
  printf(" Maximum number of collisions = %ld\n\n", max_items_list);

  printf(" Hash table histogram:\n");
  for (i = 0; i < total_items + 1; i++)
    if (h[i] > 0) printf("  %8ld : %8ld\n", i, h[i]);

  if (flag) {
    printf("\n");
    printf(" Hash table content:\n");
    for (i = 0; i < size; i++) {
      p = ht[i];
      while (p != NULL) {
	printf("  %8ld : %5d\n", p -> key, p -> index);
	p = p -> next;
      }
    }
  }

  printf("\n########################################################\n");

  delete[] h;

}
