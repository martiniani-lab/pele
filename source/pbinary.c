/* PRINT INTEGERS AND FLOATS IN BINARY. */

/* Written by Radford M. Neal, 2015 */

#include "pele/pbinary.h"
#include <stdio.h>


/* PRINT 64-BIT INTEGER IN BINARY.  Prints only the low-order n bits. */

void pbinary_int64 (int64_t v, int n)
{
  int i;
  for (i = n-1; i >= 0; i--)
  { printf ("%d", (int) ((v>>i)&1));
  }
}


/* PRINT DOUBLE-PRECISION FLOATING POINT VALUE IN BINARY. */

void pbinary_double (double d)
{
  union { double f; int64_t i; } u;
  int64_t exp;

  u.f = d;
  printf (u.i < 0 ? "- " : "+ ");
  exp = (u.i >> 52) & 0x7ff;
  pbinary_int64 (exp, 11);
  if (exp == 0) printf (" (denorm) ");
  else if (exp == 0x7ff) printf (" (InfNaN) ");
  else printf(" (%+06d) ", (int) (exp - 1023));
  pbinary_int64 (u.i & 0xfffffffffffffL, 52);
}
