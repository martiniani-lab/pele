/* modified for c++ compatibility for pele */
/* With extra function to add a single number to skip the loop */

/* INTERFACE TO FUNCTIONS FOR EXACT SUMMATION. */

/* Written by Radford M. Neal, 2015 */

#ifndef EXACTSUM_RADFORD
#define EXACTSUM_RADFORD
/* using restrict with c++ https://github.com/rurban/safeclib/issues/58 */
#define restrict __restrict__

#include <stdint.h>
#include <stdlib.h>

/* CONSTANTS DEFINING THE FLOATING POINT FORMAT. */

typedef double xsum_flt; /* C floating point type sums are done for */

typedef int64_t xsum_int;         /* Signed integer type for a fp value */
typedef uint64_t xsum_uint;       /* Unsigned integer type for a fp value */
typedef int_fast16_t xsum_expint; /* Integer type for holding an exponent */

#define XSUM_MANTISSA_BITS 52 /* Bits in fp mantissa, excludes implict 1 */
#define XSUM_EXP_BITS 11      /* Bits in fp exponent */

#define XSUM_MANTISSA_MASK                                                     \
  (((xsum_int)1 << XSUM_MANTISSA_BITS) - 1) /* Mask for mantissa bits */

#define XSUM_EXP_MASK ((1 << XSUM_EXP_BITS) - 1) /* Mask for exponent */

#define XSUM_EXP_BIAS                                                          \
  ((1 << (XSUM_EXP_BITS - 1)) - 1) /* Bias added to signed exponent */

#define XSUM_SIGN_BIT                                                          \
  (XSUM_MANTISSA_BITS + XSUM_EXP_BITS) /* Position of sign bit */

#define XSUM_SIGN_MASK ((xsum_uint)1 << XSUM_SIGN_BIT) /* Mask for sign bit */

/* CONSTANTS DEFINING THE SMALL ACCUMULATOR FORMAT. */

#define XSUM_SCHUNK_BITS 64  /* Bits in chunk of the small accumulator */
typedef int64_t xsum_schunk; /* Integer type of small accumulator chunk */

#define XSUM_LOW_EXP_BITS 5 /* # of low bits of exponent, in one chunk */

#define XSUM_LOW_EXP_MASK                                                      \
  ((1 << XSUM_LOW_EXP_BITS) - 1) /* Mask for low-order exponent bits */

#define XSUM_HIGH_EXP_BITS                                                     \
  (XSUM_EXP_BITS - XSUM_LOW_EXP_BITS) /* # of high exponent bits for index */

#define XSUM_HIGH_EXP_MASK                                                     \
  ((1 << HIGH_EXP_BITS) - 1) /* Mask for high-order exponent bits */

#define XSUM_SCHUNKS                                                           \
  ((1 << XSUM_HIGH_EXP_BITS) + 3) /* # of chunks in small accumulator */

#define XSUM_LOW_MANTISSA_BITS                                                 \
  (1 << XSUM_LOW_EXP_BITS) /* Bits in low part of mantissa */

#define XSUM_HIGH_MANTISSA_BITS                                                \
  (XSUM_MANTISSA_BITS - XSUM_LOW_MANTISSA_BITS) /* Bits in high part */

#define XSUM_LOW_MANTISSA_MASK                                                 \
  (((xsum_int)1 << XSUM_LOW_MANTISSA_BITS) - 1) /* Mask for low bits */

#define XSUM_SMALL_CARRY_BITS                                                  \
  ((XSUM_SCHUNK_BITS - 1) - XSUM_MANTISSA_BITS) /* Bits sums can carry into */

#define XSUM_SMALL_CARRY_TERMS                                                 \
  ((1 << XSUM_SMALL_CARRY_BITS) - 1) /* # terms can add before need prop. */

struct xsum_small_accumulator_t {
  xsum_schunk chunk[XSUM_SCHUNKS]; /* Chunks making up small accumulator */
  xsum_int Inf;                    /* If non-zero, +Inf, -Inf, or NaN */
  xsum_int NaN;                    /* If non-zero, a NaN value with payload */
  int adds_until_propagate;        /* Number of remaining adds before carry */
};

typedef struct xsum_small_accumulator_t
    xsum_small_accumulator; /*     propagation must be done again    */

/* CONSTANTS DEFINING THE LARGE ACCUMULATOR FORMAT. */

#define XSUM_LCHUNK_BITS 64   /* Bits in chunk of the large accumulator */
typedef uint64_t xsum_lchunk; /* Integer type of large accumulator chunk,
                                 must be EXACTLY 64 bits in size */

#define XSUM_LCOUNT_BITS (64 - XSUM_MANTISSA_BITS) /* # of bits in count */
typedef int_least16_t xsum_lcount; /* Signed int type of counts for large acc.*/

#define XSUM_LCHUNKS                                                           \
  (1 << (XSUM_EXP_BITS + 1)) /* # of chunks in large accumulator */

typedef uint_fast64_t xsum_used; /* Unsigned type for holding used flags */

struct xsum_large_accumulator_t {
  xsum_lchunk chunk[XSUM_LCHUNKS]; /* Chunks making up large accumulator */
  xsum_lcount count[XSUM_LCHUNKS]; /* Counts of # adds remaining for chunks,
                                        or -1 if not used yet or special. */
  xsum_used chunks_used[XSUM_LCHUNKS / 64]; /* Bits indicate chunks in use */
  xsum_used used_used;         /* Bits indicate chunk_used entries not 0 */
  xsum_small_accumulator sacc; /* The small accumulator to condense into */
};

typedef struct xsum_large_accumulator_t xsum_large_accumulator;

/* TYPE FOR LENGTHS OF ARRAYS.  Must be a signed integer type. */

typedef int xsum_length;

/* FUNCTIONS FOR EXACT SUMMATION. */

void xsum_small_init(xsum_small_accumulator *restrict);
void xsum_small_add1(xsum_small_accumulator *restrict, xsum_flt);
void xsum_small_addv(xsum_small_accumulator *restrict, const xsum_flt *restrict,
                     xsum_length);
void xsum_small_add_sqnorm(xsum_small_accumulator *restrict,
                           const xsum_flt *restrict, xsum_length);
void xsum_small_add_dot(xsum_small_accumulator *restrict, const xsum_flt *,
                        const xsum_flt *, xsum_length);
xsum_flt xsum_small_round(xsum_small_accumulator *restrict);
void xsum_small_display(xsum_small_accumulator *restrict);
int xsum_small_chunks_used(xsum_small_accumulator *restrict);

void xsum_large_init(xsum_large_accumulator *restrict);
void xsum_large_addv(xsum_large_accumulator *restrict, const xsum_flt *restrict,
                     xsum_length);
void xsum_large_add_sqnorm(xsum_large_accumulator *restrict,
                           const xsum_flt *restrict, xsum_length);
void xsum_large_add_dot(xsum_large_accumulator *restrict, const xsum_flt *,
                        const xsum_flt *, xsum_length);
xsum_flt xsum_large_round(xsum_large_accumulator *restrict);
void xsum_large_display(xsum_large_accumulator *restrict);
int xsum_large_chunks_used(xsum_large_accumulator *restrict);

/* FUNCTIONS FOR DOUBLE AND OTHER INEXACT SUMMATION. */

xsum_flt xsum_sum_double(const xsum_flt *restrict, xsum_length);
xsum_flt xsum_sum_double_not_ordered(const xsum_flt *restrict, xsum_length);
xsum_flt xsum_sum_float128(const xsum_flt *restrict, xsum_length);
xsum_flt xsum_sum_kahan(const xsum_flt *restrict, xsum_length);
xsum_flt xsum_sqnorm_double(const xsum_flt *restrict, xsum_length);
xsum_flt xsum_sqnorm_double_not_ordered(const xsum_flt *restrict, xsum_length);
xsum_flt xsum_dot_double(const xsum_flt *, const xsum_flt *, xsum_length);
xsum_flt xsum_dot_double_not_ordered(const xsum_flt *, const xsum_flt *,
                                     xsum_length);

/* DEBUG FLAG.  Set to non-zero for debug ouptut.  Ignored unless xsum.c
   is compiled with -DDEBUG. */

extern int xsum_debug;

/* EXTRA ADDNUMBER FUNCTION for large accumulator */

void xsum_large_add1(xsum_large_accumulator *restrict, xsum_flt);

/* ADD A SMALL ACCUMULATOR TO ANOTHER SMALL ACCUMULATOR.  */
void xsum_small_add_acc(xsum_small_accumulator *restrict sacc,
                        xsum_small_accumulator *restrict sacc_to_add);
void xsum_small_subtract_acc_and_set(
    xsum_small_accumulator *restrict sacc,
    xsum_small_accumulator *restrict sacc_to_subtract,
    xsum_small_accumulator *restrict sacc_to_assign);

/* ASSIGN A SMALL ACCUMULATOR TO ANOTHER SMALL ACCUMULATOR */

void xsum_small_equal(xsum_small_accumulator *restrict sacc,
                      xsum_small_accumulator *restrict sacc_to_assign);

/*  ASSIGNS THE ACCUMULATOR DIFFERENCE TO A THIRD ACCUMULATOR SET TO ZERO */

#endif