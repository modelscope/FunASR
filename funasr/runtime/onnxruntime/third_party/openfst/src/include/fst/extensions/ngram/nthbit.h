// See www.openfst.org for extensive documentation on this weighted
// finite-state transducer library.

#ifndef FST_EXTENSIONS_NGRAM_NTHBIT_H_
#define FST_EXTENSIONS_NGRAM_NTHBIT_H_

#include <fst/types.h>
#include <fst/compat.h>

#ifdef __BMI2__
// PDEP requires BMI2.

// Returns the position (0-63) of the r-th 1 bit in v.
// 1 <= r <= CountOnes(v) <= 64.  Therefore, v must not be 0.
inline uint32 nth_bit(uint64 v, uint32 r) {
  // PDEP example from https://stackoverflow.com/a/27453505
  return __builtin_ctzll(_pdep_u64(uint64{1} << (r - 1), v));
}

#else  // !defined(__BMI2__)

extern const uint32 nth_bit_bit_offset[];

// Returns the position (0-63) of the r-th 1 bit in v.
// 1 <= r <= CountOnes(v) <= 64.  Therefore, v must not be 0.
inline uint32 nth_bit(uint64 v, uint32 r) {
  uint32 shift = 0;
  uint32 c = __builtin_popcount(v & 0xffffffff);
  uint32 mask = -(r > c);
  r -= c & mask;
  shift += (32 & mask);

  c = __builtin_popcount((v >> shift) & 0xffff);
  mask = -(r > c);
  r -= c & mask;
  shift += (16 & mask);

  c = __builtin_popcount((v >> shift) & 0xff);
  mask = -(r > c);
  r -= c & mask;
  shift += (8 & mask);

  return shift +
         ((nth_bit_bit_offset[(v >> shift) & 0xff] >> ((r - 1) << 2)) & 0xf);
}

#endif  // !defined(__BMI2__)

#endif  // FST_EXTENSIONS_NGRAM_NTHBIT_H_
