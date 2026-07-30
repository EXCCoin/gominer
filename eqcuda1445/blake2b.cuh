// Blake2-B CUDA Implementation
// tpruvot@github July 2016
// permission granted to use under MIT license
// modified for use in Zcash by John Tromp September 2016

#pragma once
#include <cstdint>
#include "portable_endian.h"
#include "blake/blake2.h"

typedef unsigned char uchar;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

// uint2 direct ops by c++ operator definitions
static __device__ __forceinline__ uint2 operator^ (uint2 a, uint2 b) {
  return make_uint2(a.x ^ b.x, a.y ^ b.y);
}
// uint2 ROR/ROL methods
__device__ __forceinline__ uint2 ROR2(const uint2 a, const int offset) {
  uint2 result;
#if __CUDA_ARCH__ > 300
  if (offset < 32) {
          asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.x), "r"(a.y), "r"(offset));
          asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.y), "r"(a.x), "r"(offset));
  } else /* if (offset < 64) */ {
          /* offset SHOULD BE < 64 ! */
          asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.y), "r"(a.x), "r"(offset));
          asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
  }
#else
  if (!offset)
          result = a;
  else if (offset < 32) {
          result.y = ((a.y >> offset) | (a.x << (32 - offset)));
          result.x = ((a.x >> offset) | (a.y << (32 - offset)));
  } else if (offset == 32) {
          result.y = a.x;
          result.x = a.y;
  } else {
          result.y = ((a.x >> (offset - 32)) | (a.y << (64 - offset)));
          result.x = ((a.y >> (offset - 32)) | (a.x << (64 - offset)));
  }
#endif
  return result;
}

__device__ __forceinline__ uint2 SWAPUINT2(uint2 value) {
  return make_uint2(value.y, value.x);
}

#ifdef __CUDA_ARCH__
__device__ __inline__ uint2 ROR24(const uint2 a) {
  uint2 result;
  result.x = __byte_perm(a.y, a.x, 0x2107);
  result.y = __byte_perm(a.y, a.x, 0x6543);
  return result;
}
__device__ __inline__ uint2 ROR16(const uint2 a) {
  uint2 result;
  result.x = __byte_perm(a.y, a.x, 0x1076);
  result.y = __byte_perm(a.y, a.x, 0x5432);
  return result;
}
#else
#define ROR24(u) ROR2(u,24)
#define ROR16(u) ROR2(u,16)
#endif

static __constant__ const int8_t blake2b_sigma[12][16] = {
  { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15 } ,
  { 14, 10, 4,  8,  9,  15, 13, 6,  1,  12, 0,  2,  11, 7,  5,  3  } ,
  { 11, 8,  12, 0,  5,  2,  15, 13, 10, 14, 3,  6,  7,  1,  9,  4  } ,
  { 7,  9,  3,  1,  13, 12, 11, 14, 2,  6,  5,  10, 4,  0,  15, 8  } ,
  { 9,  0,  5,  7,  2,  4,  10, 15, 14, 1,  11, 12, 6,  8,  3,  13 } ,
  { 2,  12, 6,  10, 0,  11, 8,  3,  4,  13, 7,  5,  15, 14, 1,  9  } ,
  { 12, 5,  1,  15, 14, 13, 4,  10, 0,  7,  6,  3,  9,  2,  8,  11 } ,
  { 13, 11, 7,  14, 12, 1,  3,  9,  5,  0,  15, 4,  8,  6,  2,  10 } ,
  { 6,  15, 14, 9,  11, 3,  0,  8,  12, 2,  13, 7,  1,  4,  10, 5  } ,
  { 10, 2,  8,  4,  7,  6,  1,  5,  15, 11, 9,  14, 3,  12, 13, 0  } ,
  { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15 } ,
  { 14, 10, 4,  8,  9,  15, 13, 6,  1,  12, 0,  2,  11, 7,  5,  3  }
};

__device__ __forceinline__ static void G(const int r, const int i, u64 &a, u64 &b, u64 &c, u64 &d, u64 const m[16]) {
  a = a + b + m[ blake2b_sigma[r][2*i] ];
  ((uint2*)&d)[0] = SWAPUINT2( ((uint2*)&d)[0] ^ ((uint2*)&a)[0] );
  c = c + d;
  ((uint2*)&b)[0] = ROR24( ((uint2*)&b)[0] ^ ((uint2*)&c)[0] );
  a = a + b + m[ blake2b_sigma[r][2*i+1] ];
  ((uint2*)&d)[0] = ROR16( ((uint2*)&d)[0] ^ ((uint2*)&a)[0] );
  c = c + d;
  ((uint2*)&b)[0] = ROR2( ((uint2*)&b)[0] ^ ((uint2*)&c)[0], 63U);
}

__device__ __forceinline__ static void Gxy(u64 &a, u64 &b, u64 &c, u64 &d,
                                            u64 x, u64 y) {
  a = a + b + x;
  ((uint2*)&d)[0] = SWAPUINT2(((uint2*)&d)[0] ^ ((uint2*)&a)[0]);
  c = c + d;
  ((uint2*)&b)[0] = ROR24(((uint2*)&b)[0] ^ ((uint2*)&c)[0]);
  a = a + b + y;
  ((uint2*)&d)[0] = ROR16(((uint2*)&d)[0] ^ ((uint2*)&a)[0]);
  c = c + d;
  ((uint2*)&b)[0] = ROR2(((uint2*)&b)[0] ^ ((uint2*)&c)[0], 63U);
}

#define ROUND(r) \
  G(r, 0, v[0], v[4], v[ 8], v[12], m); \
  G(r, 1, v[1], v[5], v[ 9], v[13], m); \
  G(r, 2, v[2], v[6], v[10], v[14], m); \
  G(r, 3, v[3], v[7], v[11], v[15], m); \
  G(r, 4, v[0], v[5], v[10], v[15], m); \
  G(r, 5, v[1], v[6], v[11], v[12], m); \
  G(r, 6, v[2], v[7], v[ 8], v[13], m); \
  G(r, 7, v[3], v[4], v[ 9], v[14], m);

// Per-thread precomputed blake2b state: everything that does not depend on
// the hash index. digitH hoists this out of its block loop; without hoisting,
// each of the ~11M hash calls per solve re-issues the same 15 global loads.
struct blake2b_pre {
  u64 h[8];
  u64 m0, m1, m2, m3, m4, m5, m6lo;
  u64 v3base; // h3 + h7 + low message word 6
  u64 r0[12]; // first three index-independent columns of round 0
};

__device__ __forceinline__ void blake2b_precompute(const blake2b_state *state, blake2b_pre *pre) {
  const u64 *d_data = (const u64 *)state->buf;
#pragma unroll
  for (u32 i = 0; i < 8; i++)
    pre->h[i] = state->h[i];
  pre->m0 = d_data[0];
  pre->m1 = d_data[1];
  pre->m2 = d_data[2];
  pre->m3 = d_data[3];
  pre->m4 = d_data[4];
  pre->m5 = d_data[5];
  pre->m6lo = d_data[6] & 0xffffffffULL;
  const u64 t = state->counter + state->buflen + sizeof(u32);
  pre->v3base = pre->h[3] + pre->h[7] + pre->m6lo;

  pre->r0[0] = pre->h[0];
  pre->r0[1] = pre->h[4];
  pre->r0[2] = 0x6a09e667f3bcc908ULL;
  pre->r0[3] = 0x510e527fade682d1ULL ^ t;
  Gxy(pre->r0[0], pre->r0[1], pre->r0[2], pre->r0[3], pre->m0, pre->m1);
  pre->r0[4] = pre->h[1];
  pre->r0[5] = pre->h[5];
  pre->r0[6] = 0xbb67ae8584caa73bULL;
  pre->r0[7] = 0x9b05688c2b3e6c1fULL;
  Gxy(pre->r0[4], pre->r0[5], pre->r0[6], pre->r0[7], pre->m2, pre->m3);
  pre->r0[8] = pre->h[2];
  pre->r0[9] = pre->h[6];
  pre->r0[10] = 0x3c6ef372fe94f82bULL;
  pre->r0[11] = 0x1f83d9abfb41bd6bULL ^ 0xffffffffffffffffULL;
  Gxy(pre->r0[8], pre->r0[9], pre->r0[10], pre->r0[11], pre->m4, pre->m5);
}

// Single final-block blake2b compression from the precomputed state. Rounds
// are fully unrolled with literal sigma indices and scalar variables;
// additions of the nine zero message words are elided.
__device__ void blake2b_gpu_hash_pre(const blake2b_pre *pre, u32 idx, uchar *hash) {
  const u64 m0 = pre->m0;
  const u64 m1 = pre->m1;
  const u64 m2 = pre->m2;
  const u64 m3 = pre->m3;
  const u64 m4 = pre->m4;
  const u64 m5 = pre->m5;
  const u64 m6 = pre->m6lo | ((u64)idx << 32);

  u64 v0 = pre->r0[0];
  u64 v1 = pre->r0[4];
  u64 v2 = pre->r0[8];
  u64 v3 = pre->v3base;
  u64 v4 = pre->r0[1];
  u64 v5 = pre->r0[5];
  u64 v6 = pre->r0[9];
  u64 v7 = pre->h[7];
  u64 v8 = pre->r0[2];
  u64 v9 = pre->r0[6];
  u64 v10 = pre->r0[10];
  u64 v11 = 0xa54ff53a5f1d36f1ULL;
  u64 v12 = pre->r0[3];
  u64 v13 = pre->r0[7];
  u64 v14 = pre->r0[11];
  u64 v15 = 0x5be0cd19137e2179ULL;

  // round 0
  v3 = v3 + ((u64)idx << 32);
  ((uint2*)&v15)[0] = SWAPUINT2( ((uint2*)&v15)[0] ^ ((uint2*)&v3)[0] );
  v11 = v11 + v15;
  ((uint2*)&v7)[0] = ROR24( ((uint2*)&v7)[0] ^ ((uint2*)&v11)[0] );
  v3 = v3 + v7;
  ((uint2*)&v15)[0] = ROR16( ((uint2*)&v15)[0] ^ ((uint2*)&v3)[0] );
  v11 = v11 + v15;
  ((uint2*)&v7)[0] = ROR2( ((uint2*)&v7)[0] ^ ((uint2*)&v11)[0], 63U );
  v0 = v0 + v5;
  ((uint2*)&v15)[0] = SWAPUINT2( ((uint2*)&v15)[0] ^ ((uint2*)&v0)[0] );
  v10 = v10 + v15;
  ((uint2*)&v5)[0] = ROR24( ((uint2*)&v5)[0] ^ ((uint2*)&v10)[0] );
  v0 = v0 + v5;
  ((uint2*)&v15)[0] = ROR16( ((uint2*)&v15)[0] ^ ((uint2*)&v0)[0] );
  v10 = v10 + v15;
  ((uint2*)&v5)[0] = ROR2( ((uint2*)&v5)[0] ^ ((uint2*)&v10)[0], 63U );
  v1 = v1 + v6;
  ((uint2*)&v12)[0] = SWAPUINT2( ((uint2*)&v12)[0] ^ ((uint2*)&v1)[0] );
  v11 = v11 + v12;
  ((uint2*)&v6)[0] = ROR24( ((uint2*)&v6)[0] ^ ((uint2*)&v11)[0] );
  v1 = v1 + v6;
  ((uint2*)&v12)[0] = ROR16( ((uint2*)&v12)[0] ^ ((uint2*)&v1)[0] );
  v11 = v11 + v12;
  ((uint2*)&v6)[0] = ROR2( ((uint2*)&v6)[0] ^ ((uint2*)&v11)[0], 63U );
  v2 = v2 + v7;
  ((uint2*)&v13)[0] = SWAPUINT2( ((uint2*)&v13)[0] ^ ((uint2*)&v2)[0] );
  v8 = v8 + v13;
  ((uint2*)&v7)[0] = ROR24( ((uint2*)&v7)[0] ^ ((uint2*)&v8)[0] );
  v2 = v2 + v7;
  ((uint2*)&v13)[0] = ROR16( ((uint2*)&v13)[0] ^ ((uint2*)&v2)[0] );
  v8 = v8 + v13;
  ((uint2*)&v7)[0] = ROR2( ((uint2*)&v7)[0] ^ ((uint2*)&v8)[0], 63U );
  v3 = v3 + v4;
  ((uint2*)&v14)[0] = SWAPUINT2( ((uint2*)&v14)[0] ^ ((uint2*)&v3)[0] );
  v9 = v9 + v14;
  ((uint2*)&v4)[0] = ROR24( ((uint2*)&v4)[0] ^ ((uint2*)&v9)[0] );
  v3 = v3 + v4;
  ((uint2*)&v14)[0] = ROR16( ((uint2*)&v14)[0] ^ ((uint2*)&v3)[0] );
  v9 = v9 + v14;
  ((uint2*)&v4)[0] = ROR2( ((uint2*)&v4)[0] ^ ((uint2*)&v9)[0], 63U );
  // round 1
  v0 = v0 + v4;
  ((uint2*)&v12)[0] = SWAPUINT2( ((uint2*)&v12)[0] ^ ((uint2*)&v0)[0] );
  v8 = v8 + v12;
  ((uint2*)&v4)[0] = ROR24( ((uint2*)&v4)[0] ^ ((uint2*)&v8)[0] );
  v0 = v0 + v4;
  ((uint2*)&v12)[0] = ROR16( ((uint2*)&v12)[0] ^ ((uint2*)&v0)[0] );
  v8 = v8 + v12;
  ((uint2*)&v4)[0] = ROR2( ((uint2*)&v4)[0] ^ ((uint2*)&v8)[0], 63U );
  v1 = v1 + v5 + m4;
  ((uint2*)&v13)[0] = SWAPUINT2( ((uint2*)&v13)[0] ^ ((uint2*)&v1)[0] );
  v9 = v9 + v13;
  ((uint2*)&v5)[0] = ROR24( ((uint2*)&v5)[0] ^ ((uint2*)&v9)[0] );
  v1 = v1 + v5;
  ((uint2*)&v13)[0] = ROR16( ((uint2*)&v13)[0] ^ ((uint2*)&v1)[0] );
  v9 = v9 + v13;
  ((uint2*)&v5)[0] = ROR2( ((uint2*)&v5)[0] ^ ((uint2*)&v9)[0], 63U );
  v2 = v2 + v6;
  ((uint2*)&v14)[0] = SWAPUINT2( ((uint2*)&v14)[0] ^ ((uint2*)&v2)[0] );
  v10 = v10 + v14;
  ((uint2*)&v6)[0] = ROR24( ((uint2*)&v6)[0] ^ ((uint2*)&v10)[0] );
  v2 = v2 + v6;
  ((uint2*)&v14)[0] = ROR16( ((uint2*)&v14)[0] ^ ((uint2*)&v2)[0] );
  v10 = v10 + v14;
  ((uint2*)&v6)[0] = ROR2( ((uint2*)&v6)[0] ^ ((uint2*)&v10)[0], 63U );
  v3 = v3 + v7;
  ((uint2*)&v15)[0] = SWAPUINT2( ((uint2*)&v15)[0] ^ ((uint2*)&v3)[0] );
  v11 = v11 + v15;
  ((uint2*)&v7)[0] = ROR24( ((uint2*)&v7)[0] ^ ((uint2*)&v11)[0] );
  v3 = v3 + v7 + m6;
  ((uint2*)&v15)[0] = ROR16( ((uint2*)&v15)[0] ^ ((uint2*)&v3)[0] );
  v11 = v11 + v15;
  ((uint2*)&v7)[0] = ROR2( ((uint2*)&v7)[0] ^ ((uint2*)&v11)[0], 63U );
  v0 = v0 + v5 + m1;
  ((uint2*)&v15)[0] = SWAPUINT2( ((uint2*)&v15)[0] ^ ((uint2*)&v0)[0] );
  v10 = v10 + v15;
  ((uint2*)&v5)[0] = ROR24( ((uint2*)&v5)[0] ^ ((uint2*)&v10)[0] );
  v0 = v0 + v5;
  ((uint2*)&v15)[0] = ROR16( ((uint2*)&v15)[0] ^ ((uint2*)&v0)[0] );
  v10 = v10 + v15;
  ((uint2*)&v5)[0] = ROR2( ((uint2*)&v5)[0] ^ ((uint2*)&v10)[0], 63U );
  v1 = v1 + v6 + m0;
  ((uint2*)&v12)[0] = SWAPUINT2( ((uint2*)&v12)[0] ^ ((uint2*)&v1)[0] );
  v11 = v11 + v12;
  ((uint2*)&v6)[0] = ROR24( ((uint2*)&v6)[0] ^ ((uint2*)&v11)[0] );
  v1 = v1 + v6 + m2;
  ((uint2*)&v12)[0] = ROR16( ((uint2*)&v12)[0] ^ ((uint2*)&v1)[0] );
  v11 = v11 + v12;
  ((uint2*)&v6)[0] = ROR2( ((uint2*)&v6)[0] ^ ((uint2*)&v11)[0], 63U );
  v2 = v2 + v7;
  ((uint2*)&v13)[0] = SWAPUINT2( ((uint2*)&v13)[0] ^ ((uint2*)&v2)[0] );
  v8 = v8 + v13;
  ((uint2*)&v7)[0] = ROR24( ((uint2*)&v7)[0] ^ ((uint2*)&v8)[0] );
  v2 = v2 + v7;
  ((uint2*)&v13)[0] = ROR16( ((uint2*)&v13)[0] ^ ((uint2*)&v2)[0] );
  v8 = v8 + v13;
  ((uint2*)&v7)[0] = ROR2( ((uint2*)&v7)[0] ^ ((uint2*)&v8)[0], 63U );
  v3 = v3 + v4 + m5;
  ((uint2*)&v14)[0] = SWAPUINT2( ((uint2*)&v14)[0] ^ ((uint2*)&v3)[0] );
  v9 = v9 + v14;
  ((uint2*)&v4)[0] = ROR24( ((uint2*)&v4)[0] ^ ((uint2*)&v9)[0] );
  v3 = v3 + v4 + m3;
  ((uint2*)&v14)[0] = ROR16( ((uint2*)&v14)[0] ^ ((uint2*)&v3)[0] );
  v9 = v9 + v14;
  ((uint2*)&v4)[0] = ROR2( ((uint2*)&v4)[0] ^ ((uint2*)&v9)[0], 63U );
  // round 2
  v0 = v0 + v4;
  ((uint2*)&v12)[0] = SWAPUINT2( ((uint2*)&v12)[0] ^ ((uint2*)&v0)[0] );
  v8 = v8 + v12;
  ((uint2*)&v4)[0] = ROR24( ((uint2*)&v4)[0] ^ ((uint2*)&v8)[0] );
  v0 = v0 + v4;
  ((uint2*)&v12)[0] = ROR16( ((uint2*)&v12)[0] ^ ((uint2*)&v0)[0] );
  v8 = v8 + v12;
  ((uint2*)&v4)[0] = ROR2( ((uint2*)&v4)[0] ^ ((uint2*)&v8)[0], 63U );
  v1 = v1 + v5;
  ((uint2*)&v13)[0] = SWAPUINT2( ((uint2*)&v13)[0] ^ ((uint2*)&v1)[0] );
  v9 = v9 + v13;
  ((uint2*)&v5)[0] = ROR24( ((uint2*)&v5)[0] ^ ((uint2*)&v9)[0] );
  v1 = v1 + v5 + m0;
  ((uint2*)&v13)[0] = ROR16( ((uint2*)&v13)[0] ^ ((uint2*)&v1)[0] );
  v9 = v9 + v13;
  ((uint2*)&v5)[0] = ROR2( ((uint2*)&v5)[0] ^ ((uint2*)&v9)[0], 63U );
  v2 = v2 + v6 + m5;
  ((uint2*)&v14)[0] = SWAPUINT2( ((uint2*)&v14)[0] ^ ((uint2*)&v2)[0] );
  v10 = v10 + v14;
  ((uint2*)&v6)[0] = ROR24( ((uint2*)&v6)[0] ^ ((uint2*)&v10)[0] );
  v2 = v2 + v6 + m2;
  ((uint2*)&v14)[0] = ROR16( ((uint2*)&v14)[0] ^ ((uint2*)&v2)[0] );
  v10 = v10 + v14;
  ((uint2*)&v6)[0] = ROR2( ((uint2*)&v6)[0] ^ ((uint2*)&v10)[0], 63U );
  v3 = v3 + v7;
  ((uint2*)&v15)[0] = SWAPUINT2( ((uint2*)&v15)[0] ^ ((uint2*)&v3)[0] );
  v11 = v11 + v15;
  ((uint2*)&v7)[0] = ROR24( ((uint2*)&v7)[0] ^ ((uint2*)&v11)[0] );
  v3 = v3 + v7;
  ((uint2*)&v15)[0] = ROR16( ((uint2*)&v15)[0] ^ ((uint2*)&v3)[0] );
  v11 = v11 + v15;
  ((uint2*)&v7)[0] = ROR2( ((uint2*)&v7)[0] ^ ((uint2*)&v11)[0], 63U );
  v0 = v0 + v5;
  ((uint2*)&v15)[0] = SWAPUINT2( ((uint2*)&v15)[0] ^ ((uint2*)&v0)[0] );
  v10 = v10 + v15;
  ((uint2*)&v5)[0] = ROR24( ((uint2*)&v5)[0] ^ ((uint2*)&v10)[0] );
  v0 = v0 + v5;
  ((uint2*)&v15)[0] = ROR16( ((uint2*)&v15)[0] ^ ((uint2*)&v0)[0] );
  v10 = v10 + v15;
  ((uint2*)&v5)[0] = ROR2( ((uint2*)&v5)[0] ^ ((uint2*)&v10)[0], 63U );
  v1 = v1 + v6 + m3;
  ((uint2*)&v12)[0] = SWAPUINT2( ((uint2*)&v12)[0] ^ ((uint2*)&v1)[0] );
  v11 = v11 + v12;
  ((uint2*)&v6)[0] = ROR24( ((uint2*)&v6)[0] ^ ((uint2*)&v11)[0] );
  v1 = v1 + v6 + m6;
  ((uint2*)&v12)[0] = ROR16( ((uint2*)&v12)[0] ^ ((uint2*)&v1)[0] );
  v11 = v11 + v12;
  ((uint2*)&v6)[0] = ROR2( ((uint2*)&v6)[0] ^ ((uint2*)&v11)[0], 63U );
  v2 = v2 + v7;
  ((uint2*)&v13)[0] = SWAPUINT2( ((uint2*)&v13)[0] ^ ((uint2*)&v2)[0] );
  v8 = v8 + v13;
  ((uint2*)&v7)[0] = ROR24( ((uint2*)&v7)[0] ^ ((uint2*)&v8)[0] );
  v2 = v2 + v7 + m1;
  ((uint2*)&v13)[0] = ROR16( ((uint2*)&v13)[0] ^ ((uint2*)&v2)[0] );
  v8 = v8 + v13;
  ((uint2*)&v7)[0] = ROR2( ((uint2*)&v7)[0] ^ ((uint2*)&v8)[0], 63U );
  v3 = v3 + v4;
  ((uint2*)&v14)[0] = SWAPUINT2( ((uint2*)&v14)[0] ^ ((uint2*)&v3)[0] );
  v9 = v9 + v14;
  ((uint2*)&v4)[0] = ROR24( ((uint2*)&v4)[0] ^ ((uint2*)&v9)[0] );
  v3 = v3 + v4 + m4;
  ((uint2*)&v14)[0] = ROR16( ((uint2*)&v14)[0] ^ ((uint2*)&v3)[0] );
  v9 = v9 + v14;
  ((uint2*)&v4)[0] = ROR2( ((uint2*)&v4)[0] ^ ((uint2*)&v9)[0], 63U );
  // round 3
  v0 = v0 + v4;
  ((uint2*)&v12)[0] = SWAPUINT2( ((uint2*)&v12)[0] ^ ((uint2*)&v0)[0] );
  v8 = v8 + v12;
  ((uint2*)&v4)[0] = ROR24( ((uint2*)&v4)[0] ^ ((uint2*)&v8)[0] );
  v0 = v0 + v4;
  ((uint2*)&v12)[0] = ROR16( ((uint2*)&v12)[0] ^ ((uint2*)&v0)[0] );
  v8 = v8 + v12;
  ((uint2*)&v4)[0] = ROR2( ((uint2*)&v4)[0] ^ ((uint2*)&v8)[0], 63U );
  v1 = v1 + v5 + m3;
  ((uint2*)&v13)[0] = SWAPUINT2( ((uint2*)&v13)[0] ^ ((uint2*)&v1)[0] );
  v9 = v9 + v13;
  ((uint2*)&v5)[0] = ROR24( ((uint2*)&v5)[0] ^ ((uint2*)&v9)[0] );
  v1 = v1 + v5 + m1;
  ((uint2*)&v13)[0] = ROR16( ((uint2*)&v13)[0] ^ ((uint2*)&v1)[0] );
  v9 = v9 + v13;
  ((uint2*)&v5)[0] = ROR2( ((uint2*)&v5)[0] ^ ((uint2*)&v9)[0], 63U );
  v2 = v2 + v6;
  ((uint2*)&v14)[0] = SWAPUINT2( ((uint2*)&v14)[0] ^ ((uint2*)&v2)[0] );
  v10 = v10 + v14;
  ((uint2*)&v6)[0] = ROR24( ((uint2*)&v6)[0] ^ ((uint2*)&v10)[0] );
  v2 = v2 + v6;
  ((uint2*)&v14)[0] = ROR16( ((uint2*)&v14)[0] ^ ((uint2*)&v2)[0] );
  v10 = v10 + v14;
  ((uint2*)&v6)[0] = ROR2( ((uint2*)&v6)[0] ^ ((uint2*)&v10)[0], 63U );
  v3 = v3 + v7;
  ((uint2*)&v15)[0] = SWAPUINT2( ((uint2*)&v15)[0] ^ ((uint2*)&v3)[0] );
  v11 = v11 + v15;
  ((uint2*)&v7)[0] = ROR24( ((uint2*)&v7)[0] ^ ((uint2*)&v11)[0] );
  v3 = v3 + v7;
  ((uint2*)&v15)[0] = ROR16( ((uint2*)&v15)[0] ^ ((uint2*)&v3)[0] );
  v11 = v11 + v15;
  ((uint2*)&v7)[0] = ROR2( ((uint2*)&v7)[0] ^ ((uint2*)&v11)[0], 63U );
  v0 = v0 + v5 + m2;
  ((uint2*)&v15)[0] = SWAPUINT2( ((uint2*)&v15)[0] ^ ((uint2*)&v0)[0] );
  v10 = v10 + v15;
  ((uint2*)&v5)[0] = ROR24( ((uint2*)&v5)[0] ^ ((uint2*)&v10)[0] );
  v0 = v0 + v5 + m6;
  ((uint2*)&v15)[0] = ROR16( ((uint2*)&v15)[0] ^ ((uint2*)&v0)[0] );
  v10 = v10 + v15;
  ((uint2*)&v5)[0] = ROR2( ((uint2*)&v5)[0] ^ ((uint2*)&v10)[0], 63U );
  v1 = v1 + v6 + m5;
  ((uint2*)&v12)[0] = SWAPUINT2( ((uint2*)&v12)[0] ^ ((uint2*)&v1)[0] );
  v11 = v11 + v12;
  ((uint2*)&v6)[0] = ROR24( ((uint2*)&v6)[0] ^ ((uint2*)&v11)[0] );
  v1 = v1 + v6;
  ((uint2*)&v12)[0] = ROR16( ((uint2*)&v12)[0] ^ ((uint2*)&v1)[0] );
  v11 = v11 + v12;
  ((uint2*)&v6)[0] = ROR2( ((uint2*)&v6)[0] ^ ((uint2*)&v11)[0], 63U );
  v2 = v2 + v7 + m4;
  ((uint2*)&v13)[0] = SWAPUINT2( ((uint2*)&v13)[0] ^ ((uint2*)&v2)[0] );
  v8 = v8 + v13;
  ((uint2*)&v7)[0] = ROR24( ((uint2*)&v7)[0] ^ ((uint2*)&v8)[0] );
  v2 = v2 + v7 + m0;
  ((uint2*)&v13)[0] = ROR16( ((uint2*)&v13)[0] ^ ((uint2*)&v2)[0] );
  v8 = v8 + v13;
  ((uint2*)&v7)[0] = ROR2( ((uint2*)&v7)[0] ^ ((uint2*)&v8)[0], 63U );
  v3 = v3 + v4;
  ((uint2*)&v14)[0] = SWAPUINT2( ((uint2*)&v14)[0] ^ ((uint2*)&v3)[0] );
  v9 = v9 + v14;
  ((uint2*)&v4)[0] = ROR24( ((uint2*)&v4)[0] ^ ((uint2*)&v9)[0] );
  v3 = v3 + v4;
  ((uint2*)&v14)[0] = ROR16( ((uint2*)&v14)[0] ^ ((uint2*)&v3)[0] );
  v9 = v9 + v14;
  ((uint2*)&v4)[0] = ROR2( ((uint2*)&v4)[0] ^ ((uint2*)&v9)[0], 63U );
  // round 4
  v0 = v0 + v4;
  ((uint2*)&v12)[0] = SWAPUINT2( ((uint2*)&v12)[0] ^ ((uint2*)&v0)[0] );
  v8 = v8 + v12;
  ((uint2*)&v4)[0] = ROR24( ((uint2*)&v4)[0] ^ ((uint2*)&v8)[0] );
  v0 = v0 + v4 + m0;
  ((uint2*)&v12)[0] = ROR16( ((uint2*)&v12)[0] ^ ((uint2*)&v0)[0] );
  v8 = v8 + v12;
  ((uint2*)&v4)[0] = ROR2( ((uint2*)&v4)[0] ^ ((uint2*)&v8)[0], 63U );
  v1 = v1 + v5 + m5;
  ((uint2*)&v13)[0] = SWAPUINT2( ((uint2*)&v13)[0] ^ ((uint2*)&v1)[0] );
  v9 = v9 + v13;
  ((uint2*)&v5)[0] = ROR24( ((uint2*)&v5)[0] ^ ((uint2*)&v9)[0] );
  v1 = v1 + v5;
  ((uint2*)&v13)[0] = ROR16( ((uint2*)&v13)[0] ^ ((uint2*)&v1)[0] );
  v9 = v9 + v13;
  ((uint2*)&v5)[0] = ROR2( ((uint2*)&v5)[0] ^ ((uint2*)&v9)[0], 63U );
  v2 = v2 + v6 + m2;
  ((uint2*)&v14)[0] = SWAPUINT2( ((uint2*)&v14)[0] ^ ((uint2*)&v2)[0] );
  v10 = v10 + v14;
  ((uint2*)&v6)[0] = ROR24( ((uint2*)&v6)[0] ^ ((uint2*)&v10)[0] );
  v2 = v2 + v6 + m4;
  ((uint2*)&v14)[0] = ROR16( ((uint2*)&v14)[0] ^ ((uint2*)&v2)[0] );
  v10 = v10 + v14;
  ((uint2*)&v6)[0] = ROR2( ((uint2*)&v6)[0] ^ ((uint2*)&v10)[0], 63U );
  v3 = v3 + v7;
  ((uint2*)&v15)[0] = SWAPUINT2( ((uint2*)&v15)[0] ^ ((uint2*)&v3)[0] );
  v11 = v11 + v15;
  ((uint2*)&v7)[0] = ROR24( ((uint2*)&v7)[0] ^ ((uint2*)&v11)[0] );
  v3 = v3 + v7;
  ((uint2*)&v15)[0] = ROR16( ((uint2*)&v15)[0] ^ ((uint2*)&v3)[0] );
  v11 = v11 + v15;
  ((uint2*)&v7)[0] = ROR2( ((uint2*)&v7)[0] ^ ((uint2*)&v11)[0], 63U );
  v0 = v0 + v5;
  ((uint2*)&v15)[0] = SWAPUINT2( ((uint2*)&v15)[0] ^ ((uint2*)&v0)[0] );
  v10 = v10 + v15;
  ((uint2*)&v5)[0] = ROR24( ((uint2*)&v5)[0] ^ ((uint2*)&v10)[0] );
  v0 = v0 + v5 + m1;
  ((uint2*)&v15)[0] = ROR16( ((uint2*)&v15)[0] ^ ((uint2*)&v0)[0] );
  v10 = v10 + v15;
  ((uint2*)&v5)[0] = ROR2( ((uint2*)&v5)[0] ^ ((uint2*)&v10)[0], 63U );
  v1 = v1 + v6;
  ((uint2*)&v12)[0] = SWAPUINT2( ((uint2*)&v12)[0] ^ ((uint2*)&v1)[0] );
  v11 = v11 + v12;
  ((uint2*)&v6)[0] = ROR24( ((uint2*)&v6)[0] ^ ((uint2*)&v11)[0] );
  v1 = v1 + v6;
  ((uint2*)&v12)[0] = ROR16( ((uint2*)&v12)[0] ^ ((uint2*)&v1)[0] );
  v11 = v11 + v12;
  ((uint2*)&v6)[0] = ROR2( ((uint2*)&v6)[0] ^ ((uint2*)&v11)[0], 63U );
  v2 = v2 + v7 + m6;
  ((uint2*)&v13)[0] = SWAPUINT2( ((uint2*)&v13)[0] ^ ((uint2*)&v2)[0] );
  v8 = v8 + v13;
  ((uint2*)&v7)[0] = ROR24( ((uint2*)&v7)[0] ^ ((uint2*)&v8)[0] );
  v2 = v2 + v7;
  ((uint2*)&v13)[0] = ROR16( ((uint2*)&v13)[0] ^ ((uint2*)&v2)[0] );
  v8 = v8 + v13;
  ((uint2*)&v7)[0] = ROR2( ((uint2*)&v7)[0] ^ ((uint2*)&v8)[0], 63U );
  v3 = v3 + v4 + m3;
  ((uint2*)&v14)[0] = SWAPUINT2( ((uint2*)&v14)[0] ^ ((uint2*)&v3)[0] );
  v9 = v9 + v14;
  ((uint2*)&v4)[0] = ROR24( ((uint2*)&v4)[0] ^ ((uint2*)&v9)[0] );
  v3 = v3 + v4;
  ((uint2*)&v14)[0] = ROR16( ((uint2*)&v14)[0] ^ ((uint2*)&v3)[0] );
  v9 = v9 + v14;
  ((uint2*)&v4)[0] = ROR2( ((uint2*)&v4)[0] ^ ((uint2*)&v9)[0], 63U );
  // round 5
  v0 = v0 + v4 + m2;
  ((uint2*)&v12)[0] = SWAPUINT2( ((uint2*)&v12)[0] ^ ((uint2*)&v0)[0] );
  v8 = v8 + v12;
  ((uint2*)&v4)[0] = ROR24( ((uint2*)&v4)[0] ^ ((uint2*)&v8)[0] );
  v0 = v0 + v4;
  ((uint2*)&v12)[0] = ROR16( ((uint2*)&v12)[0] ^ ((uint2*)&v0)[0] );
  v8 = v8 + v12;
  ((uint2*)&v4)[0] = ROR2( ((uint2*)&v4)[0] ^ ((uint2*)&v8)[0], 63U );
  v1 = v1 + v5 + m6;
  ((uint2*)&v13)[0] = SWAPUINT2( ((uint2*)&v13)[0] ^ ((uint2*)&v1)[0] );
  v9 = v9 + v13;
  ((uint2*)&v5)[0] = ROR24( ((uint2*)&v5)[0] ^ ((uint2*)&v9)[0] );
  v1 = v1 + v5;
  ((uint2*)&v13)[0] = ROR16( ((uint2*)&v13)[0] ^ ((uint2*)&v1)[0] );
  v9 = v9 + v13;
  ((uint2*)&v5)[0] = ROR2( ((uint2*)&v5)[0] ^ ((uint2*)&v9)[0], 63U );
  v2 = v2 + v6 + m0;
  ((uint2*)&v14)[0] = SWAPUINT2( ((uint2*)&v14)[0] ^ ((uint2*)&v2)[0] );
  v10 = v10 + v14;
  ((uint2*)&v6)[0] = ROR24( ((uint2*)&v6)[0] ^ ((uint2*)&v10)[0] );
  v2 = v2 + v6;
  ((uint2*)&v14)[0] = ROR16( ((uint2*)&v14)[0] ^ ((uint2*)&v2)[0] );
  v10 = v10 + v14;
  ((uint2*)&v6)[0] = ROR2( ((uint2*)&v6)[0] ^ ((uint2*)&v10)[0], 63U );
  v3 = v3 + v7;
  ((uint2*)&v15)[0] = SWAPUINT2( ((uint2*)&v15)[0] ^ ((uint2*)&v3)[0] );
  v11 = v11 + v15;
  ((uint2*)&v7)[0] = ROR24( ((uint2*)&v7)[0] ^ ((uint2*)&v11)[0] );
  v3 = v3 + v7 + m3;
  ((uint2*)&v15)[0] = ROR16( ((uint2*)&v15)[0] ^ ((uint2*)&v3)[0] );
  v11 = v11 + v15;
  ((uint2*)&v7)[0] = ROR2( ((uint2*)&v7)[0] ^ ((uint2*)&v11)[0], 63U );
  v0 = v0 + v5 + m4;
  ((uint2*)&v15)[0] = SWAPUINT2( ((uint2*)&v15)[0] ^ ((uint2*)&v0)[0] );
  v10 = v10 + v15;
  ((uint2*)&v5)[0] = ROR24( ((uint2*)&v5)[0] ^ ((uint2*)&v10)[0] );
  v0 = v0 + v5;
  ((uint2*)&v15)[0] = ROR16( ((uint2*)&v15)[0] ^ ((uint2*)&v0)[0] );
  v10 = v10 + v15;
  ((uint2*)&v5)[0] = ROR2( ((uint2*)&v5)[0] ^ ((uint2*)&v10)[0], 63U );
  v1 = v1 + v6;
  ((uint2*)&v12)[0] = SWAPUINT2( ((uint2*)&v12)[0] ^ ((uint2*)&v1)[0] );
  v11 = v11 + v12;
  ((uint2*)&v6)[0] = ROR24( ((uint2*)&v6)[0] ^ ((uint2*)&v11)[0] );
  v1 = v1 + v6 + m5;
  ((uint2*)&v12)[0] = ROR16( ((uint2*)&v12)[0] ^ ((uint2*)&v1)[0] );
  v11 = v11 + v12;
  ((uint2*)&v6)[0] = ROR2( ((uint2*)&v6)[0] ^ ((uint2*)&v11)[0], 63U );
  v2 = v2 + v7;
  ((uint2*)&v13)[0] = SWAPUINT2( ((uint2*)&v13)[0] ^ ((uint2*)&v2)[0] );
  v8 = v8 + v13;
  ((uint2*)&v7)[0] = ROR24( ((uint2*)&v7)[0] ^ ((uint2*)&v8)[0] );
  v2 = v2 + v7;
  ((uint2*)&v13)[0] = ROR16( ((uint2*)&v13)[0] ^ ((uint2*)&v2)[0] );
  v8 = v8 + v13;
  ((uint2*)&v7)[0] = ROR2( ((uint2*)&v7)[0] ^ ((uint2*)&v8)[0], 63U );
  v3 = v3 + v4 + m1;
  ((uint2*)&v14)[0] = SWAPUINT2( ((uint2*)&v14)[0] ^ ((uint2*)&v3)[0] );
  v9 = v9 + v14;
  ((uint2*)&v4)[0] = ROR24( ((uint2*)&v4)[0] ^ ((uint2*)&v9)[0] );
  v3 = v3 + v4;
  ((uint2*)&v14)[0] = ROR16( ((uint2*)&v14)[0] ^ ((uint2*)&v3)[0] );
  v9 = v9 + v14;
  ((uint2*)&v4)[0] = ROR2( ((uint2*)&v4)[0] ^ ((uint2*)&v9)[0], 63U );
  // round 6
  v0 = v0 + v4;
  ((uint2*)&v12)[0] = SWAPUINT2( ((uint2*)&v12)[0] ^ ((uint2*)&v0)[0] );
  v8 = v8 + v12;
  ((uint2*)&v4)[0] = ROR24( ((uint2*)&v4)[0] ^ ((uint2*)&v8)[0] );
  v0 = v0 + v4 + m5;
  ((uint2*)&v12)[0] = ROR16( ((uint2*)&v12)[0] ^ ((uint2*)&v0)[0] );
  v8 = v8 + v12;
  ((uint2*)&v4)[0] = ROR2( ((uint2*)&v4)[0] ^ ((uint2*)&v8)[0], 63U );
  v1 = v1 + v5 + m1;
  ((uint2*)&v13)[0] = SWAPUINT2( ((uint2*)&v13)[0] ^ ((uint2*)&v1)[0] );
  v9 = v9 + v13;
  ((uint2*)&v5)[0] = ROR24( ((uint2*)&v5)[0] ^ ((uint2*)&v9)[0] );
  v1 = v1 + v5;
  ((uint2*)&v13)[0] = ROR16( ((uint2*)&v13)[0] ^ ((uint2*)&v1)[0] );
  v9 = v9 + v13;
  ((uint2*)&v5)[0] = ROR2( ((uint2*)&v5)[0] ^ ((uint2*)&v9)[0], 63U );
  v2 = v2 + v6;
  ((uint2*)&v14)[0] = SWAPUINT2( ((uint2*)&v14)[0] ^ ((uint2*)&v2)[0] );
  v10 = v10 + v14;
  ((uint2*)&v6)[0] = ROR24( ((uint2*)&v6)[0] ^ ((uint2*)&v10)[0] );
  v2 = v2 + v6;
  ((uint2*)&v14)[0] = ROR16( ((uint2*)&v14)[0] ^ ((uint2*)&v2)[0] );
  v10 = v10 + v14;
  ((uint2*)&v6)[0] = ROR2( ((uint2*)&v6)[0] ^ ((uint2*)&v10)[0], 63U );
  v3 = v3 + v7 + m4;
  ((uint2*)&v15)[0] = SWAPUINT2( ((uint2*)&v15)[0] ^ ((uint2*)&v3)[0] );
  v11 = v11 + v15;
  ((uint2*)&v7)[0] = ROR24( ((uint2*)&v7)[0] ^ ((uint2*)&v11)[0] );
  v3 = v3 + v7;
  ((uint2*)&v15)[0] = ROR16( ((uint2*)&v15)[0] ^ ((uint2*)&v3)[0] );
  v11 = v11 + v15;
  ((uint2*)&v7)[0] = ROR2( ((uint2*)&v7)[0] ^ ((uint2*)&v11)[0], 63U );
  v0 = v0 + v5 + m0;
  ((uint2*)&v15)[0] = SWAPUINT2( ((uint2*)&v15)[0] ^ ((uint2*)&v0)[0] );
  v10 = v10 + v15;
  ((uint2*)&v5)[0] = ROR24( ((uint2*)&v5)[0] ^ ((uint2*)&v10)[0] );
  v0 = v0 + v5;
  ((uint2*)&v15)[0] = ROR16( ((uint2*)&v15)[0] ^ ((uint2*)&v0)[0] );
  v10 = v10 + v15;
  ((uint2*)&v5)[0] = ROR2( ((uint2*)&v5)[0] ^ ((uint2*)&v10)[0], 63U );
  v1 = v1 + v6 + m6;
  ((uint2*)&v12)[0] = SWAPUINT2( ((uint2*)&v12)[0] ^ ((uint2*)&v1)[0] );
  v11 = v11 + v12;
  ((uint2*)&v6)[0] = ROR24( ((uint2*)&v6)[0] ^ ((uint2*)&v11)[0] );
  v1 = v1 + v6 + m3;
  ((uint2*)&v12)[0] = ROR16( ((uint2*)&v12)[0] ^ ((uint2*)&v1)[0] );
  v11 = v11 + v12;
  ((uint2*)&v6)[0] = ROR2( ((uint2*)&v6)[0] ^ ((uint2*)&v11)[0], 63U );
  v2 = v2 + v7;
  ((uint2*)&v13)[0] = SWAPUINT2( ((uint2*)&v13)[0] ^ ((uint2*)&v2)[0] );
  v8 = v8 + v13;
  ((uint2*)&v7)[0] = ROR24( ((uint2*)&v7)[0] ^ ((uint2*)&v8)[0] );
  v2 = v2 + v7 + m2;
  ((uint2*)&v13)[0] = ROR16( ((uint2*)&v13)[0] ^ ((uint2*)&v2)[0] );
  v8 = v8 + v13;
  ((uint2*)&v7)[0] = ROR2( ((uint2*)&v7)[0] ^ ((uint2*)&v8)[0], 63U );
  v3 = v3 + v4;
  ((uint2*)&v14)[0] = SWAPUINT2( ((uint2*)&v14)[0] ^ ((uint2*)&v3)[0] );
  v9 = v9 + v14;
  ((uint2*)&v4)[0] = ROR24( ((uint2*)&v4)[0] ^ ((uint2*)&v9)[0] );
  v3 = v3 + v4;
  ((uint2*)&v14)[0] = ROR16( ((uint2*)&v14)[0] ^ ((uint2*)&v3)[0] );
  v9 = v9 + v14;
  ((uint2*)&v4)[0] = ROR2( ((uint2*)&v4)[0] ^ ((uint2*)&v9)[0], 63U );
  // round 7
  v0 = v0 + v4;
  ((uint2*)&v12)[0] = SWAPUINT2( ((uint2*)&v12)[0] ^ ((uint2*)&v0)[0] );
  v8 = v8 + v12;
  ((uint2*)&v4)[0] = ROR24( ((uint2*)&v4)[0] ^ ((uint2*)&v8)[0] );
  v0 = v0 + v4;
  ((uint2*)&v12)[0] = ROR16( ((uint2*)&v12)[0] ^ ((uint2*)&v0)[0] );
  v8 = v8 + v12;
  ((uint2*)&v4)[0] = ROR2( ((uint2*)&v4)[0] ^ ((uint2*)&v8)[0], 63U );
  v1 = v1 + v5;
  ((uint2*)&v13)[0] = SWAPUINT2( ((uint2*)&v13)[0] ^ ((uint2*)&v1)[0] );
  v9 = v9 + v13;
  ((uint2*)&v5)[0] = ROR24( ((uint2*)&v5)[0] ^ ((uint2*)&v9)[0] );
  v1 = v1 + v5;
  ((uint2*)&v13)[0] = ROR16( ((uint2*)&v13)[0] ^ ((uint2*)&v1)[0] );
  v9 = v9 + v13;
  ((uint2*)&v5)[0] = ROR2( ((uint2*)&v5)[0] ^ ((uint2*)&v9)[0], 63U );
  v2 = v2 + v6;
  ((uint2*)&v14)[0] = SWAPUINT2( ((uint2*)&v14)[0] ^ ((uint2*)&v2)[0] );
  v10 = v10 + v14;
  ((uint2*)&v6)[0] = ROR24( ((uint2*)&v6)[0] ^ ((uint2*)&v10)[0] );
  v2 = v2 + v6 + m1;
  ((uint2*)&v14)[0] = ROR16( ((uint2*)&v14)[0] ^ ((uint2*)&v2)[0] );
  v10 = v10 + v14;
  ((uint2*)&v6)[0] = ROR2( ((uint2*)&v6)[0] ^ ((uint2*)&v10)[0], 63U );
  v3 = v3 + v7 + m3;
  ((uint2*)&v15)[0] = SWAPUINT2( ((uint2*)&v15)[0] ^ ((uint2*)&v3)[0] );
  v11 = v11 + v15;
  ((uint2*)&v7)[0] = ROR24( ((uint2*)&v7)[0] ^ ((uint2*)&v11)[0] );
  v3 = v3 + v7;
  ((uint2*)&v15)[0] = ROR16( ((uint2*)&v15)[0] ^ ((uint2*)&v3)[0] );
  v11 = v11 + v15;
  ((uint2*)&v7)[0] = ROR2( ((uint2*)&v7)[0] ^ ((uint2*)&v11)[0], 63U );
  v0 = v0 + v5 + m5;
  ((uint2*)&v15)[0] = SWAPUINT2( ((uint2*)&v15)[0] ^ ((uint2*)&v0)[0] );
  v10 = v10 + v15;
  ((uint2*)&v5)[0] = ROR24( ((uint2*)&v5)[0] ^ ((uint2*)&v10)[0] );
  v0 = v0 + v5 + m0;
  ((uint2*)&v15)[0] = ROR16( ((uint2*)&v15)[0] ^ ((uint2*)&v0)[0] );
  v10 = v10 + v15;
  ((uint2*)&v5)[0] = ROR2( ((uint2*)&v5)[0] ^ ((uint2*)&v10)[0], 63U );
  v1 = v1 + v6;
  ((uint2*)&v12)[0] = SWAPUINT2( ((uint2*)&v12)[0] ^ ((uint2*)&v1)[0] );
  v11 = v11 + v12;
  ((uint2*)&v6)[0] = ROR24( ((uint2*)&v6)[0] ^ ((uint2*)&v11)[0] );
  v1 = v1 + v6 + m4;
  ((uint2*)&v12)[0] = ROR16( ((uint2*)&v12)[0] ^ ((uint2*)&v1)[0] );
  v11 = v11 + v12;
  ((uint2*)&v6)[0] = ROR2( ((uint2*)&v6)[0] ^ ((uint2*)&v11)[0], 63U );
  v2 = v2 + v7;
  ((uint2*)&v13)[0] = SWAPUINT2( ((uint2*)&v13)[0] ^ ((uint2*)&v2)[0] );
  v8 = v8 + v13;
  ((uint2*)&v7)[0] = ROR24( ((uint2*)&v7)[0] ^ ((uint2*)&v8)[0] );
  v2 = v2 + v7 + m6;
  ((uint2*)&v13)[0] = ROR16( ((uint2*)&v13)[0] ^ ((uint2*)&v2)[0] );
  v8 = v8 + v13;
  ((uint2*)&v7)[0] = ROR2( ((uint2*)&v7)[0] ^ ((uint2*)&v8)[0], 63U );
  v3 = v3 + v4 + m2;
  ((uint2*)&v14)[0] = SWAPUINT2( ((uint2*)&v14)[0] ^ ((uint2*)&v3)[0] );
  v9 = v9 + v14;
  ((uint2*)&v4)[0] = ROR24( ((uint2*)&v4)[0] ^ ((uint2*)&v9)[0] );
  v3 = v3 + v4;
  ((uint2*)&v14)[0] = ROR16( ((uint2*)&v14)[0] ^ ((uint2*)&v3)[0] );
  v9 = v9 + v14;
  ((uint2*)&v4)[0] = ROR2( ((uint2*)&v4)[0] ^ ((uint2*)&v9)[0], 63U );
  // round 8
  v0 = v0 + v4 + m6;
  ((uint2*)&v12)[0] = SWAPUINT2( ((uint2*)&v12)[0] ^ ((uint2*)&v0)[0] );
  v8 = v8 + v12;
  ((uint2*)&v4)[0] = ROR24( ((uint2*)&v4)[0] ^ ((uint2*)&v8)[0] );
  v0 = v0 + v4;
  ((uint2*)&v12)[0] = ROR16( ((uint2*)&v12)[0] ^ ((uint2*)&v0)[0] );
  v8 = v8 + v12;
  ((uint2*)&v4)[0] = ROR2( ((uint2*)&v4)[0] ^ ((uint2*)&v8)[0], 63U );
  v1 = v1 + v5;
  ((uint2*)&v13)[0] = SWAPUINT2( ((uint2*)&v13)[0] ^ ((uint2*)&v1)[0] );
  v9 = v9 + v13;
  ((uint2*)&v5)[0] = ROR24( ((uint2*)&v5)[0] ^ ((uint2*)&v9)[0] );
  v1 = v1 + v5;
  ((uint2*)&v13)[0] = ROR16( ((uint2*)&v13)[0] ^ ((uint2*)&v1)[0] );
  v9 = v9 + v13;
  ((uint2*)&v5)[0] = ROR2( ((uint2*)&v5)[0] ^ ((uint2*)&v9)[0], 63U );
  v2 = v2 + v6;
  ((uint2*)&v14)[0] = SWAPUINT2( ((uint2*)&v14)[0] ^ ((uint2*)&v2)[0] );
  v10 = v10 + v14;
  ((uint2*)&v6)[0] = ROR24( ((uint2*)&v6)[0] ^ ((uint2*)&v10)[0] );
  v2 = v2 + v6 + m3;
  ((uint2*)&v14)[0] = ROR16( ((uint2*)&v14)[0] ^ ((uint2*)&v2)[0] );
  v10 = v10 + v14;
  ((uint2*)&v6)[0] = ROR2( ((uint2*)&v6)[0] ^ ((uint2*)&v10)[0], 63U );
  v3 = v3 + v7 + m0;
  ((uint2*)&v15)[0] = SWAPUINT2( ((uint2*)&v15)[0] ^ ((uint2*)&v3)[0] );
  v11 = v11 + v15;
  ((uint2*)&v7)[0] = ROR24( ((uint2*)&v7)[0] ^ ((uint2*)&v11)[0] );
  v3 = v3 + v7;
  ((uint2*)&v15)[0] = ROR16( ((uint2*)&v15)[0] ^ ((uint2*)&v3)[0] );
  v11 = v11 + v15;
  ((uint2*)&v7)[0] = ROR2( ((uint2*)&v7)[0] ^ ((uint2*)&v11)[0], 63U );
  v0 = v0 + v5;
  ((uint2*)&v15)[0] = SWAPUINT2( ((uint2*)&v15)[0] ^ ((uint2*)&v0)[0] );
  v10 = v10 + v15;
  ((uint2*)&v5)[0] = ROR24( ((uint2*)&v5)[0] ^ ((uint2*)&v10)[0] );
  v0 = v0 + v5 + m2;
  ((uint2*)&v15)[0] = ROR16( ((uint2*)&v15)[0] ^ ((uint2*)&v0)[0] );
  v10 = v10 + v15;
  ((uint2*)&v5)[0] = ROR2( ((uint2*)&v5)[0] ^ ((uint2*)&v10)[0], 63U );
  v1 = v1 + v6;
  ((uint2*)&v12)[0] = SWAPUINT2( ((uint2*)&v12)[0] ^ ((uint2*)&v1)[0] );
  v11 = v11 + v12;
  ((uint2*)&v6)[0] = ROR24( ((uint2*)&v6)[0] ^ ((uint2*)&v11)[0] );
  v1 = v1 + v6;
  ((uint2*)&v12)[0] = ROR16( ((uint2*)&v12)[0] ^ ((uint2*)&v1)[0] );
  v11 = v11 + v12;
  ((uint2*)&v6)[0] = ROR2( ((uint2*)&v6)[0] ^ ((uint2*)&v11)[0], 63U );
  v2 = v2 + v7 + m1;
  ((uint2*)&v13)[0] = SWAPUINT2( ((uint2*)&v13)[0] ^ ((uint2*)&v2)[0] );
  v8 = v8 + v13;
  ((uint2*)&v7)[0] = ROR24( ((uint2*)&v7)[0] ^ ((uint2*)&v8)[0] );
  v2 = v2 + v7 + m4;
  ((uint2*)&v13)[0] = ROR16( ((uint2*)&v13)[0] ^ ((uint2*)&v2)[0] );
  v8 = v8 + v13;
  ((uint2*)&v7)[0] = ROR2( ((uint2*)&v7)[0] ^ ((uint2*)&v8)[0], 63U );
  v3 = v3 + v4;
  ((uint2*)&v14)[0] = SWAPUINT2( ((uint2*)&v14)[0] ^ ((uint2*)&v3)[0] );
  v9 = v9 + v14;
  ((uint2*)&v4)[0] = ROR24( ((uint2*)&v4)[0] ^ ((uint2*)&v9)[0] );
  v3 = v3 + v4 + m5;
  ((uint2*)&v14)[0] = ROR16( ((uint2*)&v14)[0] ^ ((uint2*)&v3)[0] );
  v9 = v9 + v14;
  ((uint2*)&v4)[0] = ROR2( ((uint2*)&v4)[0] ^ ((uint2*)&v9)[0], 63U );
  // round 9
  v0 = v0 + v4;
  ((uint2*)&v12)[0] = SWAPUINT2( ((uint2*)&v12)[0] ^ ((uint2*)&v0)[0] );
  v8 = v8 + v12;
  ((uint2*)&v4)[0] = ROR24( ((uint2*)&v4)[0] ^ ((uint2*)&v8)[0] );
  v0 = v0 + v4 + m2;
  ((uint2*)&v12)[0] = ROR16( ((uint2*)&v12)[0] ^ ((uint2*)&v0)[0] );
  v8 = v8 + v12;
  ((uint2*)&v4)[0] = ROR2( ((uint2*)&v4)[0] ^ ((uint2*)&v8)[0], 63U );
  v1 = v1 + v5;
  ((uint2*)&v13)[0] = SWAPUINT2( ((uint2*)&v13)[0] ^ ((uint2*)&v1)[0] );
  v9 = v9 + v13;
  ((uint2*)&v5)[0] = ROR24( ((uint2*)&v5)[0] ^ ((uint2*)&v9)[0] );
  v1 = v1 + v5 + m4;
  ((uint2*)&v13)[0] = ROR16( ((uint2*)&v13)[0] ^ ((uint2*)&v1)[0] );
  v9 = v9 + v13;
  ((uint2*)&v5)[0] = ROR2( ((uint2*)&v5)[0] ^ ((uint2*)&v9)[0], 63U );
  v2 = v2 + v6;
  ((uint2*)&v14)[0] = SWAPUINT2( ((uint2*)&v14)[0] ^ ((uint2*)&v2)[0] );
  v10 = v10 + v14;
  ((uint2*)&v6)[0] = ROR24( ((uint2*)&v6)[0] ^ ((uint2*)&v10)[0] );
  v2 = v2 + v6 + m6;
  ((uint2*)&v14)[0] = ROR16( ((uint2*)&v14)[0] ^ ((uint2*)&v2)[0] );
  v10 = v10 + v14;
  ((uint2*)&v6)[0] = ROR2( ((uint2*)&v6)[0] ^ ((uint2*)&v10)[0], 63U );
  v3 = v3 + v7 + m1;
  ((uint2*)&v15)[0] = SWAPUINT2( ((uint2*)&v15)[0] ^ ((uint2*)&v3)[0] );
  v11 = v11 + v15;
  ((uint2*)&v7)[0] = ROR24( ((uint2*)&v7)[0] ^ ((uint2*)&v11)[0] );
  v3 = v3 + v7 + m5;
  ((uint2*)&v15)[0] = ROR16( ((uint2*)&v15)[0] ^ ((uint2*)&v3)[0] );
  v11 = v11 + v15;
  ((uint2*)&v7)[0] = ROR2( ((uint2*)&v7)[0] ^ ((uint2*)&v11)[0], 63U );
  v0 = v0 + v5;
  ((uint2*)&v15)[0] = SWAPUINT2( ((uint2*)&v15)[0] ^ ((uint2*)&v0)[0] );
  v10 = v10 + v15;
  ((uint2*)&v5)[0] = ROR24( ((uint2*)&v5)[0] ^ ((uint2*)&v10)[0] );
  v0 = v0 + v5;
  ((uint2*)&v15)[0] = ROR16( ((uint2*)&v15)[0] ^ ((uint2*)&v0)[0] );
  v10 = v10 + v15;
  ((uint2*)&v5)[0] = ROR2( ((uint2*)&v5)[0] ^ ((uint2*)&v10)[0], 63U );
  v1 = v1 + v6;
  ((uint2*)&v12)[0] = SWAPUINT2( ((uint2*)&v12)[0] ^ ((uint2*)&v1)[0] );
  v11 = v11 + v12;
  ((uint2*)&v6)[0] = ROR24( ((uint2*)&v6)[0] ^ ((uint2*)&v11)[0] );
  v1 = v1 + v6;
  ((uint2*)&v12)[0] = ROR16( ((uint2*)&v12)[0] ^ ((uint2*)&v1)[0] );
  v11 = v11 + v12;
  ((uint2*)&v6)[0] = ROR2( ((uint2*)&v6)[0] ^ ((uint2*)&v11)[0], 63U );
  v2 = v2 + v7 + m3;
  ((uint2*)&v13)[0] = SWAPUINT2( ((uint2*)&v13)[0] ^ ((uint2*)&v2)[0] );
  v8 = v8 + v13;
  ((uint2*)&v7)[0] = ROR24( ((uint2*)&v7)[0] ^ ((uint2*)&v8)[0] );
  v2 = v2 + v7;
  ((uint2*)&v13)[0] = ROR16( ((uint2*)&v13)[0] ^ ((uint2*)&v2)[0] );
  v8 = v8 + v13;
  ((uint2*)&v7)[0] = ROR2( ((uint2*)&v7)[0] ^ ((uint2*)&v8)[0], 63U );
  v3 = v3 + v4;
  ((uint2*)&v14)[0] = SWAPUINT2( ((uint2*)&v14)[0] ^ ((uint2*)&v3)[0] );
  v9 = v9 + v14;
  ((uint2*)&v4)[0] = ROR24( ((uint2*)&v4)[0] ^ ((uint2*)&v9)[0] );
  v3 = v3 + v4 + m0;
  ((uint2*)&v14)[0] = ROR16( ((uint2*)&v14)[0] ^ ((uint2*)&v3)[0] );
  v9 = v9 + v14;
  ((uint2*)&v4)[0] = ROR2( ((uint2*)&v4)[0] ^ ((uint2*)&v9)[0], 63U );
  // round 10
  v0 = v0 + v4 + m0;
  ((uint2*)&v12)[0] = SWAPUINT2( ((uint2*)&v12)[0] ^ ((uint2*)&v0)[0] );
  v8 = v8 + v12;
  ((uint2*)&v4)[0] = ROR24( ((uint2*)&v4)[0] ^ ((uint2*)&v8)[0] );
  v0 = v0 + v4 + m1;
  ((uint2*)&v12)[0] = ROR16( ((uint2*)&v12)[0] ^ ((uint2*)&v0)[0] );
  v8 = v8 + v12;
  ((uint2*)&v4)[0] = ROR2( ((uint2*)&v4)[0] ^ ((uint2*)&v8)[0], 63U );
  v1 = v1 + v5 + m2;
  ((uint2*)&v13)[0] = SWAPUINT2( ((uint2*)&v13)[0] ^ ((uint2*)&v1)[0] );
  v9 = v9 + v13;
  ((uint2*)&v5)[0] = ROR24( ((uint2*)&v5)[0] ^ ((uint2*)&v9)[0] );
  v1 = v1 + v5 + m3;
  ((uint2*)&v13)[0] = ROR16( ((uint2*)&v13)[0] ^ ((uint2*)&v1)[0] );
  v9 = v9 + v13;
  ((uint2*)&v5)[0] = ROR2( ((uint2*)&v5)[0] ^ ((uint2*)&v9)[0], 63U );
  v2 = v2 + v6 + m4;
  ((uint2*)&v14)[0] = SWAPUINT2( ((uint2*)&v14)[0] ^ ((uint2*)&v2)[0] );
  v10 = v10 + v14;
  ((uint2*)&v6)[0] = ROR24( ((uint2*)&v6)[0] ^ ((uint2*)&v10)[0] );
  v2 = v2 + v6 + m5;
  ((uint2*)&v14)[0] = ROR16( ((uint2*)&v14)[0] ^ ((uint2*)&v2)[0] );
  v10 = v10 + v14;
  ((uint2*)&v6)[0] = ROR2( ((uint2*)&v6)[0] ^ ((uint2*)&v10)[0], 63U );
  v3 = v3 + v7 + m6;
  ((uint2*)&v15)[0] = SWAPUINT2( ((uint2*)&v15)[0] ^ ((uint2*)&v3)[0] );
  v11 = v11 + v15;
  ((uint2*)&v7)[0] = ROR24( ((uint2*)&v7)[0] ^ ((uint2*)&v11)[0] );
  v3 = v3 + v7;
  ((uint2*)&v15)[0] = ROR16( ((uint2*)&v15)[0] ^ ((uint2*)&v3)[0] );
  v11 = v11 + v15;
  ((uint2*)&v7)[0] = ROR2( ((uint2*)&v7)[0] ^ ((uint2*)&v11)[0], 63U );
  v0 = v0 + v5;
  ((uint2*)&v15)[0] = SWAPUINT2( ((uint2*)&v15)[0] ^ ((uint2*)&v0)[0] );
  v10 = v10 + v15;
  ((uint2*)&v5)[0] = ROR24( ((uint2*)&v5)[0] ^ ((uint2*)&v10)[0] );
  v0 = v0 + v5;
  ((uint2*)&v15)[0] = ROR16( ((uint2*)&v15)[0] ^ ((uint2*)&v0)[0] );
  v10 = v10 + v15;
  ((uint2*)&v5)[0] = ROR2( ((uint2*)&v5)[0] ^ ((uint2*)&v10)[0], 63U );
  v1 = v1 + v6;
  ((uint2*)&v12)[0] = SWAPUINT2( ((uint2*)&v12)[0] ^ ((uint2*)&v1)[0] );
  v11 = v11 + v12;
  ((uint2*)&v6)[0] = ROR24( ((uint2*)&v6)[0] ^ ((uint2*)&v11)[0] );
  v1 = v1 + v6;
  ((uint2*)&v12)[0] = ROR16( ((uint2*)&v12)[0] ^ ((uint2*)&v1)[0] );
  v11 = v11 + v12;
  ((uint2*)&v6)[0] = ROR2( ((uint2*)&v6)[0] ^ ((uint2*)&v11)[0], 63U );
  v2 = v2 + v7;
  ((uint2*)&v13)[0] = SWAPUINT2( ((uint2*)&v13)[0] ^ ((uint2*)&v2)[0] );
  v8 = v8 + v13;
  ((uint2*)&v7)[0] = ROR24( ((uint2*)&v7)[0] ^ ((uint2*)&v8)[0] );
  v2 = v2 + v7;
  ((uint2*)&v13)[0] = ROR16( ((uint2*)&v13)[0] ^ ((uint2*)&v2)[0] );
  v8 = v8 + v13;
  ((uint2*)&v7)[0] = ROR2( ((uint2*)&v7)[0] ^ ((uint2*)&v8)[0], 63U );
  v3 = v3 + v4;
  ((uint2*)&v14)[0] = SWAPUINT2( ((uint2*)&v14)[0] ^ ((uint2*)&v3)[0] );
  v9 = v9 + v14;
  ((uint2*)&v4)[0] = ROR24( ((uint2*)&v4)[0] ^ ((uint2*)&v9)[0] );
  v3 = v3 + v4;
  ((uint2*)&v14)[0] = ROR16( ((uint2*)&v14)[0] ^ ((uint2*)&v3)[0] );
  v9 = v9 + v14;
  ((uint2*)&v4)[0] = ROR2( ((uint2*)&v4)[0] ^ ((uint2*)&v9)[0], 63U );
  // round 11
  v0 = v0 + v4;
  ((uint2*)&v12)[0] = SWAPUINT2( ((uint2*)&v12)[0] ^ ((uint2*)&v0)[0] );
  v8 = v8 + v12;
  ((uint2*)&v4)[0] = ROR24( ((uint2*)&v4)[0] ^ ((uint2*)&v8)[0] );
  v0 = v0 + v4;
  ((uint2*)&v12)[0] = ROR16( ((uint2*)&v12)[0] ^ ((uint2*)&v0)[0] );
  v8 = v8 + v12;
  ((uint2*)&v4)[0] = ROR2( ((uint2*)&v4)[0] ^ ((uint2*)&v8)[0], 63U );
  v1 = v1 + v5 + m4;
  ((uint2*)&v13)[0] = SWAPUINT2( ((uint2*)&v13)[0] ^ ((uint2*)&v1)[0] );
  v9 = v9 + v13;
  ((uint2*)&v5)[0] = ROR24( ((uint2*)&v5)[0] ^ ((uint2*)&v9)[0] );
  v1 = v1 + v5;
  ((uint2*)&v13)[0] = ROR16( ((uint2*)&v13)[0] ^ ((uint2*)&v1)[0] );
  v9 = v9 + v13;
  ((uint2*)&v5)[0] = ROR2( ((uint2*)&v5)[0] ^ ((uint2*)&v9)[0], 63U );
  v2 = v2 + v6;
  ((uint2*)&v14)[0] = SWAPUINT2( ((uint2*)&v14)[0] ^ ((uint2*)&v2)[0] );
  v10 = v10 + v14;
  ((uint2*)&v6)[0] = ROR24( ((uint2*)&v6)[0] ^ ((uint2*)&v10)[0] );
  v2 = v2 + v6;
  ((uint2*)&v14)[0] = ROR16( ((uint2*)&v14)[0] ^ ((uint2*)&v2)[0] );
  v10 = v10 + v14;
  ((uint2*)&v6)[0] = ROR2( ((uint2*)&v6)[0] ^ ((uint2*)&v10)[0], 63U );
  v3 = v3 + v7;
  ((uint2*)&v15)[0] = SWAPUINT2( ((uint2*)&v15)[0] ^ ((uint2*)&v3)[0] );
  v11 = v11 + v15;
  ((uint2*)&v7)[0] = ROR24( ((uint2*)&v7)[0] ^ ((uint2*)&v11)[0] );
  v3 = v3 + v7 + m6;
  ((uint2*)&v15)[0] = ROR16( ((uint2*)&v15)[0] ^ ((uint2*)&v3)[0] );
  v11 = v11 + v15;
  ((uint2*)&v7)[0] = ROR2( ((uint2*)&v7)[0] ^ ((uint2*)&v11)[0], 63U );
  v0 = v0 + v5 + m1;
  ((uint2*)&v15)[0] = SWAPUINT2( ((uint2*)&v15)[0] ^ ((uint2*)&v0)[0] );
  v10 = v10 + v15;
  ((uint2*)&v5)[0] = ROR24( ((uint2*)&v5)[0] ^ ((uint2*)&v10)[0] );
  v0 = v0 + v5;
  ((uint2*)&v15)[0] = ROR16( ((uint2*)&v15)[0] ^ ((uint2*)&v0)[0] );
  v10 = v10 + v15;
  ((uint2*)&v5)[0] = ROR2( ((uint2*)&v5)[0] ^ ((uint2*)&v10)[0], 63U );
  v1 = v1 + v6 + m0;
  ((uint2*)&v12)[0] = SWAPUINT2( ((uint2*)&v12)[0] ^ ((uint2*)&v1)[0] );
  v11 = v11 + v12;
  ((uint2*)&v6)[0] = ROR24( ((uint2*)&v6)[0] ^ ((uint2*)&v11)[0] );
  v1 = v1 + v6 + m2;
  ((uint2*)&v12)[0] = ROR16( ((uint2*)&v12)[0] ^ ((uint2*)&v1)[0] );
  v11 = v11 + v12;
  ((uint2*)&v6)[0] = ROR2( ((uint2*)&v6)[0] ^ ((uint2*)&v11)[0], 63U );
  v2 = v2 + v7;
  ((uint2*)&v13)[0] = SWAPUINT2( ((uint2*)&v13)[0] ^ ((uint2*)&v2)[0] );
  v8 = v8 + v13;
  ((uint2*)&v7)[0] = ROR24( ((uint2*)&v7)[0] ^ ((uint2*)&v8)[0] );
  v2 = v2 + v7;
  ((uint2*)&v13)[0] = ROR16( ((uint2*)&v13)[0] ^ ((uint2*)&v2)[0] );
  v8 = v8 + v13;
  ((uint2*)&v7)[0] = ROR2( ((uint2*)&v7)[0] ^ ((uint2*)&v8)[0], 63U );
  v3 = v3 + v4 + m5;
  ((uint2*)&v14)[0] = SWAPUINT2( ((uint2*)&v14)[0] ^ ((uint2*)&v3)[0] );
  v9 = v9 + v14;
  ((uint2*)&v4)[0] = ROR24( ((uint2*)&v4)[0] ^ ((uint2*)&v9)[0] );
  v3 = v3 + v4 + m3;
  ((uint2*)&v14)[0] = ROR16( ((uint2*)&v14)[0] ^ ((uint2*)&v3)[0] );
  v9 = v9 + v14;
  ((uint2*)&v4)[0] = ROR2( ((uint2*)&v4)[0] ^ ((uint2*)&v9)[0], 63U );

  u64 *out = (u64 *)hash;
  out[0] = pre->h[0] ^ v0 ^ v8;
  out[1] = pre->h[1] ^ v1 ^ v9;
  out[2] = pre->h[2] ^ v2 ^ v10;
  out[3] = pre->h[3] ^ v3 ^ v11;
  out[4] = pre->h[4] ^ v4 ^ v12;
  out[5] = pre->h[5] ^ v5 ^ v13;
  const u64 tail = pre->h[6] ^ v6 ^ v14;
  memcpy(hash + 48, &tail, 6);
}

// One-shot variant for callers outside the hot loop.
__device__ void blake2b_gpu_hash(const blake2b_state *state, u32 idx, uchar *hash) {
  blake2b_pre pre;
  blake2b_precompute(state, &pre);
  blake2b_gpu_hash_pre(&pre, idx, hash);
}
