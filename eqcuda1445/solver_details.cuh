// Equihash CUDA solver
// Copyright (c) 2016 John Tromp
// Copyright (c) 2018 The ExchangeCoin team

#pragma once
#include "blake/blake2.h"
#include "blake2b.cuh"
#include "portable_endian.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifdef __APPLE__
#include "osx_barrier.h"
#endif

#if !defined(_WIN16) && !defined(_WIN32) && !defined(_WIN64) && !defined(__WINDOWS__)
#pragma clang diagnostic push
#pragma ide diagnostic ignored "OCUnusedGlobalDeclarationInspection"
#endif

#define RESTBITS 4
// 2_log of number of buckets
#define BUCKBITS (DIGITBITS - RESTBITS)

#ifndef SAVEMEM
#if RESTBITS == 4
// can't save memory in such small buckets
#define SAVEMEM 1
#elif RESTBITS >= 8
// take advantage of law of large numbers (sum of 2^8 random numbers)
// this reduces (200,9) memory to under 144MB, with negligible discarding
#define SAVEMEM 9 / 14
#endif
#endif

// number of buckets
static const u32 NBUCKETS = 1 << BUCKBITS;
// 2_log of number of slots per bucket
static const u32 SLOTBITS = RESTBITS + 1 + 1;
static const u32 SLOTRANGE = 1 << SLOTBITS;
// number of slots per bucket
static const u32 NSLOTS = SLOTRANGE * SAVEMEM;
// SLOTBITS mask
static const u32 SLOTMASK = SLOTRANGE - 1;
// number of possible values of xhash (rest of n) bits
static const u32 NRESTS = 1 << RESTBITS;
// RESTBITS mask
static const u32 RESTMASK = NRESTS - 1;
// number of blocks of hashes extracted from single 512 bit blake2b output
static const u32 NBLOCKS = (NHASHES + HASHESPERBLAKE - 1) / HASHESPERBLAKE;

// tree node identifying its children as two different slots in
// a bucket on previous layer with the same rest bits (x-tra hash)
struct tree {
    u32 bid_s0_s1_x; // manual bitfields

    __device__ tree(const u32 idx, const u32 xh) { bid_s0_s1_x = idx << RESTBITS | xh; }
    __device__ tree(const u32 idx) { bid_s0_s1_x = idx; }
#ifdef XINTREE
    __device__ tree(const u32 bid, const u32 s0, const u32 s1, const u32 xh) {
        bid_s0_s1_x = ((((bid << SLOTBITS) | s0) << SLOTBITS) | s1) << RESTBITS | xh;
#else
    __device__ tree(const u32 bid, const u32 s0, const u32 s1) {
        bid_s0_s1_x = (((bid << SLOTBITS) | s0) << SLOTBITS) | s1;
#endif
    }
    __device__ u32 getindex() const {
#ifdef XINTREE
        return bid_s0_s1_x >> RESTBITS;
#else
        return bid_s0_s1_x;
#endif
    }
    __device__ u32 bucketid() const {
#ifdef XINTREE
        return bid_s0_s1_x >> (2 * SLOTBITS + RESTBITS);
#else
        return bid_s0_s1_x >> (2 * SLOTBITS);
#endif
    }
    __device__ u32 slotid0() const {
#ifdef XINTREE
        return (bid_s0_s1_x >> SLOTBITS + RESTBITS) & SLOTMASK;
#else
        return (bid_s0_s1_x >> SLOTBITS) & SLOTMASK;
#endif
    }
    __device__ u32 slotid1() const {
#ifdef XINTREE
        return (bid_s0_s1_x >> RESTBITS) & SLOTMASK;
#else
        return bid_s0_s1_x & SLOTMASK;
#endif
    }
    __device__ u32 xhash() const { return bid_s0_s1_x & RESTMASK; }
    __device__ bool prob_disjoint(const tree other) const {
        tree xort(bid_s0_s1_x ^ other.bid_s0_s1_x);
        return xort.bucketid() || (xort.slotid0() && xort.slotid1());
        // next two tests catch much fewer cases and are therefore skipped
        // && slotid0() != other.slotid1() && slotid1() != other.slotid0()
    }
};

union hashunit {
    u32 word;
    uchar bytes[sizeof(u32)];
};

#define WORDS(bits) ((bits + 31) / 32)
#define HASHWORDS0 WORDS(WN - DIGITBITS + RESTBITS)
#define HASHWORDS1 WORDS(WN - 2 * DIGITBITS + RESTBITS)

struct slot0 {
    tree attr;
    hashunit hash[HASHWORDS0];
};

struct slot1 {
    tree attr;
    hashunit hash[HASHWORDS1];
};

// a bucket is NSLOTS treenodes
typedef slot0 bucket0[NSLOTS];
typedef slot1 bucket1[NSLOTS];
// the N-bit hash consists of K+1 n-bit "digits"
// each of which corresponds to a layer of NBUCKETS buckets
typedef bucket0 digit0[NBUCKETS];
typedef bucket1 digit1[NBUCKETS];

#define checkCudaErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

static void cuda_init() {
	checkCudaErrors(cudaSetDeviceFlags(cudaDeviceScheduleYield));
}

static void setperson(blake2b_state *ctx) {
    blake2b_param P;
    memset(&P, 0, sizeof(blake2b_param));

    P.fanout        = 1;
    P.depth         = 1;
    P.digest_length = (512 / WN) * WN / 8;

    memcpy(P.personal, "ZcashPoW", 8);
    *(uint32_t *)(P.personal + 8)  = htole32(WN);
    *(uint32_t *)(P.personal + 12) = htole32(WK);

    blake2b_init_param(ctx, &P);
}

static void setheader(blake2b_state *ctx, const uint8_t *input, u32 input_len, u32 nonce) {
    uint8_t localInput[180];
    memcpy(localInput, input, input_len);
    nonce = htole32(nonce);
    memcpy(localInput + 140, &nonce, sizeof(nonce));

    blake2b_update(ctx, localInput, input_len);
}

static void genhash(const blake2b_state *ctx, u32 idx, uchar *hash) {
    blake2b_state state = *ctx;
    u32 leb = htole32(idx / HASHESPERBLAKE);
    blake2b_update(&state, (uchar *)&leb, sizeof(u32));
    uchar blakehash[HASHOUT];
    blake2b_final(&state, blakehash, HASHOUT);
    memcpy(hash, blakehash + (idx % HASHESPERBLAKE) * WN / 8, WN / 8);
}

static verify_code verifyrec(const blake2b_state *ctx, const proof indices, uchar *hash, int r) {
    if (r == 0) {
        genhash(ctx, *indices, hash);
        return verify_code::POW_OK;
    }

    const u32 *indices1 = indices + u32(1 << (r - 1));
    if (*indices >= *indices1)
        return verify_code::POW_OUT_OF_ORDER;

    uchar hash0[WN / 8], hash1[WN / 8];
    verify_code vrf0 = verifyrec(ctx, indices, hash0, r - 1);
    if (vrf0 != verify_code::POW_OK)
        return vrf0;

    verify_code vrf1 = verifyrec(ctx, indices1, hash1, r - 1);
    if (vrf1 != verify_code::POW_OK)
        return vrf1;

    for (int i = 0; i < WN / 8; i++)
        hash[i] = hash0[i] ^ hash1[i];

    int i, b = r < WK ? r * DIGITBITS : WN;
    for (i = 0; i < b / 8; i++)
        if (hash[i])
            return verify_code::POW_NONZERO_XOR;

    if ((b % 8) && hash[i] >> (8 - (b % 8)))
        return verify_code::POW_NONZERO_XOR;

    return verify_code::POW_OK;
}

static int compu32(const void *pa, const void *pb) {
    u32 a = *(u32 *)pa, b = *(u32 *)pb;
    return a < b ? -1 : a == b ? 0 : +1;
}

static bool duped(const proof prf) {
    proof sortprf;
    memcpy(sortprf, prf, sizeof(proof));
    qsort(sortprf, PROOFSIZE, sizeof(u32), &compu32);
    for (u32 i = 1; i < PROOFSIZE; i++)
        if (sortprf[i] <= sortprf[i - 1])
            return true;

    return false;
}

static std::string to_hex(const uchar *data, u64 len) {
    static const char hexmap[] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};

    std::string s(len * 2, ' ');
    for (u64 i = 0; i < len; ++i) {
        s[2 * i] = hexmap[(data[i] & 0xF0) >> 4];
        s[2 * i + 1] = hexmap[data[i] & 0x0F];
    }
    return s;
}

static void compress_solution(const proof sol, cproof output) {
    uchar b;

    for (u32 i = 0, j = 0, bits_left = DIGITBITS + 1;
         j < COMPRESSED_SOL_SIZE; output[j++] = b) {
        if (bits_left >=8) {
            // Read next 8 bits, stay at same sol index
            b = sol[i] >> (bits_left -= 8);
        } else { // less than 8 bits to read
            // Read remaining bits and shift left to make space for next sol index
            b = sol[i];
            b <<= (8 - bits_left); // may also set b=0 if bits_left was 0, which is fine
            // Go to next sol index and read remaining bits
            bits_left += DIGITBITS + 1 - 8;
            b |= sol[++i] >> bits_left;
        }
    }
}

static u32 array_to_index(const uchar *array) {
    u32 bei;
    memcpy(&bei, array, sizeof(bei));
    return be32toh(bei);
}

static void expand_array(const uchar *in, u64 in_len, uchar *out, u64 out_len, u64 bit_len, u64 byte_pad) {
    assert(bit_len >= 8);
    assert(8 * sizeof(u32) >= 7 + bit_len);

    u64 out_width{(bit_len + 7) / 8 + byte_pad};
    assert(out_len == 8 * out_width * in_len / bit_len);

    u32 bit_len_mask{((u32)1 << bit_len) - 1};

    // The acc_bits least-significant bits of acc_value represent a bit sequence
    // in big-endian order.
    u64 acc_bits  = 0;
    u32 acc_value = 0;

    u64 j = 0;
    for (u64 i = 0; i < in_len; i++) {
        acc_value = (acc_value << 8) | in[i];
        acc_bits += 8;

        // When we have bit_len or more bits in the accumulator, write the next
        // output element.
        if (acc_bits >= bit_len) {
            acc_bits -= bit_len;
            for (u64 x = 0; x < byte_pad; x++) {
                out[j + x] = 0;
            }
            for (u64 x = byte_pad; x < out_width; x++) {
                out[j + x] = (
                                     // Big-endian
                                     acc_value >> (acc_bits + (8 * (out_width - x - 1)))) &
                             (
                                     // Apply bit_len_mask across byte boundaries
                                     (bit_len_mask >> (8 * (out_width - x - 1))) & 0xFF);
            }
            j += out_width;
        }
    }
}

static void uncompress_solution(const cproof sol, proof output) {
    const u64 collision_bit_length = WN / (WK + 1);
    const u64 solution_width       = (1 << WK) * (collision_bit_length + 1) / 8;

    assert(((collision_bit_length + 1) + 7) / 8 <= sizeof(u32));

    const u64 len_indices{8 * sizeof(u32) * solution_width / (collision_bit_length + 1)};
    const u64 byte_pad{sizeof(u32) - ((collision_bit_length + 1) + 7) / 8};

    uchar array[len_indices];
    expand_array(sol, solution_width, array, len_indices, collision_bit_length + 1, byte_pad);

    for (u64 i = 0, j = 0; i < len_indices; i += sizeof(u32), ++j) {
        output[j] = array_to_index(array + i);
    }
}

// size (in bytes) of hash in round 0 <= r < WK
static u32 hhashsize(const u32 r) {
#ifdef XINTREE
    const u32 hashbits = WN - (r + 1) * DIGITBITS;
#else
    const u32 hashbits = WN - (r + 1) * DIGITBITS + RESTBITS;
#endif
    return (hashbits + 7) / 8;
}

// size (in bytes) of hash in round 0 <= r < WK
static __device__ u32 hashsize(const u32 r) {
#ifdef XINTREE
    const u32 hashbits = WN - (r + 1) * DIGITBITS;
#else
    const u32 hashbits = WN - (r + 1) * DIGITBITS + RESTBITS;
#endif
    return (hashbits + 7) / 8;
}

static u32 hhashwords(u32 bytes) {
	return (bytes + 3) / 4;
}

static __device__ u32 hashwords(u32 bytes) {
	return (bytes + 3) / 4;
}

// manages hash and tree data
struct htalloc {
    bucket0 *trees0[(WK + 1) / 2];
    bucket1 *trees1[WK / 2];
};

typedef u32 bsizes[NBUCKETS];

struct equi {
    blake2b_state blake_ctx;
    htalloc hta;
    bsizes *nslots;
    proof *sols;
    u32 nsols;
    u32 nthreads;
    equi(const u32 n_threads) { nthreads = n_threads; }
    void setstate(const uint8_t *input, u32 input_len, u32 nonce) {
        setperson(&blake_ctx);
        setheader(&blake_ctx, input, input_len, nonce);
        nsols = 0;
    }
    __device__ u32 getnslots0(const u32 bid) {
        u32 &nslot = nslots[0][bid];
        const u32 n = min(nslot, NSLOTS);
        nslot = 0;
        return n;
    }
    __device__ u32 getnslots1(const u32 bid) {
        u32 &nslot = nslots[1][bid];
        const u32 n = min(nslot, NSLOTS);
        nslot = 0;
        return n;
    }
    __device__ bool orderindices(u32 *indices, u32 size) {
        if (indices[0] > indices[size]) {
            for (u32 i = 0; i < size; i++) {
                const u32 tmp = indices[i];
                indices[i] = indices[size + i];
                indices[size + i] = tmp;
            }
        }
        return false;
    }
    // Per-layer slot strides in u32 words (attr + significant hash words).
    // heap0 layers 0,1,2 are written by rounds 0,2,4; heap1 layers 0,1 by
    // rounds 1,3. Later rounds carry fewer hash words, so slots shrink.
    // Layers share 5-word cells (trees pointers are offset by l words); a
    // layer's slot occupies words [0 .. hashwords] of its shifted cell, so
    // later (smaller) layers never clobber earlier layers' attrs.
    __device__ __forceinline__ u32 attr0(u32 l, u32 bid, u32 s) {
        return ((const u32 *)hta.trees0[l])[(bid * NSLOTS + s) * 5];
    }
    __device__ __forceinline__ u32 attr1(u32 l, u32 bid, u32 s) {
        return ((const u32 *)hta.trees1[l])[(bid * NSLOTS + s) * 5];
    }
    __device__ bool listindices1(const tree t, u32 *indices) {
        const u32 size = 1 << 0;
        indices[0] = tree(attr0(0, t.bucketid(), t.slotid0())).getindex();
        indices[size] = tree(attr0(0, t.bucketid(), t.slotid1())).getindex();
        orderindices(indices, size);
        return false;
    }
    __device__ bool listindices2(const tree t, u32 *indices) {
        const u32 size = 1 << 1;
        return listindices1(tree(attr1(0, t.bucketid(), t.slotid0())), indices) ||
               listindices1(tree(attr1(0, t.bucketid(), t.slotid1())), indices + size) ||
               orderindices(indices, size) || indices[0] == indices[size];
    }
    __device__ bool listindices3(const tree t, u32 *indices) {
        const u32 size = 1 << 2;
        return listindices2(tree(attr0(1, t.bucketid(), t.slotid0())), indices) ||
               listindices2(tree(attr0(1, t.bucketid(), t.slotid1())), indices + size) ||
               orderindices(indices, size) || indices[0] == indices[size];
    }
    __device__ bool listindices4(const tree t, u32 *indices) {
        const u32 size = 1 << 3;
        return listindices3(tree(attr1(1, t.bucketid(), t.slotid0())), indices) ||
               listindices3(tree(attr1(1, t.bucketid(), t.slotid1())), indices + size) ||
               orderindices(indices, size) || indices[0] == indices[size];
    }
    __device__ bool listindices5(const tree t, u32 *indices) {
        const u32 size = 1 << 4;
        return listindices4(tree(attr0(2, t.bucketid(), t.slotid0())), indices) ||
               listindices4(tree(attr0(2, t.bucketid(), t.slotid1())), indices + size) ||
               orderindices(indices, size) || indices[0] == indices[size];
    }

#if WK == 9
    __device__ bool listindices6(const tree t, u32 *indices) {
        const bucket1 &buck = hta.trees1[2][t.bucketid()];
        const u32 size = 1 << 5;
        return listindices5(buck[t.slotid0()].attr, indices) || listindices5(buck[t.slotid1()].attr, indices + size) ||
               orderindices(indices, size) || indices[0] == indices[size];
    }
    __device__ bool listindices7(const tree t, u32 *indices) {
        const bucket0 &buck = hta.trees0[3][t.bucketid()];
        const u32 size = 1 << 6;
        return listindices6(buck[t.slotid0()].attr, indices) || listindices6(buck[t.slotid1()].attr, indices + size) ||
               orderindices(indices, size) || indices[0] == indices[size];
    }
    __device__ bool listindices8(const tree t, u32 *indices) {
        const bucket1 &buck = hta.trees1[3][t.bucketid()];
        const u32 size = 1 << 7;
        return listindices7(buck[t.slotid0()].attr, indices) || listindices7(buck[t.slotid1()].attr, indices + size) ||
               orderindices(indices, size) || indices[0] == indices[size];
    }
    __device__ bool listindices9(const tree t, u32 *indices) {
        const bucket0 &buck = hta.trees0[4][t.bucketid()];
        const u32 size = 1 << 8;
        return listindices8(buck[t.slotid0()].attr, indices) || listindices8(buck[t.slotid1()].attr, indices + size) ||
               orderindices(indices, size) || indices[0] == indices[size];
    }
#endif
    __device__ void candidate(const tree t) {
        proof prf;
#if WK == 9
        if (listindices9(t, prf))
            return;
#elif WK == 5
        if (listindices5(t, prf))
            return;
#else
#error not implemented
#endif
        u32 soli = atomicAdd(&nsols, 1);
        if (soli < MAXSOLS)
#if WK == 9
            listindices9(t, sols[soli]);
#elif WK == 5
            listindices5(t, sols[soli]);
#else
#error not implemented
#endif
    }
    void showbsizes(u32 r) {
#if defined(HIST) || defined(SPARK) || defined(LOGSPARK)
        u32 ns[NBUCKETS];
        checkCudaErrors(cudaMemcpy(ns, nslots[r & 1], NBUCKETS * sizeof(u32), cudaMemcpyDeviceToHost));
        u32 binsizes[65];
        memset(binsizes, 0, 65 * sizeof(u32));
        for (u32 bucketid = 0; bucketid < NBUCKETS; bucketid++) {
            u32 bsize = min(ns[bucketid], NSLOTS) >> (SLOTBITS - 6);
            binsizes[bsize]++;
        }

        for (u32 i = 0; i < 65; i++) {
#ifdef HIST
            printf(" %d:%d", i, binsizes[i]);
#else
#ifdef SPARK
            u32 sparks = binsizes[i] / SPARKSCALE;
#else
            u32 sparks = 0;
            for (u32 bs = binsizes[i]; bs; bs >>= 1)
                sparks++;

            sparks = sparks * 7 / SPARKSCALE;
#endif
            printf("\342\226%c", '\201' + sparks);
#endif
        }
        printf("\n");
#endif
    }
    struct htlayout {
        htalloc hta;
        u32 prevhashunits;
        u32 nexthashunits;
        u32 dunits;
        u32 prevbo;
        u32 nextbo;

        __device__ htlayout(equi *eq, u32 r) : hta(eq->hta), prevhashunits(0), dunits(0) {
            u32 nexthashbytes = hashsize(r);
            nexthashunits = hashwords(nexthashbytes);
            prevbo = 0;
            nextbo = nexthashunits * sizeof(hashunit) - nexthashbytes; // 0-3
            if (r) {
                u32 prevhashbytes = hashsize(r - 1);
                prevhashunits = hashwords(prevhashbytes);
                prevbo = prevhashunits * sizeof(hashunit) - prevhashbytes; // 0-3
                dunits = prevhashunits - nexthashunits;
            }
        }
        __device__ u32 getxhash0(const slot0 *pslot) const {
#ifdef XINTREE
            return pslot->attr.xhash();
#elif DIGITBITS % 8 == 4 && RESTBITS == 4
            return pslot->hash->bytes[prevbo] >> 4;
#elif DIGITBITS % 8 == 4 && RESTBITS == 6
            return (pslot->hash->bytes[prevbo] & 0x3) << 4 | pslot->hash->bytes[prevbo + 1] >> 4;
#elif DIGITBITS % 8 == 4 && RESTBITS == 8
            return (pslot->hash->bytes[prevbo] & 0xf) << 4 | pslot->hash->bytes[prevbo + 1] >> 4;
#elif DIGITBITS % 8 == 4 && RESTBITS == 10
            return (pslot->hash->bytes[prevbo] & 0x3f) << 4 | pslot->hash->bytes[prevbo + 1] >> 4;
#elif DIGITBITS % 8 == 0 && RESTBITS == 4
            return pslot->hash->bytes[prevbo] & 0xf;
#elif RESTBITS == 0
            return 0;
#else
#error non implemented
#endif
        }
        __device__ u32 getxhash1(const slot1 *pslot) const {
#ifdef XINTREE
            return pslot->attr.xhash();
#elif DIGITBITS % 4 == 0 && RESTBITS == 4
            return pslot->hash->bytes[prevbo] & 0xf;
#elif DIGITBITS % 4 == 0 && RESTBITS == 6
            return pslot->hash->bytes[prevbo] & 0x3f;
#elif DIGITBITS % 4 == 0 && RESTBITS == 8
            return pslot->hash->bytes[prevbo];
#elif DIGITBITS % 4 == 0 && RESTBITS == 10
            return (pslot->hash->bytes[prevbo] & 0x3) << 8 | pslot->hash->bytes[prevbo + 1];
#elif RESTBITS == 0
            return 0;
#else
#error non implemented
#endif
        }
        __device__ bool equal(const hashunit *hash0, const hashunit *hash1) const {
            return hash0[prevhashunits - 1].word == hash1[prevhashunits - 1].word;
        }
    };

    struct collisiondata {
#ifdef XBITMAP
#if NSLOTS > 64
#error cant use XBITMAP with more than 64 slots
#endif
        u64 xhashmap[NRESTS];
        u64 xmap;
#else
#if RESTBITS <= 6
        typedef uchar xslot;
#else
        typedef u16 xslot;
#endif
        static const xslot xnil = ~0;
        xslot xhashslots[NRESTS];
        xslot nextxhashslot[NSLOTS];
        xslot nextslot;
#endif
        u32 s0;

        __device__ void clear() {
#ifdef XBITMAP
            memset(xhashmap, 0, NRESTS * sizeof(u64));
#else
            memset(xhashslots, xnil, NRESTS * sizeof(xslot));
            memset(nextxhashslot, xnil, NSLOTS * sizeof(xslot));
#endif
        }
        __device__ void addslot(u32 s1, u32 xh) {
#ifdef XBITMAP
            xmap = xhashmap[xh];
            xhashmap[xh] |= (u64)1 << s1;
            s0 = ~0;
#else
            nextslot = xhashslots[xh];
            nextxhashslot[s1] = nextslot;
            xhashslots[xh] = s1;
#endif
        }
        __device__ bool nextcollision() const {
#ifdef XBITMAP
            return xmap != 0;
#else
            return nextslot != xnil;
#endif
        }
        __device__ u32 slot() {
#ifdef XBITMAP
            const u32 ffs = __ffsll(xmap);
            s0 += ffs;
            xmap >>= ffs;
#else
            nextslot = nextxhashslot[s0 = nextslot];
#endif
            return s0;
        }
    };
};

__global__ void digitH(equi *eq) {
    uchar hash[HASHOUT];
    const u32 id = blockIdx.x * blockDim.x + threadIdx.x;
    // State that does not depend on the hash index is loaded once per thread
    // instead of once per hash (33M redundant global loads per solve).
    blake2b_pre pre;
    blake2b_precompute(&eq->blake_ctx, &pre);
    for (u32 block = id; block < NBLOCKS; block += eq->nthreads) {
        blake2b_gpu_hash_pre(&pre, block, hash);
        // Reserve all three bucket slots first: the atomics pipeline against
        // each other instead of each store waiting on its own atomic's
        // round-trip latency.
        u32 slots[HASHESPERBLAKE];
        u32 bucketids[HASHESPERBLAKE];
#pragma unroll
        for (u32 i = 0; i < HASHESPERBLAKE; i++) {
            const uchar *ph = hash + i * WN / 8;
#if BUCKBITS == 20 && RESTBITS == 4
            bucketids[i] = ((((u32)ph[0] << 8) | ph[1]) << 4) | ph[2] >> 4;
#else
#error not implemented
#endif
            slots[i] = atomicAdd(&eq->nslots[0][bucketids[i]], 1);
        }
#pragma unroll
        for (u32 i = 0; i < HASHESPERBLAKE; i++) {
            if (slots[i] >= NSLOTS)
                continue;
            const uchar *ph = hash + i * WN / 8;
            slot0 &s = eq->hta.trees0[0][bucketids[i]][slots[i]];
            s.attr = tree(block * HASHESPERBLAKE + i);
            memcpy(s.hash->bytes, ph + WN / 8 - 16, 16);
        }
    }
}

// Warp-cooperative collision round for (144,5): one warp per bucket. The
// whole bucket (64 slots x 5 words, contiguous) is staged into shared memory
// with coalesced loads, and all pair matching runs out of shared memory.
// Produces exactly the same pair set as the original per-thread linked-list
// version, so solutions are identical.
#if WN == 144 && BUCKBITS == 20 && RESTBITS == 4 && !defined(XINTREE)

// Warp-per-bucket collision round, fully specialized at compile time per
// round R: slot strides shrink as hash words are consumed (5,5,4,3 words in,
// 5,4,3,2 words out), and pair discovery uses a per-warp shared hashtable
// over the 16 rest-nibble values (djeZo's design): each slot inserts itself
// with one atomicExch and walks only its actual same-nibble predecessors.
// Insert and walk are separate phases so every chain link is written before
// any lane follows it. The pair set is identical to Tromp's per-thread
// linked-list version.
template <u32 R>
__global__ void digitRT(equi *eq) {
    static_assert(R >= 1 && R < WK, "collision round");
    constexpr u32 PREVUNITS = R <= 2 ? 4 : (R == 3 ? 3 : 2);
    constexpr u32 PREVBO = R == 1 ? 0 : (R == 2 ? 3 : (R == 3 ? 2 : 1));
    constexpr u32 DUNITS = R == 1 ? 0 : 1;
    // Cell stride is uniform (layers interleave in 5-word cells); only the
    // number of significant words changes per round.
    constexpr u32 INSTRIDE = 5;
    constexpr u32 OUTSTRIDE = 5;
    constexpr u32 INWORDS = 1 + PREVUNITS;
    constexpr bool ODD = (R & 1) != 0;

    __shared__ u32 sh[8][NSLOTS * INSTRIDE]; // 8 warps per 256-thread block
    __shared__ u32 shht[8][NRESTS];
    __shared__ u32 shnxt[8][NSLOTS];
    const u32 warp = threadIdx.x >> 5;
    u32 *mysh = sh[warp];
    u32 *ht = shht[warp];
    u32 *nxt = shnxt[warp];
    const u32 lane = threadIdx.x & 31;
    const u32 nwarps = (gridDim.x * blockDim.x) >> 5;
    const u32 warpId = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;

    const u32 *inlayer = ODD ? (const u32 *)eq->hta.trees0[(R - 1) / 2]
                             : (const u32 *)eq->hta.trees1[(R - 1) / 2];
    u32 *outlayer = ODD ? (u32 *)eq->hta.trees1[R / 2] : (u32 *)eq->hta.trees0[R / 2];
    u32 *nslotsIn = (u32 *)&eq->nslots[ODD ? 0 : 1][0];
    u32 *nslotsOut = (u32 *)&eq->nslots[ODD ? 1 : 0][0];

    for (u32 bucketid = warpId; bucketid < NBUCKETS; bucketid += nwarps) {
        u32 bsize = 0;
        if (lane == 0) {
            bsize = min(nslotsIn[bucketid], NSLOTS);
            nslotsIn[bucketid] = 0;
        }
        bsize = __shfl_sync(0xffffffff, bsize, 0);
        if (bsize == 0)
            continue;

        // stage the significant words of each slot cell through shared memory
        const u32 *gb = inlayer + bucketid * NSLOTS * INSTRIDE;
        for (u32 w = lane; w < bsize * INSTRIDE; w += 32) {
            if ((w % INSTRIDE) < INWORDS)
                mysh[w] = gb[w];
        }
        if (lane < NRESTS)
            ht[lane] = 0xffffffff;
        __syncwarp();

        for (u32 s = lane; s < bsize; s += 32) {
            const u32 x = (mysh[s * INSTRIDE + 1 + PREVBO / 4] >> (8 * (PREVBO & 3))) & 0xf;
            nxt[s] = atomicExch(&ht[x], s);
        }
        __syncwarp();

        for (u32 s1 = lane; s1 < bsize; s1 += 32) {
            const u32 *h1 = &mysh[s1 * INSTRIDE];
            for (u32 s0 = nxt[s1]; s0 != 0xffffffff; s0 = nxt[s0]) {
                const u32 *h0 = &mysh[s0 * INSTRIDE];
                // equal last significant word => likely duplicate, skip
                if (h0[PREVUNITS] == h1[PREVUNITS])
                    continue;

                const u32 xb1 = ((h0[1 + (PREVBO + 1) / 4] ^ h1[1 + (PREVBO + 1) / 4]) >> (8 * ((PREVBO + 1) & 3))) & 0xff;
                const u32 xb2 = ((h0[1 + (PREVBO + 2) / 4] ^ h1[1 + (PREVBO + 2) / 4]) >> (8 * ((PREVBO + 2) & 3))) & 0xff;
                const u32 xb3 = ((h0[1 + (PREVBO + 3) / 4] ^ h1[1 + (PREVBO + 3) / 4]) >> (8 * ((PREVBO + 3) & 3))) & 0xff;
                const u32 xorbucketid = (((xb1 << 8) | xb2) << 4) | (xb3 >> 4);

                const u32 xorslot = atomicAdd(&nslotsOut[xorbucketid], 1);
                if (xorslot >= NSLOTS)
                    continue;

                u32 *xs = outlayer + (xorbucketid * NSLOTS + xorslot) * OUTSTRIDE;
                xs[0] = tree(bucketid, s0, s1).bid_s0_s1_x;
#pragma unroll
                for (u32 i = DUNITS; i < PREVUNITS; i++)
                    xs[1 + i - DUNITS] = h0[1 + i] ^ h1[1 + i];
            }
        }
        __syncwarp();
    }
}

#else
#error warp-cooperative rounds are only implemented for (144,5) without XINTREE
#endif

#ifdef UNROLL
// bucket mask
static const u32 BUCKMASK = NBUCKETS - 1;

__global__ void digit_1(equi *eq) {
    equi::htlayout htl(eq, 1);
    equi::collisiondata cd;
    const u32 id = blockIdx.x * blockDim.x + threadIdx.x;
    for (u32 bucketid = id; bucketid < NBUCKETS; bucketid += eq->nthreads) {
        cd.clear();
        slot0 *buck = htl.hta.trees0[0][bucketid];
        u32 bsize = eq->getnslots0(bucketid);
        for (u32 s1 = 0; s1 < bsize; s1++) {
            const slot0 *pslot1 = buck + s1;
            for (cd.addslot(s1, htl.getxhash0(pslot1)); cd.nextcollision();) {
                const u32 s0 = cd.slot();
                const slot0 *pslot0 = buck + s0;
                if (htl.equal(pslot0->hash, pslot1->hash))
                    continue;

                const u32 xor0 = pslot0->hash->word ^ pslot1->hash->word;
                const u32 bexor = __byte_perm(xor0, 0, 0x0123);
                const u32 xorbucketid = bexor >> 4 & BUCKMASK;
                const u32 xhash = bexor & 0xf;
                const u32 xorslot = atomicAdd(&eq->nslots[1][xorbucketid], 1);
                if (xorslot >= NSLOTS)
                    continue;

                slot1 &xs = htl.hta.trees1[0][xorbucketid][xorslot];
                xs.attr = tree(bucketid, s0, s1, xhash);
                xs.hash[0].word = pslot0->hash[1].word ^ pslot1->hash[1].word;
                xs.hash[1].word = pslot0->hash[2].word ^ pslot1->hash[2].word;
                xs.hash[2].word = pslot0->hash[3].word ^ pslot1->hash[3].word;
                xs.hash[3].word = pslot0->hash[4].word ^ pslot1->hash[4].word;
                xs.hash[4].word = pslot0->hash[5].word ^ pslot1->hash[5].word;
            }
        }
    }
}

__global__ void digit2(equi *eq) {
    equi::htlayout htl(eq, 2);
    equi::collisiondata cd;
    const u32 id = blockIdx.x * blockDim.x + threadIdx.x;
    for (u32 bucketid = id; bucketid < NBUCKETS; bucketid += eq->nthreads) {
        cd.clear();
        slot1 *buck = htl.hta.trees1[0][bucketid];
        u32 bsize = eq->getnslots1(bucketid);
        for (u32 s1 = 0; s1 < bsize; s1++) {
            const slot1 *pslot1 = buck + s1;
            for (cd.addslot(s1, htl.getxhash1(pslot1)); cd.nextcollision();) {
                const u32 s0 = cd.slot();
                const slot1 *pslot0 = buck + s0;
                if (htl.equal(pslot0->hash, pslot1->hash))
                    continue;

                const u32 xor0 = pslot0->hash->word ^ pslot1->hash->word;
                const u32 bexor = __byte_perm(xor0, 0, 0x0123);
                const u32 xorbucketid = bexor >> 16;
                const u32 xhash = bexor >> 12 & 0xf;
                const u32 xorslot = atomicAdd(&eq->nslots[0][xorbucketid], 1);
                if (xorslot >= NSLOTS)
                    continue;

                slot0 &xs = htl.hta.trees0[1][xorbucketid][xorslot];
                xs.attr = tree(bucketid, s0, s1, xhash);
                xs.hash[0].word = xor0;
                xs.hash[1].word = pslot0->hash[1].word ^ pslot1->hash[1].word;
                xs.hash[2].word = pslot0->hash[2].word ^ pslot1->hash[2].word;
                xs.hash[3].word = pslot0->hash[3].word ^ pslot1->hash[3].word;
                xs.hash[4].word = pslot0->hash[4].word ^ pslot1->hash[4].word;
            }
        }
    }
}

__global__ void digit3(equi *eq) {
    equi::htlayout htl(eq, 3);
    equi::collisiondata cd;
    const u32 id = blockIdx.x * blockDim.x + threadIdx.x;
    for (u32 bucketid = id; bucketid < NBUCKETS; bucketid += eq->nthreads) {
        cd.clear();
        slot0 *buck = htl.hta.trees0[1][bucketid];
        u32 bsize = eq->getnslots0(bucketid);
        for (u32 s1 = 0; s1 < bsize; s1++) {
            const slot0 *pslot1 = buck + s1;
            for (cd.addslot(s1, htl.getxhash0(pslot1)); cd.nextcollision();) {
                const u32 s0 = cd.slot();
                const slot0 *pslot0 = buck + s0;
                if (htl.equal(pslot0->hash, pslot1->hash))
                    continue;

                const u32 xor0 = pslot0->hash->word ^ pslot1->hash->word;
                const u32 xor1 = pslot0->hash[1].word ^ pslot1->hash[1].word;
                const u32 bexor = __byte_perm(xor0, xor1, 0x1234);
                const u32 xorbucketid = bexor >> 4 & BUCKMASK;
                const u32 xhash = bexor & 0xf;
                const u32 xorslot = atomicAdd(&eq->nslots[1][xorbucketid], 1);
                if (xorslot >= NSLOTS)
                    continue;

                slot1 &xs = htl.hta.trees1[1][xorbucketid][xorslot];
                xs.attr = tree(bucketid, s0, s1, xhash);
                xs.hash[0].word = xor1;
                xs.hash[1].word = pslot0->hash[2].word ^ pslot1->hash[2].word;
                xs.hash[2].word = pslot0->hash[3].word ^ pslot1->hash[3].word;
                xs.hash[3].word = pslot0->hash[4].word ^ pslot1->hash[4].word;
            }
        }
    }
}

__global__ void digit4(equi *eq) {
    equi::htlayout htl(eq, 4);
    equi::collisiondata cd;
    const u32 id = blockIdx.x * blockDim.x + threadIdx.x;
    for (u32 bucketid = id; bucketid < NBUCKETS; bucketid += eq->nthreads) {
        cd.clear();
        slot1 *buck = htl.hta.trees1[1][bucketid];
        u32 bsize = eq->getnslots1(bucketid);
        for (u32 s1 = 0; s1 < bsize; s1++) {
            const slot1 *pslot1 = buck + s1;
            for (cd.addslot(s1, htl.getxhash1(pslot1)); cd.nextcollision();) {
                const u32 s0 = cd.slot();
                const slot1 *pslot0 = buck + s0;
                if (htl.equal(pslot0->hash, pslot1->hash))
                    continue;

                const u32 xor0 = pslot0->hash->word ^ pslot1->hash->word;
                const u32 bexor = __byte_perm(xor0, 0, 0x4123);
                const u32 xorbucketid = bexor >> 8;
                const u32 xhash = bexor >> 4 & 0xf;
                const u32 xorslot = atomicAdd(&eq->nslots[0][xorbucketid], 1);
                if (xorslot >= NSLOTS)
                    continue;

                slot0 &xs = htl.hta.trees0[2][xorbucketid][xorslot];
                xs.attr = tree(bucketid, s0, s1, xhash);
                xs.hash[0].word = xor0;
                xs.hash[1].word = pslot0->hash[1].word ^ pslot1->hash[1].word;
                xs.hash[2].word = pslot0->hash[2].word ^ pslot1->hash[2].word;
                xs.hash[3].word = pslot0->hash[3].word ^ pslot1->hash[3].word;
            }
        }
    }
}

__global__ void digit5(equi *eq) {
    equi::htlayout htl(eq, 5);
    equi::collisiondata cd;
    const u32 id = blockIdx.x * blockDim.x + threadIdx.x;
    for (u32 bucketid = id; bucketid < NBUCKETS; bucketid += eq->nthreads) {
        cd.clear();
        slot0 *buck = htl.hta.trees0[2][bucketid];
        u32 bsize = eq->getnslots0(bucketid);
        for (u32 s1 = 0; s1 < bsize; s1++) {
            const slot0 *pslot1 = buck + s1;
            for (cd.addslot(s1, htl.getxhash0(pslot1)); cd.nextcollision();) {
                const u32 s0 = cd.slot();
                const slot0 *pslot0 = buck + s0;
                if (htl.equal(pslot0->hash, pslot1->hash))
                    continue;

                const u32 xor0 = pslot0->hash->word ^ pslot1->hash->word;
                const u32 xor1 = pslot0->hash[1].word ^ pslot1->hash[1].word;
                const u32 bexor = __byte_perm(xor0, xor1, 0x2345);
                const u32 xorbucketid = bexor >> 4 & BUCKMASK;
                const u32 xhash = bexor & 0xf;
                const u32 xorslot = atomicAdd(&eq->nslots[1][xorbucketid], 1);
                if (xorslot >= NSLOTS)
                    continue;

                slot1 &xs = htl.hta.trees1[2][xorbucketid][xorslot];
                xs.attr = tree(bucketid, s0, s1, xhash);
                xs.hash[0].word = xor1;
                xs.hash[1].word = pslot0->hash[2].word ^ pslot1->hash[2].word;
                xs.hash[2].word = pslot0->hash[3].word ^ pslot1->hash[3].word;
            }
        }
    }
}

__global__ void digit6(equi *eq) {
    equi::htlayout htl(eq, 6);
    equi::collisiondata cd;
    const u32 id = blockIdx.x * blockDim.x + threadIdx.x;
    for (u32 bucketid = id; bucketid < NBUCKETS; bucketid += eq->nthreads) {
        cd.clear();
        slot1 *buck = htl.hta.trees1[2][bucketid];
        u32 bsize = eq->getnslots1(bucketid);
        for (u32 s1 = 0; s1 < bsize; s1++) {
            const slot1 *pslot1 = buck + s1;
            for (cd.addslot(s1, htl.getxhash1(pslot1)); cd.nextcollision();) {
                const u32 s0 = cd.slot();
                const slot1 *pslot0 = buck + s0;
                if (htl.equal(pslot0->hash, pslot1->hash))
                    continue;

                const u32 xor0 = pslot0->hash->word ^ pslot1->hash->word;
                const u32 xor1 = pslot0->hash[1].word ^ pslot1->hash[1].word;
                const u32 bexor = __byte_perm(xor0, xor1, 0x2345);
                const u32 xorbucketid = bexor >> 16;
                const u32 xhash = bexor >> 12 & 0xf;
                const u32 xorslot = atomicAdd(&eq->nslots[0][xorbucketid], 1);
                if (xorslot >= NSLOTS)
                    continue;

                slot0 &xs = htl.hta.trees0[3][xorbucketid][xorslot];
                xs.attr = tree(bucketid, s0, s1, xhash);
                xs.hash[0].word = xor1;
                xs.hash[1].word = pslot0->hash[2].word ^ pslot1->hash[2].word;
            }
        }
    }
}

__global__ void digit7(equi *eq) {
    equi::htlayout htl(eq, 7);
    equi::collisiondata cd;
    const u32 id = blockIdx.x * blockDim.x + threadIdx.x;
    for (u32 bucketid = id; bucketid < NBUCKETS; bucketid += eq->nthreads) {
        cd.clear();
        slot0 *buck = htl.hta.trees0[3][bucketid];
        u32 bsize = eq->getnslots0(bucketid);
        for (u32 s1 = 0; s1 < bsize; s1++) {
            const slot0 *pslot1 = buck + s1;
            for (cd.addslot(s1, htl.getxhash0(pslot1)); cd.nextcollision();) {
                const u32 s0 = cd.slot();
                const slot0 *pslot0 = buck + s0;
                if (htl.equal(pslot0->hash, pslot1->hash))
                    continue;

                const u32 xor0 = pslot0->hash->word ^ pslot1->hash->word;
                const u32 bexor = __byte_perm(xor0, 0, 0x4012);
                const u32 xorbucketid = bexor >> 4 & BUCKMASK;
                const u32 xhash = bexor & 0xf;
                const u32 xorslot = atomicAdd(&eq->nslots[1][xorbucketid], 1);
                if (xorslot >= NSLOTS)
                    continue;

                slot1 &xs = htl.hta.trees1[3][xorbucketid][xorslot];
                xs.attr = tree(bucketid, s0, s1, xhash);
                xs.hash[0].word = xor0;
                xs.hash[1].word = pslot0->hash[1].word ^ pslot1->hash[1].word;
            }
        }
    }
}

__global__ void digit8(equi *eq) {
    equi::htlayout htl(eq, 8);
    equi::collisiondata cd;
    const u32 id = blockIdx.x * blockDim.x + threadIdx.x;
    for (u32 bucketid = id; bucketid < NBUCKETS; bucketid += eq->nthreads) {
        cd.clear();
        slot1 *buck = htl.hta.trees1[3][bucketid];
        u32 bsize = eq->getnslots1(bucketid);
        for (u32 s1 = 0; s1 < bsize; s1++) {
            const slot1 *pslot1 = buck + s1;
            for (cd.addslot(s1, htl.getxhash1(pslot1)); cd.nextcollision();) {
                const u32 s0 = cd.slot();
                const slot1 *pslot0 = buck + s0;
                if (htl.equal(pslot0->hash, pslot1->hash))
                    continue;

                const u32 xor0 = pslot0->hash->word ^ pslot1->hash->word;
                const u32 xor1 = pslot0->hash[1].word ^ pslot1->hash[1].word;
                const u32 bexor = __byte_perm(xor0, xor1, 0x3456);
                const u32 xorbucketid = bexor >> 16;
                const u32 xhash = bexor >> 12 & 0xf;
                const u32 xorslot = atomicAdd(&eq->nslots[0][xorbucketid], 1);
                if (xorslot >= NSLOTS)
                    continue;

                slot0 &xs = htl.hta.trees0[4][xorbucketid][xorslot];
                xs.attr = tree(bucketid, s0, s1, xhash);
                xs.hash[0].word = xor1;
            }
        }
    }
}
#endif

__global__ void digitK(equi *eq) {
    equi::collisiondata cd;
    const u32 id = blockIdx.x * blockDim.x + threadIdx.x;
    for (u32 bucketid = id; bucketid < NBUCKETS; bucketid += eq->nthreads) {
        cd.clear();
        // final layer (trees0 L2): [attr, last hash word] within 5-word cells
        const u32 *buck = (const u32 *)eq->hta.trees0[(WK - 1) / 2] + bucketid * NSLOTS * 5;
        u32 bsize = eq->getnslots0(bucketid); // assume WK odd
        for (u32 s1 = 0; s1 < bsize; s1++) {
            const u32 w1 = buck[s1 * 5 + 1];
            for (cd.addslot(s1, w1 & 0xf); cd.nextcollision();) { // assume WK odd
                const u32 s0 = cd.slot();
                if (buck[s0 * 5 + 1] == w1 &&
                    tree(buck[s0 * 5]).prob_disjoint(tree(buck[s1 * 5]))) {
                    eq->candidate(tree(bucketid, s0, s1));
                }
            }
        }
    }
}

#if !defined(_WIN16) && !defined(_WIN32) && !defined(_WIN64) && !defined(__WINDOWS__)
#pragma clang diagnostic pop
#endif
